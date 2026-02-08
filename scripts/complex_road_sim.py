"""
Multi-agent simulation on a complex road network with:
  - A main 3-lane highway
  - An exit ramp branching off to the right
  - A T-junction where a side road merges from the left
  - A crosswalk near the junction
  - A stop sign at the side-road entry

Vehicles:
  0  SDC         – drives straight on the highway (IDM)
  1  Follower    – follows SDC in the same lane (IDM)
  2  Exit-taker  – moves to exit ramp (expert/log trajectory)
  3  Merger      – enters from side road and merges left (expert/log)
  4  Fast car    – constant 10 m/s in the left lane
  5  Slow car    – constant 3 m/s in the right lane
  6  Parked      – stationary on the shoulder
  7  Parked      – stationary on the shoulder
"""

import dataclasses, math

import jax
import mediapy
import numpy as np
from jax import numpy as jnp
from tqdm import tqdm
from waymax import agents
from waymax import config as _config
from waymax import datatypes, dynamics
from waymax import env as _env
from waymax import visualization

# ── road geometry helpers ──────────────────────────────────────────────────

def _straight_pts(x0, y0, x1, y1, n=120):
    """Return (n, 2) array of evenly spaced points + unit direction."""
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    ys = np.linspace(y0, y1, n, dtype=np.float32)
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy) or 1.0
    return xs, ys, np.float32(dx / length), np.float32(dy / length)


def _arc_pts(cx, cy, radius, angle_start, angle_end, n=80):
    """Return (n, 2) array of points along a circular arc + tangent dirs."""
    angles = np.linspace(angle_start, angle_end, n, dtype=np.float32)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    # tangent direction
    dx = -np.sin(angles) * np.sign(angle_end - angle_start)
    dy = np.cos(angles) * np.sign(angle_end - angle_start)
    return xs, ys, dx.astype(np.float32), dy.astype(np.float32)


# ── build scenario ────────────────────────────────────────────────────────

def create_complex_scenario(num_objects=8, num_timesteps=91):
    dt = 0.1  # 10 Hz, 9.1 seconds total

    # ── ROADGRAPH ──────────────────────────────────────────────────────────
    # We'll collect (x, y, dir_x, dir_y, type, id) per point, then pad to
    # the required 30 000 length.
    road_pts = []  # list of (x, y, dx, dy, type_id, elem_id)
    eid = 0  # element id counter

    # --- Main highway: x ∈ [-60, 160], 3 lanes at y = -4, 0, 4 -----------
    for lane_y in [-4.0, 0.0, 4.0]:
        eid += 1
        xs, ys, dx, dy = _straight_pts(-60, lane_y, 160, lane_y, n=200)
        for i in range(len(xs)):
            road_pts.append((xs[i], ys[i], dx, dy,
                             datatypes.MapElementIds.LANE_FREEWAY, eid))

    # --- Lane markings (broken white between lanes) -----------------------
    for mark_y in [-2.0, 2.0]:
        eid += 1
        xs, ys, dx, dy = _straight_pts(-60, mark_y, 160, mark_y, n=200)
        for i in range(len(xs)):
            road_pts.append((xs[i], ys[i], dx, dy,
                             datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE, eid))

    # --- Road edges (boundary) -- top & bottom of highway -----------------
    for edge_y, edge_x_end in [(-6.0, 160.0), (6.0, 70.0)]:
        eid += 1
        xs, ys, dx, dy = _straight_pts(-60, edge_y, edge_x_end, edge_y, n=160)
        for i in range(len(xs)):
            road_pts.append((xs[i], ys[i], dx, dy,
                             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY, eid))

    # Right edge continues after exit ramp opening (x > 90) ----------------
    eid += 1
    xs, ys, dx, dy = _straight_pts(90, 6.0, 160, 6.0, n=80)
    for i in range(len(xs)):
        road_pts.append((xs[i], ys[i], dx, dy,
                         datatypes.MapElementIds.ROAD_EDGE_BOUNDARY, eid))

    # --- Exit ramp: curves off to upper-right from x≈70, y=6 --------------
    # Curved section
    eid += 1
    xs, ys, dxs, dys = _arc_pts(70, 20, 14.0,
                                  -math.pi / 2, -math.pi / 6, n=80)
    for i in range(len(xs)):
        road_pts.append((xs[i], ys[i], dxs[i], dys[i],
                         datatypes.MapElementIds.LANE_SURFACE_STREET, eid))

    # Straight extension of exit ramp
    ramp_end_x = xs[-1] + 30 * math.cos(-math.pi / 6 + math.pi / 2)
    ramp_end_y = ys[-1] + 30 * math.sin(-math.pi / 6 + math.pi / 2)
    eid += 1
    xs2, ys2, dx2, dy2 = _straight_pts(float(xs[-1]), float(ys[-1]),
                                         float(ramp_end_x), float(ramp_end_y), n=60)
    for i in range(len(xs2)):
        road_pts.append((xs2[i], ys2[i], dx2, dy2,
                         datatypes.MapElementIds.LANE_SURFACE_STREET, eid))

    # Exit ramp edges
    for off in [-2.0, 2.0]:
        eid += 1
        cos_a = math.cos(-math.pi / 6 + math.pi / 2)
        sin_a = math.sin(-math.pi / 6 + math.pi / 2)
        perp_dx, perp_dy = -sin_a, cos_a
        ex0 = float(xs[-1]) + off * perp_dx
        ey0 = float(ys[-1]) + off * perp_dy
        ex1 = float(ramp_end_x) + off * perp_dx
        ey1 = float(ramp_end_y) + off * perp_dy
        xs3, ys3, dx3, dy3 = _straight_pts(ex0, ey0, ex1, ey1, n=40)
        for i in range(len(xs3)):
            road_pts.append((xs3[i], ys3[i], dx3, dy3,
                             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY, eid))

    # --- Side road (T-junction from the left at x≈20) ---------------------
    # Comes from y = -40 up to highway bottom edge y = -6
    eid += 1
    xs, ys, dx, dy = _straight_pts(20, -45, 20, -6, n=100)
    for i in range(len(xs)):
        road_pts.append((xs[i], ys[i], dx, dy,
                         datatypes.MapElementIds.LANE_SURFACE_STREET, eid))

    # Side road edges
    for side_x in [17.0, 23.0]:
        eid += 1
        xs, ys, dx, dy = _straight_pts(side_x, -45, side_x, -6, n=80)
        for i in range(len(xs)):
            road_pts.append((xs[i], ys[i], dx, dy,
                             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY, eid))

    # --- Crosswalk across the highway near the junction (x ≈ 25) ----------
    eid += 1
    xs, ys, dx, dy = _straight_pts(24, -6, 24, 6, n=40)
    for i in range(len(xs)):
        road_pts.append((xs[i], ys[i], dx, dy,
                         datatypes.MapElementIds.CROSSWALK, eid))
    eid += 1
    xs, ys, dx, dy = _straight_pts(26, -6, 26, 6, n=40)
    for i in range(len(xs)):
        road_pts.append((xs[i], ys[i], dx, dy,
                         datatypes.MapElementIds.CROSSWALK, eid))

    # --- Stop sign at side-road entry (x=20, y=-7) ------------------------
    eid += 1
    # Just a cluster of points forming the stop sign marker
    for sx in np.linspace(19, 21, 8):
        for sy in np.linspace(-8, -6.5, 8):
            road_pts.append((np.float32(sx), np.float32(sy),
                             np.float32(0), np.float32(1),
                             datatypes.MapElementIds.STOP_SIGN, eid))

    # ── pack into arrays (pad to 30 000) ──────────────────────────────────
    num_road_pts = 30000
    rg_x = np.zeros(num_road_pts, dtype=np.float32)
    rg_y = np.zeros(num_road_pts, dtype=np.float32)
    rg_z = np.zeros(num_road_pts, dtype=np.float32)
    rg_dir_x = np.zeros(num_road_pts, dtype=np.float32)
    rg_dir_y = np.zeros(num_road_pts, dtype=np.float32)
    rg_dir_z = np.zeros(num_road_pts, dtype=np.float32)
    rg_types = np.full(num_road_pts, -1, dtype=np.int32)
    rg_ids = np.zeros(num_road_pts, dtype=np.int32)
    rg_valid = np.zeros(num_road_pts, dtype=bool)

    n_pts = min(len(road_pts), num_road_pts)
    for i in range(n_pts):
        px, py, pdx, pdy, ptype, pid = road_pts[i]
        rg_x[i] = px
        rg_y[i] = py
        rg_dir_x[i] = pdx
        rg_dir_y[i] = pdy
        rg_types[i] = int(ptype)
        rg_ids[i] = pid
        rg_valid[i] = True

    roadgraph = datatypes.RoadgraphPoints(
        x=jnp.array(rg_x), y=jnp.array(rg_y), z=jnp.array(rg_z),
        dir_x=jnp.array(rg_dir_x), dir_y=jnp.array(rg_dir_y),
        dir_z=jnp.array(rg_dir_z),
        types=jnp.array(rg_types), ids=jnp.array(rg_ids),
        valid=jnp.array(rg_valid),
    )

    # ── TRAJECTORIES ──────────────────────────────────────────────────────
    x   = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    y   = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    vx  = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    vy  = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    yaw = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    valid = np.ones((num_objects, num_timesteps), dtype=bool)

    ts = np.arange(num_timesteps) * dt  # time array

    # 0 – SDC: straight along centre lane y=0
    x[0] = -30 + 8.0 * ts
    y[0] = 0.0
    vx[0] = 8.0
    yaw[0] = 0.0

    # 1 – Follower: same lane, starts behind SDC
    x[1] = -45 + 7.0 * ts
    y[1] = 0.0
    vx[1] = 7.0
    yaw[1] = 0.0

    # 2 – Exit-taker: starts in right lane, follows exit ramp arc exactly.
    # Ramp arc: centre (70, 20), radius 14, from angle -π/2 to -π/6.
    # Ramp start = (70, 6), ramp end = (82.1, 13).
    ramp_cx, ramp_cy, ramp_R = 70.0, 20.0, 14.0
    ramp_a0, ramp_a1 = -math.pi / 2, -math.pi / 6  # arc angular range
    ramp_arc_len = ramp_R * abs(ramp_a1 - ramp_a0)  # ≈14.66 m
    exit_speed = 10.0
    x2_start = 20.0
    # Phase 1: cruise in right lane y=4
    # Phase 2: smooth sinusoidal lane-change from y=4→6 over 2 seconds
    #          (starts at t=3s, ends at t=5s when x=70)
    t_lc_start = 3.0    # begin lane change
    t_lc_dur = 2.0      # lane change duration
    t_ramp_entry = t_lc_start + t_lc_dur  # = 5.0 s, x = 70
    dy_shift = 2.0       # y shift: 4 → 6
    # Phase 3: follow arc at constant speed
    ramp_angular_vel = exit_speed / ramp_R  # rad/s on the arc
    t_arc = ramp_arc_len / exit_speed  # ≈1.47 s
    # Phase 4: straight along ramp extension (same heading as arc end)
    ramp_exit_heading = ramp_a1 + math.pi / 2  # tangent at arc end
    for t in range(num_timesteps):
        tt = ts[t]
        if tt < t_lc_start:
            # Phase 1: cruise in right lane
            x[2, t] = x2_start + exit_speed * tt
            y[2, t] = 4.0
            vx[2, t] = exit_speed
            yaw[2, t] = 0.0
        elif tt < t_ramp_entry:
            # Phase 2: smooth sinusoidal lane-change (zero vy at both ends)
            frac = (tt - t_lc_start) / t_lc_dur
            x[2, t] = x2_start + exit_speed * tt
            # sinusoidal profile: y = 4 + dy*(frac - sin(2π·frac)/(2π))
            y[2, t] = 4.0 + dy_shift * (frac - math.sin(2 * math.pi * frac) / (2 * math.pi))
            vx[2, t] = exit_speed
            vy[2, t] = (dy_shift / t_lc_dur) * (1 - math.cos(2 * math.pi * frac))
            yaw[2, t] = math.atan2(vy[2, t], vx[2, t])
        elif tt < t_ramp_entry + t_arc:
            # Phase 3: follow the arc exactly
            dt2 = tt - t_ramp_entry
            theta = ramp_a0 + ramp_angular_vel * dt2
            x[2, t] = ramp_cx + ramp_R * math.cos(theta)
            y[2, t] = ramp_cy + ramp_R * math.sin(theta)
            # tangent = perpendicular to radius, CCW
            heading = theta + math.pi / 2
            vx[2, t] = exit_speed * math.cos(heading)
            vy[2, t] = exit_speed * math.sin(heading)
            yaw[2, t] = heading
        else:
            # Phase 4: straight along ramp extension
            dt2 = tt - (t_ramp_entry + t_arc)
            arc_end_x = ramp_cx + ramp_R * math.cos(ramp_a1)
            arc_end_y = ramp_cy + ramp_R * math.sin(ramp_a1)
            x[2, t] = arc_end_x + exit_speed * math.cos(ramp_exit_heading) * dt2
            y[2, t] = arc_end_y + exit_speed * math.sin(ramp_exit_heading) * dt2
            vx[2, t] = exit_speed * math.cos(ramp_exit_heading)
            vy[2, t] = exit_speed * math.sin(ramp_exit_heading)
            yaw[2, t] = ramp_exit_heading

    # 3 – Merger: comes from side road (x=20, y=-40) and merges onto highway
    # Uses a circular arc to turn from northbound to eastbound, staying in
    # the left lane y=-4 and well clear of the centre lane y=0.
    turn_radius = 10.0  # tight enough to stay near x≈20
    turn_cx = 20.0 + turn_radius  # arc centre at (30, -6)
    turn_cy = -6.0  # at highway bottom edge
    # Arc from θ=π (pointing left=north approach) to θ=3π/2 (pointing down=east exit)
    # i.e. vehicle goes from (20, -6) heading north → sweeps to (30, -16)… no.
    # Actually: car arrives at (20, -6) heading north (yaw=π/2).
    # Turn right: arc centre is to the right of the car = (30, -6).
    # Arc from angle=π (west, giving pos (20,-6)) to angle=3π/2 (south, giving (30,-16)).
    # But we want to end heading east at y≈-4, so let's use a simpler approach:
    # Arc centre at (20, -6-turn_radius) doesn't work either. Let me just
    # parameterise it cleanly:
    #   Phase 1: drive north on side road to (20, -10)
    #   Phase 2: quarter-circle arc turning east, ending at (30, -4) heading east
    #   Phase 3: cruise east at y=-4
    t_arrive = 3.0      # reach junction entrance at t=3s
    y_junction = -10.0   # y where the turn begins
    t_turn = 2.0         # seconds for the 90° turn
    merge_speed = 6.0
    for t in range(num_timesteps):
        tt = ts[t]
        if tt < t_arrive:
            # driving north on side road
            x[3, t] = 20.0
            y[3, t] = -40 + ((-10 - (-40)) / t_arrive) * tt  # → y=-10 at t=3
            vx[3, t] = 0.0
            vy[3, t] = merge_speed
            yaw[3, t] = math.pi / 2
        elif tt < t_arrive + t_turn:
            # quarter-circle: centre at (20 + R, -10), radius R
            # angle goes from π (start) to π/2 (end, pointing east)
            R = 6.0  # turn radius – ends at y = -10 + 0 = -10? No:
            # centre (26, -10), start angle π → pos (20, -10), end angle π/2 → pos (26, -4)
            dt2 = tt - t_arrive
            frac = dt2 / t_turn
            theta = math.pi * (1.0 - 0.5 * frac)  # π → π/2
            cx, cy = 26.0, -10.0
            R = 6.0
            x[3, t] = cx + R * math.cos(theta)
            y[3, t] = cy + R * math.sin(theta)
            # heading = tangent direction
            heading = theta - math.pi / 2  # tangent for CCW
            yaw[3, t] = heading
            arc_speed = merge_speed
            vx[3, t] = arc_speed * math.cos(heading)
            vy[3, t] = arc_speed * math.sin(heading)
        else:
            # cruising east on highway left lane y=-4
            dt2 = tt - (t_arrive + t_turn)
            x[3, t] = 26.0 + merge_speed * dt2  # starts at end of arc x=26
            y[3, t] = -4.0
            vx[3, t] = merge_speed
            vy[3, t] = 0.0
            yaw[3, t] = 0.0

    # 4 – Fast car: left lane y=-4, constant 10 m/s
    x[4] = -50 + 10.0 * ts
    y[4] = -4.0
    vx[4] = 10.0
    yaw[4] = 0.0

    # 5 – Slow car: right lane y=4, constant 3 m/s (starts well ahead of agent 2)
    x[5] = 80 + 3.0 * ts
    y[5] = 4.0
    vx[5] = 3.0
    yaw[5] = 0.0

    # 6, 7 – Parked on shoulder
    x[6] = 120.0; y[6] = 7.0; valid[6] = True
    x[7] = 135.0; y[7] = 7.0; valid[7] = True

    z = np.zeros_like(x)
    timestamp_micros = (np.arange(num_timesteps) * int(dt * 1e6))[None, :] \
                       .repeat(num_objects, axis=0).astype(np.int32)
    length = np.full((num_objects, num_timesteps), 4.5, dtype=np.float32)
    width  = np.full((num_objects, num_timesteps), 2.0, dtype=np.float32)
    height = np.full((num_objects, num_timesteps), 1.5, dtype=np.float32)

    traj = datatypes.Trajectory(
        x=jnp.array(x), y=jnp.array(y), z=jnp.array(z),
        vel_x=jnp.array(vx), vel_y=jnp.array(vy), yaw=jnp.array(yaw),
        valid=jnp.array(valid),
        timestamp_micros=jnp.array(timestamp_micros),
        length=jnp.array(length), width=jnp.array(width),
        height=jnp.array(height),
    )

    # ── OBJECT METADATA ───────────────────────────────────────────────────
    obj_types = np.ones(num_objects, dtype=np.int32)  # 1 = vehicle
    is_sdc = np.zeros(num_objects, dtype=bool); is_sdc[0] = True
    is_valid = np.ones(num_objects, dtype=bool)

    metadata = datatypes.ObjectMetadata(
        ids=jnp.arange(num_objects, dtype=jnp.int32),
        object_types=jnp.array(obj_types),
        is_sdc=jnp.array(is_sdc),
        is_modeled=jnp.array(is_valid),
        is_valid=jnp.array(is_valid),
        objects_of_interest=jnp.array(is_sdc),
        is_controlled=jnp.zeros(num_objects, dtype=bool),
    )

    # ── TRAFFIC LIGHTS (unused) ───────────────────────────────────────────
    num_tl = 16
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((num_tl, num_timesteps)),
        y=jnp.zeros((num_tl, num_timesteps)),
        z=jnp.zeros((num_tl, num_timesteps)),
        state=jnp.zeros((num_tl, num_timesteps), dtype=jnp.int32),
        lane_ids=jnp.zeros((num_tl, num_timesteps), dtype=jnp.int32),
        valid=jnp.zeros((num_tl, num_timesteps), dtype=bool),
    )

    return datatypes.SimulatorState(
        sim_trajectory=traj,
        log_trajectory=traj,
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=jnp.int32(0),
        roadgraph_points=roadgraph,
        sdc_paths=None,
    )


# ── main ──────────────────────────────────────────────────────────────────

num_objects = 8
scenario = create_complex_scenario(num_objects=num_objects)
print(f"Scenario: {num_objects} objects, "
      f"{scenario.log_trajectory.x.shape[1]} timesteps, "
      f"road pts = {int(scenario.roadgraph_points.valid.sum())}")

dynamics_model = dynamics.StateDynamics()

env = _env.BaseEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=num_objects,
        controlled_object=_config.ObjectType.VALID,
    ),
)

obj_idx = jnp.arange(num_objects)

# Parked + slow shoulder cars
static_actor = agents.create_constant_speed_actor(
    speed=0.0,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: (obj_idx == 6) | (obj_idx == 7),
)

# IDM for SDC (0) and follower (1)
idm_actor = agents.IDMRoutePolicy(
    is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1),
)

# Expert (log-replay) for exit-taker (2) and merger (3)
expert_actor = agents.create_expert_actor(
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: (obj_idx == 2) | (obj_idx == 3),
)

# Constant-speed actors for fast car (4) and slow car (5)
fast_actor = agents.create_constant_speed_actor(
    speed=10.0,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx == 4,
)
slow_actor = agents.create_constant_speed_actor(
    speed=3.0,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx == 5,
)

actors = [static_actor, idm_actor, expert_actor, fast_actor, slow_actor]

# ── simulate ──────────────────────────────────────────────────────────────
jit_step = jax.jit(env.step)
jit_actions = [jax.jit(a.select_action) for a in actors]

states = [env.reset(scenario)]
total_steps = int(states[0].remaining_timesteps)
print(f"Simulating {total_steps} steps …")

for _ in tqdm(range(total_steps)):
    s = states[-1]
    outputs = [jit_act({}, s, None, None) for jit_act in jit_actions]
    action = agents.merge_actions(outputs)
    states.append(jit_step(s, action))

# ── render ────────────────────────────────────────────────────────────────
print("Rendering frames …")

# Use a wider viewport so the junction + exit ramp are visible
viz_cfg = {"front_x": 100, "back_x": 60, "front_y": 60, "back_y": 60}

imgs = []
for s in tqdm(states):
    imgs.append(visualization.plot_simulator_state(
        s, use_log_traj=False, viz_config=viz_cfg))

mediapy.write_video("complex_sim.mp4", imgs, fps=10)
print("Video saved as complex_sim.mp4")
