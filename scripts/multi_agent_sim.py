import dataclasses

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
from waymax.utils.test_utils import make_zeros_state


# ---------------------------------------------------------------------------
# Helper: build a synthetic scenario with vehicles on a straight road
# ---------------------------------------------------------------------------
def create_synthetic_scenario(num_objects=8, num_timesteps=91):
    """Create a SimulatorState with vehicles spread along a road so the
    simulation is visually meaningful even without real WOMD data."""

    # -- trajectories (num_objects, num_timesteps) --------------------------
    # Place vehicles in a line along the x-axis, each with a small forward
    # velocity so actors can modify their behaviour visibly.
    dt = 0.1  # 10 Hz

    x = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    y = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    vel_x = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    vel_y = np.zeros((num_objects, num_timesteps), dtype=np.float32)
    yaw = np.zeros((num_objects, num_timesteps), dtype=np.float32)

    # Starting x positions — manually assigned to avoid any overlap.
    # Moving vehicles (0-4) are spaced in the back; parked cars (5-7) are
    # placed far ahead (x >= 160 m) so no moving vehicle can reach them.
    start_x = np.zeros(num_objects, dtype=np.float32)
    start_x[0] = 0.0     # SDC
    start_x[1] = 15.0    # IDM follower
    start_x[2] = 30.0    # constant-speed
    start_x[3] = 10.0    # expert 1
    start_x[4] = 25.0    # expert 2
    start_x[5] = 160.0   # parked
    start_x[6] = 175.0   # parked
    start_x[7] = 190.0   # parked

    # Stagger lanes so objects are not all in a single file
    lane_y = np.zeros(num_objects, dtype=np.float32)
    lane_y[0] = 0.0    # SDC – centre lane
    lane_y[1] = 0.0    # IDM follower – same lane as SDC
    lane_y[2] = 4.0    # constant-speed – right lane
    lane_y[3] = -4.0   # expert actor 1 – left lane
    lane_y[4] = 4.0    # expert actor 2 – right lane
    # Parked cars on road lanes (safe — far ahead of all moving vehicles)
    lane_y[5] = 0.0    # centre lane
    lane_y[6] = 4.0    # right lane
    lane_y[7] = -4.0   # left lane

    # Give each object a constant-speed log trajectory so expert/log actors
    # have something to replay, and IDM has a leader to follow.
    speeds = np.zeros(num_objects, dtype=np.float32)
    speeds[0] = 8.0   # SDC drives forward
    speeds[1] = 6.0   # IDM follower, slightly slower initially
    speeds[2] = 5.0   # constant-speed actor
    speeds[3] = 7.0   # expert 1
    speeds[4] = 4.0   # expert 2
    # Objects 5+ stay parked (speed = 0)

    for t in range(num_timesteps):
        x[:, t] = start_x + speeds * t * dt
        y[:, t] = lane_y
        vel_x[:, t] = speeds
        # All vehicles face positive-x direction (yaw = 0)

    z = np.zeros_like(x)
    valid = np.ones((num_objects, num_timesteps), dtype=bool)
    timestamp_micros = (np.arange(num_timesteps) * int(dt * 1e6))[None, :].repeat(num_objects, axis=0).astype(np.int32)

    # Vehicle dimensions (typical car)
    length = np.full((num_objects, num_timesteps), 4.5, dtype=np.float32)
    width  = np.full((num_objects, num_timesteps), 2.0, dtype=np.float32)
    height = np.full((num_objects, num_timesteps), 1.5, dtype=np.float32)

    traj = datatypes.Trajectory(
        x=jnp.array(x), y=jnp.array(y), z=jnp.array(z),
        vel_x=jnp.array(vel_x), vel_y=jnp.array(vel_y),
        yaw=jnp.array(yaw), valid=jnp.array(valid),
        timestamp_micros=jnp.array(timestamp_micros),
        length=jnp.array(length), width=jnp.array(width),
        height=jnp.array(height),
    )

    # -- object metadata ----------------------------------------------------
    obj_types = np.ones(num_objects, dtype=np.int32)  # 1 = vehicle
    is_sdc = np.zeros(num_objects, dtype=bool)
    is_sdc[0] = True
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

    # -- roadgraph: two straight lane centre-lines + edges ------------------
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

    # Points along each lane / edge
    pts_per_element = 200
    road_x_range = np.linspace(-20, 200, pts_per_element, dtype=np.float32)

    elements = [
        # (y-offset, type, element_id)
        (-2.0, datatypes.MapElementIds.ROAD_EDGE_BOUNDARY, 1),
        (0.0,  datatypes.MapElementIds.LANE_FREEWAY, 2),          # centre lane
        (2.0,  datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE, 3),
        (4.0,  datatypes.MapElementIds.LANE_FREEWAY, 4),          # right lane
        (6.0,  datatypes.MapElementIds.ROAD_EDGE_BOUNDARY, 5),
        (-4.0, datatypes.MapElementIds.LANE_FREEWAY, 6),          # left lane
        (-6.0, datatypes.MapElementIds.ROAD_EDGE_BOUNDARY, 7),
        (-2.0, datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE, 8),
    ]

    idx = 0
    for y_off, etype, eid in elements:
        end = idx + pts_per_element
        if end > num_road_pts:
            break
        rg_x[idx:end] = road_x_range
        rg_y[idx:end] = y_off
        rg_dir_x[idx:end] = 1.0  # road direction = positive x
        rg_types[idx:end] = int(etype)
        rg_ids[idx:end] = eid
        rg_valid[idx:end] = True
        idx = end

    roadgraph = datatypes.RoadgraphPoints(
        x=jnp.array(rg_x), y=jnp.array(rg_y), z=jnp.array(rg_z),
        dir_x=jnp.array(rg_dir_x), dir_y=jnp.array(rg_dir_y),
        dir_z=jnp.array(rg_dir_z),
        types=jnp.array(rg_types), ids=jnp.array(rg_ids),
        valid=jnp.array(rg_valid),
    )

    # -- traffic lights (unused, keep zeros) --------------------------------
    num_tl = 16
    traffic_lights = datatypes.TrafficLights(
        x=jnp.zeros((num_tl, num_timesteps)),
        y=jnp.zeros((num_tl, num_timesteps)),
        z=jnp.zeros((num_tl, num_timesteps)),
        state=jnp.zeros((num_tl, num_timesteps), dtype=jnp.int32),
        lane_ids=jnp.zeros((num_tl, num_timesteps), dtype=jnp.int32),
        valid=jnp.zeros((num_tl, num_timesteps), dtype=bool),
    )

    # -- assemble SimulatorState -------------------------------------------
    state = datatypes.SimulatorState(
        sim_trajectory=traj,
        log_trajectory=traj,       # log == sim at the start
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=jnp.int32(0),
        roadgraph_points=roadgraph,
        sdc_paths=None,
    )
    return state


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
max_num_objects = 8  # keep small so the scene is readable

scenario = create_synthetic_scenario(num_objects=max_num_objects)
print(f"Scenario created: {max_num_objects} objects, "
      f"{scenario.log_trajectory.x.shape[1]} timesteps")

# Environment
dynamics_model = dynamics.StateDynamics()
env = _env.BaseEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        _config.EnvironmentConfig(),
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.VALID,
    ),
)

# ---------------------------------------------------------------------------
# Actors – each policy controls a subset of objects
# ---------------------------------------------------------------------------
obj_idx = jnp.arange(max_num_objects)

# Objects 5-7: parked cars (zero speed)
static_actor = agents.create_constant_speed_actor(
    speed=0.0,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx > 4,
)

# Objects 0 & 1: IDM car-following policy
actor_0 = agents.IDMRoutePolicy(
    is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1)
)

# Object 2: constant speed (5 m/s)
actor_1 = agents.create_constant_speed_actor(
    speed=5.0,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx == 2,
)

# Objects 3 & 4: replay logged (expert) trajectory
actor_2 = agents.create_expert_actor(
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: (obj_idx == 3) | (obj_idx == 4),
)

actors = [static_actor, actor_0, actor_1, actor_2]

# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
jit_step = jax.jit(env.step)
jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

states = [env.reset(scenario)]
total_steps = int(states[0].remaining_timesteps)
print(f"Simulating {total_steps} steps …")

for i in tqdm(range(total_steps)):
    current_state = states[-1]
    outputs = [
        jit_select_action({}, current_state, None, None)
        for jit_select_action in jit_select_action_list
    ]
    action = agents.merge_actions(outputs)
    next_state = jit_step(current_state, action)
    states.append(next_state)

# ---------------------------------------------------------------------------
# Render video
# ---------------------------------------------------------------------------
print("Rendering frames …")
imgs = []
for state in tqdm(states):
    imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))

mediapy.write_video("simulation.mp4", imgs, fps=10)
print("Video saved as simulation.mp4")
