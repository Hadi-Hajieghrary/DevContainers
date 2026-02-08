"""
City-grid multi-agent simulation with complex road structures (Highway, Exit Ramp, Roundabout) and Ego planning.

Road network
  A modified 6×6 city grid.
  - Avenue 0 (West-most) is replaced by a high-speed **Highway**.
  - An **Exit Ramp** connects the Highway to the city grid at intersection (2, 1).
  - Intersection (3, 3) is replaced by a **Roundabout**.
  - A diagonal "Express Avenue" cuts from (1,1) to (4,4).
  - Explicit **Goal Area** visualization at the target.

Vehicles (16 total)
  Object 0 — **Ego** : Starts on Highway, takes the exit ramp, navigates the roundabout, and parks at the goal.
             Uses IDM (Intelligent Driver Model) for collision avoidance.
  Objects 1-11 — Background traffic.
  Objects 12-15 — Parked vehicles.

Output
  city_planning_sim.mp4  (≈ 20 s @ 10 fps, 201 timesteps)
"""

from __future__ import annotations

import dataclasses
import heapq
import math
from typing import Dict, List, Tuple

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
from waymax.visualization import utils as viz_utils  # Fix for VizConfig

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
BLOCK = 50.0  # metres per city block
N_EW = 6  # number of east-west streets (rows)
N_NS = 6  # number of north-south avenues (columns)
LANE_W = 4.0  # lane width in metres
DT = 0.1  # timestep (10 Hz)
N_TIMESTEPS = 220  # Trajectory capacity
SIM_STEPS = 200  # Actual simulation steps
NUM_OBJECTS = 16  # total vehicles


# ═══════════════════════════════════════════════════════════════════════════
# Road Building Primitives
# ═══════════════════════════════════════════════════════════════════════════
def _pts_line(x0, y0, x1, y1, n=80):
    """Straight road points."""
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    ys = np.linspace(y0, y1, n, dtype=np.float32)
    dx, dy = float(x1 - x0), float(y1 - y0)
    L = math.hypot(dx, dy) or 1.0
    return xs, ys, np.float32(dx / L), np.float32(dy / L)


def _pts_arc(cx, cy, r, ang_start, ang_end, n=50):
    """Circular arc road points."""
    thetas = np.linspace(ang_start, ang_end, n, dtype=np.float32)
    xs = cx + r * np.cos(thetas)
    ys = cy + r * np.sin(thetas)
    # Directions tangent to circle (-sin, cos) * sign(ang_change)
    direction = 1.0 if ang_end > ang_start else -1.0
    dxs = -np.sin(thetas) * direction
    dys = np.cos(thetas) * direction
    return (
        xs.astype(np.float32),
        ys.astype(np.float32),
        dxs.astype(np.float32),
        dys.astype(np.float32),
    )


def build_road_network(goal_pos: Tuple[float, float] = None):
    pts: list[tuple] = []
    eid = 0

    def _add(xs, ys, dx, dy, typ):
        nonlocal eid
        eid += 1
        for i in range(len(xs)):
            ddx = dx if np.isscalar(dx) else dx[i]
            ddy = dy if np.isscalar(dy) else dy[i]
            pts.append((xs[i], ys[i], ddx, ddy, int(typ), eid))

    # ── 1. The Grid (Modified) ────────────────────────────────────────────
    # We skip Col 0 (Highway) and (3,3) (Roundabout) for normal generation
    skip_cols = [0]
    roundabout_node = (3, 3)  # Row 3, Col 3

    # -- East-West Streets --
    for row in range(N_EW):
        cy = row * BLOCK
        x0, x1 = -10.0, (N_NS - 1) * BLOCK + 10.0
        # Drawing regular streets, breaking for roundabout
        segs = []
        if row == roundabout_node[0]:
            # Break at col 3
            segs.append((-10.0, (roundabout_node[1] - 0.6) * BLOCK))
            segs.append(((roundabout_node[1] + 0.6) * BLOCK, (N_NS - 1) * BLOCK + 10))
        else:
            segs.append((-10.0, (N_NS - 1) * BLOCK + 10))

        for x0, x1 in segs:
            # Lanes
            if abs(x1 - x0) < 1.0:
                continue
            for offset, direction in [(LANE_W / 2, 1), (-LANE_W / 2, -1)]:
                xs, ys, dx, dy = _pts_line(
                    x0, cy + offset, x1, cy + offset, n=int(abs(x1 - x0) / 2)
                )
                if direction == -1:
                    xs, dx = xs[::-1], -dx
                _add(xs, ys, dx, dy, datatypes.MapElementIds.LANE_SURFACE_STREET)
            # Edges
            for edge_off in [LANE_W, -LANE_W]:
                xs, ys, dx, dy = _pts_line(
                    x0, cy + edge_off, x1, cy + edge_off, n=int(abs(x1 - x0) / 2)
                )
                _add(xs, ys, dx, dy, datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)

    # -- North-South Avenues --
    for col in range(N_NS):
        if col in skip_cols:
            continue
        cx = col * BLOCK
        y0, y1 = -10.0, (N_EW - 1) * BLOCK + 10.0

        segs = []
        if col == roundabout_node[1]:
            # Break at row 3
            segs.append((-10.0, (roundabout_node[0] - 0.6) * BLOCK))
            segs.append(((roundabout_node[0] + 0.6) * BLOCK, (N_EW - 1) * BLOCK + 10))
        else:
            segs.append((-10.0, (N_EW - 1) * BLOCK + 10))

        for y0, y1 in segs:
            # Lanes
            if abs(y1 - y0) < 1.0:
                continue
            for offset, direction in [(LANE_W / 2, 1), (-LANE_W / 2, -1)]:
                xs, ys, dx, dy = _pts_line(
                    cx + offset, y0, cx + offset, y1, n=int(abs(y1 - y0) / 2)
                )
                if direction == -1:
                    ys, dy = ys[::-1], -dy
                _add(xs, ys, dx, dy, datatypes.MapElementIds.LANE_SURFACE_STREET)
            # Edges
            for edge_off in [LANE_W, -LANE_W]:
                xs, ys, dx, dy = _pts_line(
                    cx + edge_off, y0, cx + edge_off, y1, n=int(abs(y1 - y0) / 2)
                )
                _add(xs, ys, dx, dy, datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)

    # ── 2. Highway (Col 0 replacement) ─────────────────────────────────────
    # A generic high-speed road at x = -20 (shifted left of grid col 0)
    hw_x = -20.0
    hw_y0, hw_y1 = -50.0, N_EW * BLOCK + 50.0

    # 3 Lanes Northbound
    for i in range(3):
        off = i * LANE_W
        xs, ys, dx, dy = _pts_line(hw_x + off, hw_y0, hw_x + off, hw_y1, n=200)
        _add(xs, ys, dx, dy, datatypes.MapElementIds.LANE_FREEWAY)
    # Edges
    _add(
        *_pts_line(hw_x - LANE_W / 2, hw_y0, hw_x - LANE_W / 2, hw_y1, n=150),
        datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
    )
    _add(
        *_pts_line(hw_x + 2.5 * LANE_W, hw_y0, hw_x + 2.5 * LANE_W, hw_y1, n=150),
        datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
    )

    # ── 3. Exit Ramp (Highway -> Grid) ─────────────────────────────────────
    # Connects Highway (approx y=100) to intersection (2, 1) -> y=100, x=50
    # Ramp Geometry: S-curve or simple Arc
    # Start: (-10, 60), End: (40, 100) (merging into intersection (2,1) from left)
    ramp_start = (hw_x + 2.5 * LANE_W, 1.2 * BLOCK)  # (-10, 60)
    ramp_end = (1.0 * BLOCK - 10, 2.0 * BLOCK)  # (40, 100)

    # Draw simple quadratic bezier for ramp
    t = np.linspace(0, 1, 50, dtype=np.float32)
    p0 = np.array(ramp_start)
    p1 = np.array([20.0, 60.0])  # Control point
    p2 = np.array(ramp_end)

    bx = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    by = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]

    # Calculate derivatives for heading
    bx_d = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
    by_d = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
    n = np.hypot(bx_d, by_d)
    bx_d /= n
    by_d /= n

    _add(bx, by, bx_d, by_d, datatypes.MapElementIds.LANE_FREEWAY)
    # Ramp Edges
    _add(bx + 2, by - 2, bx_d, by_d, datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    _add(bx - 2, by + 2, bx_d, by_d, datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)

    # ── 4. Roundabout at (3,3) ─────────────────────────────────────────────
    # Intersection center: (150, 150)
    rcx, rcy = roundabout_node[1] * BLOCK, roundabout_node[0] * BLOCK
    R_inner = 12.0
    R_outer = 18.0

    # Draw Inner Circle (Edge)
    _add(
        *_pts_arc(rcx, rcy, R_inner, 0, 2 * np.pi),
        datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
    )
    # Draw Driving Lane (center)
    _add(
        *_pts_arc(rcx, rcy, (R_inner + R_outer) / 2, 0, 2 * np.pi),
        datatypes.MapElementIds.LANE_SURFACE_STREET,
    )
    # Draw Outer Circle (Edge) - approx
    _add(
        *_pts_arc(rcx, rcy, R_outer, 0, 2 * np.pi),
        datatypes.MapElementIds.ROAD_EDGE_BOUNDARY,
    )

    # ── 5. Diagonal Express ────────────────────────────────────────────────
    d_start, d_end = (1 * BLOCK, 1 * BLOCK), (4 * BLOCK, 4 * BLOCK)
    dx, dy = d_end[0] - d_start[0], d_end[1] - d_start[1]
    xs, ys, ddx, ddy = _pts_line(d_start[0], d_start[1], d_end[0], d_end[1], n=100)
    _add(xs, ys, ddx, ddy, datatypes.MapElementIds.LANE_SURFACE_STREET)

    # ── 6. Massive Goal Marker ─────────────────────────────────────────────
    if goal_pos:
        gx, gy = goal_pos
        # Concentric circles
        for r in [2, 4, 6, 8]:
            _add(
                *_pts_arc(gx, gy, r, 0, 2 * np.pi, n=20),
                datatypes.MapElementIds.SPEED_BUMP,
            )
        # Bounding Box
        gbox_x = np.array(
            [gx - 10, gx + 10, gx + 10, gx - 10, gx - 10], dtype=np.float32
        )
        gbox_y = np.array(
            [gy - 10, gy - 10, gy + 10, gy + 10, gy - 10], dtype=np.float32
        )
        _add(
            gbox_x, gbox_y, 0, 1, datatypes.MapElementIds.STOP_SIGN
        )  # STOP signs are distinct

    # ── Pack ──────────────────────────────────────────────────────────────
    N = 40000
    rg_x = np.zeros(N, np.float32)
    rg_y = np.zeros(N, np.float32)
    rg_dx = np.zeros(N, np.float32)
    rg_dy = np.zeros(N, np.float32)
    rg_t = np.full(N, -1, np.int32)
    rg_id = np.zeros(N, np.int32)
    rg_v = np.zeros(N, bool)

    n_pts = min(len(pts), N)
    for i in range(n_pts):
        px, py, pdx, pdy, pt, pid = pts[i]
        rg_x[i] = px
        rg_y[i] = py
        rg_dx[i] = pdx
        rg_dy[i] = pdy
        rg_t[i] = pt
        rg_id[i] = pid
        rg_v[i] = True

    return datatypes.RoadgraphPoints(
        x=jnp.array(rg_x),
        y=jnp.array(rg_y),
        z=jnp.zeros(N),
        dir_x=jnp.array(rg_dx),
        dir_y=jnp.array(rg_dy),
        dir_z=jnp.zeros(N),
        types=jnp.array(rg_t),
        ids=jnp.array(rg_id),
        valid=jnp.array(rg_v),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Hardcoded Complex Planner (State Machine / Waypoints)
# ═══════════════════════════════════════════════════════════════════════════


def get_complex_ego_trajectory():
    waypoints = []

    # 1. Highway Section (South to North approching ramp)
    # Starts at x=-10, y=-20. Ramp starts at y=60
    for y in np.linspace(-20, 60, 20):
        waypoints.append((-10.0, y, np.pi / 2, 16.0))  # (x, y, h, speed)

    # 2. Ramp Section (Bezier)
    # Start: (-10, 60), End: (40, 100)
    p0 = np.array([-10.0, 60.0])
    p1 = np.array([20.0, 60.0])
    p2 = np.array([40.0, 100.0])
    for t in np.linspace(0, 1, 30):
        bx = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        by = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]

        bx_d = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
        by_d = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
        h = math.atan2(by_d, bx_d)
        waypoints.append((bx, by, h, 10.0))  # Slow down on ramp

    # 3. Grid Street Section: (2,1)->(2,3) i.e. x=50->150 along y=100
    # Make sure we merge smoothly
    for x in np.linspace(40, 140, 40):
        waypoints.append((x, 100, 0.0, 10.0))

    # 4. Turn Left at (2,3) [x=150, y=100]
    # Simple curve
    p0 = np.array([140.0, 100.0])
    p1 = np.array([150.0 + LANE_W / 2, 100.0])  # Aim for right lane of Northbound
    p2 = np.array([150.0 + LANE_W / 2, 120.0])
    for t in np.linspace(0, 1, 20):
        bx = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
        by = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]

        bx_d = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
        by_d = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
        h = math.atan2(by_d, bx_d)
        waypoints.append((bx, by, h, 6.0))

    # 5. Approach Roundabout (Northbound on Col 3)
    # From y=120 to y=138
    for y in np.linspace(120, 138, 10):
        waypoints.append((150.0 + LANE_W / 2, y, np.pi / 2, 6.0))

    # 6. Roundabout Traversal (CCW)
    # Enter (150+LaneW/2, 138) -> Radius of lane is ~15?
    # Center (150, 150). Entry is at approx angle -pi/2
    rcx, rcy = 150.0, 150.0
    R_path = 15.0
    # Drive 3/4 way around? No, just 90 deg right?
    # Entered from South, Exiting East. That's -pi/2 to 0.
    for ang in np.linspace(-math.pi / 2, 0, 30):
        wx = rcx + R_path * math.cos(ang)
        wy = rcy + R_path * math.sin(ang)
        h = ang + math.pi / 2  # Tangent
        waypoints.append((wx, wy, h, 6.0))

    # 7. Exit Roundabout Eastbound
    # From x=165, y=150 to Goal (~250)
    goal_x = (N_NS - 1) * BLOCK
    for x in np.linspace(165, goal_x, 40):
        waypoints.append((x, 150 + LANE_W / 2, 0.0, 12.0))

    return waypoints


def interpolate_trajectory(waypoints):
    """Convert waypoints (x,y,h,v) to dense trajectory."""
    # Compute times
    ts_pts = [0.0]
    for i in range(1, len(waypoints)):
        d = math.hypot(
            waypoints[i][0] - waypoints[i - 1][0], waypoints[i][1] - waypoints[i - 1][1]
        )
        v = (waypoints[i][3] + waypoints[i - 1][3]) / 2
        if v < 0.1:
            v = 0.1
        ts_pts.append(ts_pts[-1] + d / v)

    x_out, y_out, vx_out, vy_out, yaw_out = np.zeros((5, N_TIMESTEPS))
    ts_sim = np.arange(N_TIMESTEPS) * DT

    # Simple interpolation
    x_vals = [w[0] for w in waypoints]
    y_vals = [w[1] for w in waypoints]
    h_vals = [w[2] for w in waypoints]
    v_vals = [w[3] for w in waypoints]

    x_out = np.interp(ts_sim, ts_pts, x_vals)
    y_out = np.interp(ts_sim, ts_pts, y_vals)
    # Unwrap headings for clean interp
    h_unwrap = np.unwrap(h_vals)
    yaw_out = np.interp(ts_sim, ts_pts, h_unwrap)
    v_out = np.interp(ts_sim, ts_pts, v_vals)

    vx_out = v_out * np.cos(yaw_out)
    vy_out = v_out * np.sin(yaw_out)

    # Handle end of trajectory (stop)
    end_mask = ts_sim > ts_pts[-1]
    vx_out[end_mask] = 0
    vy_out[end_mask] = 0

    return x_out, y_out, vx_out, vy_out, yaw_out


# ═══════════════════════════════════════════════════════════════════════════
# Simple Background Planner
# ═══════════════════════════════════════════════════════════════════════════
def make_straight_route(start_pos, heading, length, speed, t_offset=0):
    sx, sy = start_pos
    ts = np.arange(N_TIMESTEPS) * DT - t_offset
    d = speed * np.maximum(ts, 0)
    d = np.minimum(d, length)

    x = sx + d * math.cos(heading)
    y = sy + d * math.sin(heading)
    yaw = np.full(N_TIMESTEPS, heading)

    active = (ts >= 0) & (d < length)
    vx = np.where(active, speed * math.cos(heading), 0)
    vy = np.where(active, speed * math.sin(heading), 0)

    return x, y, vx, vy, yaw


# ═══════════════════════════════════════════════════════════════════════════
# Main Scenario Construction
# ═══════════════════════════════════════════════════════════════════════════
def create_city_scenario():
    # Goal: End of Row 3 (East side)
    goal_pos = ((N_NS - 1) * BLOCK, 3 * BLOCK)  # (250, 150)

    roadgraph = build_road_network(goal_pos=goal_pos)

    # Ego Trajectory
    wp = get_complex_ego_trajectory()
    ego_x, ego_y, ego_vx, ego_vy, ego_yaw = interpolate_trajectory(wp)

    # Background Traffic
    bg_data = []
    # 1. Slow truck on Highway (Directly in front of Ego)
    # Ego starts at (-10, -20) @ 16m/s. Truck at (-10, 30) @ 10m/s.
    bg_data.append(make_straight_route((-10, 30), np.pi / 2, 400, 10.0, 0.0))

    # 2. Car on Highway passing exit (Left lane)
    bg_data.append(make_straight_route((-20 + 4, -20), np.pi / 2, 400, 16.0, 0.0))
    # 3. Car on Avenue 2 Southbound
    bg_data.append(
        make_straight_route((2 * BLOCK + 2, 5 * BLOCK), -np.pi / 2, 300, 8.0, 1.0)
    )
    # 4. Car on Row 2 Eastbound (conflicting with Ego merge)
    bg_data.append(make_straight_route((-10, 2 * BLOCK - 2), 0, 300, 7.0, 14.0))
    # 5. Roundabout Circulator (Start North, go CCW)
    bg_data.append(make_straight_route((4 * BLOCK + 2, 0), np.pi / 2, 300, 6.0, 2.0))
    bg_data.append(make_straight_route((0, 4 * BLOCK - 2), 0, 300, 5.0, 0.0))

    # Fill Arrays
    x = np.zeros((NUM_OBJECTS, N_TIMESTEPS))
    y = np.zeros((NUM_OBJECTS, N_TIMESTEPS))
    vx = np.zeros((NUM_OBJECTS, N_TIMESTEPS))
    vy = np.zeros((NUM_OBJECTS, N_TIMESTEPS))
    yaw = np.zeros((NUM_OBJECTS, N_TIMESTEPS))
    valid = np.ones((NUM_OBJECTS, N_TIMESTEPS), bool)

    # Ego
    x[0], y[0], vx[0], vy[0], yaw[0] = ego_x, ego_y, ego_vx, ego_vy, ego_yaw

    # BG
    for i, (bx, by, bvx, bvy, byaw) in enumerate(bg_data):
        idx = i + 1
        if idx >= NUM_OBJECTS:
            break
        x[idx], y[idx], vx[idx], vy[idx], yaw[idx] = bx, by, bvx, bvy, byaw

    # Parked (Static)
    parked = [(150, 260), (160, 260), (170, 260)]
    for i, (px, py) in enumerate(parked):
        idx = 10 + i
        x[idx] = px
        y[idx] = py

    # Pack
    ts = (
        (np.arange(N_TIMESTEPS) * int(DT * 1e6))[None, :]
        .repeat(NUM_OBJECTS, 0)
        .astype(np.int32)
    )
    traj = datatypes.Trajectory(
        x=jnp.array(x),
        y=jnp.array(y),
        z=jnp.zeros_like(x),
        vel_x=jnp.array(vx),
        vel_y=jnp.array(vy),
        yaw=jnp.array(yaw),
        valid=jnp.array(valid),
        timestamp_micros=jnp.array(ts),
        length=jnp.full_like(x, 4.5),
        width=jnp.full_like(x, 2.0),
        height=jnp.full_like(x, 1.5),
    )

    metadata = datatypes.ObjectMetadata(
        ids=jnp.arange(NUM_OBJECTS, dtype=jnp.int32),
        object_types=jnp.ones(NUM_OBJECTS, dtype=jnp.int32),
        is_sdc=jnp.array([True] + [False] * (NUM_OBJECTS - 1)),
        is_modeled=jnp.ones(NUM_OBJECTS, bool),
        is_valid=jnp.ones(NUM_OBJECTS, bool),
        objects_of_interest=jnp.array([True] + [False] * (NUM_OBJECTS - 1)),
        is_controlled=jnp.ones(NUM_OBJECTS, bool),
    )

    # Empty Traffic Lights
    tl = datatypes.TrafficLights(
        x=jnp.zeros((1, N_TIMESTEPS)),
        y=jnp.zeros((1, N_TIMESTEPS)),
        z=jnp.zeros((1, N_TIMESTEPS)),
        state=jnp.zeros((1, N_TIMESTEPS), dtype=jnp.int32),
        lane_ids=jnp.zeros((1, N_TIMESTEPS), dtype=jnp.int32),
        valid=jnp.zeros((1, N_TIMESTEPS), dtype=bool),
    )

    return datatypes.SimulatorState(
        sim_trajectory=traj,
        log_trajectory=traj,
        log_traffic_light=tl,
        object_metadata=metadata,
        timestep=jnp.int32(0),
        roadgraph_points=roadgraph,
        sdc_paths=None,
    )


if __name__ == "__main__":
    scenario = create_city_scenario()

    dynamics_model = dynamics.StateDynamics()
    # Use IDM for safety
    idm_actor = agents.IDMRoutePolicy(
        is_controlled_func=lambda state: state.object_metadata.ids < 10
    )
    parked_actor = agents.create_constant_speed_actor(
        speed=0.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: state.object_metadata.ids >= 10,
    )

    env = _env.BaseEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(), max_num_objects=NUM_OBJECTS
        ),
    )

    states = [env.reset(scenario)]
    print(f"Simulating {SIM_STEPS} steps...")

    jit_step = jax.jit(env.step)
    jit_act_idm = jax.jit(idm_actor.select_action)
    jit_act_park = jax.jit(parked_actor.select_action)

    for _ in tqdm(range(SIM_STEPS)):
        s = states[-1]
        act_idm = jit_act_idm({}, s, None, None)
        act_park = jit_act_park({}, s, None, None)
        action = agents.merge_actions([act_idm, act_park])
        states.append(jit_step(s, action))

    print("Rendering...")
    viz_cfg = {
        "front_x": 150,
        "back_x": 100,
        "front_y": 150,
        "back_y": 100,  # Expanded view
        "px_per_meter": 2.0,
        "show_agent_id": True,
    }
    # Create config object manually if needed, or pass dict if supported (Waymax 0.1 usually needs object or dicts are auto-converted)
    # The previous error suggested plot_simulator_state might be strict or the matplotlib backend issue.
    # We will try passing dict as kwargs or just default.
    # Actually, plot_simulator_state takes viz_config.

    # Try using default viz mostly but set limits differently if possible.
    # Since visualizer is finicky, let's stick to dict if it worked before, but ensure numeric consistency.

    imgs = [
        visualization.plot_simulator_state(s, viz_config=viz_cfg) for s in tqdm(states)
    ]
    mediapy.write_video("city_planning_sim.mp4", imgs, fps=10)
    print("Video saved as city_planning_sim.mp4")
