"""
Complex city-grid multi-agent simulation with 7 road segments and 32 actors.

Road Network (7 segments, south-to-north then east):
  A — 3-lane Highway (northbound, dense traffic)
  B — Exit Ramp (Bezier curve from highway to city grid)
  C — Signalized Intersection + Crosswalk (pedestrians, red light)
  D — School / Construction Zone (speed bumps, narrow lanes, VRUs)
  E — Roundabout (yield, cyclists sharing lane)
  F — Narrow Street with Double-Parked Truck (oncoming traffic)
  G — Complex Final Intersection (left turn, pedestrians, cross-traffic)

Actors (32 total):
  Object  0       — Ego vehicle traversing all segments
  Objects 1-20    — Background vehicles (highway, intersection, roundabout, etc.)
  Objects 21-26   — Pedestrians at crosswalks and school zone
  Objects 27-30   — Cyclists (roundabout, school zone, narrow street)
  Object  31      — Parked truck blocking lane (forces opposing-lane pass)

Traffic Lights:
  TL-0 at Seg C (EW) — red when ego arrives (~13s)
  TL-1 at Seg C (NS) — green when TL-0 is red
  TL-2 at Seg G (NS) — red when ego arrives (~33s)

Rule Triggering:
  Seg C: 9 levels, 15 rules (L10,9,8,7,6,3,2,1,0)
  Seg D: 10 levels, 15 rules (L10,9,8,7,5,4,3,2,1,0)
  Seg G: 10 levels, 17 rules (L10,9,8,7,6,4,3,2,1,0)
  Total: 30 unique rules across all 11 levels over full trajectory

Output:
  city_planning_sim.mp4  (~35 s @ 10 fps, 351 frames)
"""

from __future__ import annotations

import dataclasses
import math
from typing import List, Tuple

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

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
BLOCK = 50.0          # metres per city block
LANE_W = 4.0          # lane width (m)
DT = 0.1              # timestep (10 Hz)
N_TIMESTEPS = 370     # trajectory capacity (~37 s)
SIM_STEPS = 350       # actual simulation steps (35 s)
NUM_OBJECTS = 32      # total actors
N_ROAD_PTS = 60000    # roadgraph point capacity


# ═══════════════════════════════════════════════════════════════════════════════
# Road-Building Primitives
# ═══════════════════════════════════════════════════════════════════════════════
def _pts_line(x0, y0, x1, y1, n=80):
    """Straight road points with unit direction vector."""
    xs = np.linspace(x0, x1, n, dtype=np.float32)
    ys = np.linspace(y0, y1, n, dtype=np.float32)
    dx, dy = float(x1 - x0), float(y1 - y0)
    L = math.hypot(dx, dy) or 1.0
    return xs, ys, np.float32(dx / L), np.float32(dy / L)


def _pts_arc(cx, cy, r, ang_start, ang_end, n=50):
    """Circular arc road points with tangent directions."""
    thetas = np.linspace(ang_start, ang_end, n, dtype=np.float32)
    xs = (cx + r * np.cos(thetas)).astype(np.float32)
    ys = (cy + r * np.sin(thetas)).astype(np.float32)
    direction = 1.0 if ang_end > ang_start else -1.0
    dxs = (-np.sin(thetas) * direction).astype(np.float32)
    dys = (np.cos(thetas) * direction).astype(np.float32)
    return xs, ys, dxs, dys


def _pts_bezier(p0, p1, p2, n=60):
    """Quadratic Bezier curve points with unit tangent directions."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    xs = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
    ys = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    dxs = 2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0])
    dys = 2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
    norms = np.hypot(dxs, dys)
    norms[norms < 1e-8] = 1.0
    return xs.astype(np.float32), ys.astype(np.float32), \
        (dxs / norms).astype(np.float32), (dys / norms).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Road Network
# ═══════════════════════════════════════════════════════════════════════════════
def build_road_network():
    """Construct the 7-segment road network as RoadgraphPoints."""
    pts: list[tuple] = []
    eid = 0

    def _add(xs, ys, dx, dy, typ):
        nonlocal eid
        eid += 1
        for i in range(len(xs)):
            ddx = dx if np.isscalar(dx) else dx[i]
            ddy = dy if np.isscalar(dy) else dy[i]
            pts.append((xs[i], ys[i], ddx, ddy, int(typ), eid))

    # ── Segment A: 3-Lane Highway ─────────────────────────────────────────
    hw_x = -30.0
    hw_y0, hw_y1 = -50.0, 350.0
    # 3 northbound freeway lanes
    for i in range(3):
        offset = (i - 1) * LANE_W  # -4, 0, +4
        _add(*_pts_line(hw_x + offset, hw_y0, hw_x + offset, hw_y1, n=200),
             datatypes.MapElementIds.LANE_FREEWAY)
    # Lane markings (broken white between lanes)
    for mark_off in [-LANE_W / 2, LANE_W / 2]:
        _add(*_pts_line(hw_x + mark_off, hw_y0, hw_x + mark_off, hw_y1, n=150),
             datatypes.MapElementIds.ROAD_LINE_BROKEN_SINGLE_WHITE)
    # Road edges
    for edge_off in [-1.5 * LANE_W, 1.5 * LANE_W]:
        _add(*_pts_line(hw_x + edge_off, hw_y0, hw_x + edge_off, hw_y1, n=150),
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)

    # ── Segment B: Exit Ramp (Bezier) ─────────────────────────────────────
    ramp_p0 = np.array([-24.0, 200.0])
    ramp_p1 = np.array([60.0, 200.0])
    ramp_p2 = np.array([150.0, 250.0])
    bx, by, bdx, bdy = _pts_bezier(ramp_p0, ramp_p1, ramp_p2, n=60)
    _add(bx, by, bdx, bdy, datatypes.MapElementIds.LANE_SURFACE_STREET)
    # Ramp edge boundaries (offset ±3m perpendicular to tangent)
    perp_x = -bdy  # perpendicular direction
    perp_y = bdx
    for sign in [3.0, -3.0]:
        _add(bx + sign * perp_x, by + sign * perp_y, bdx, bdy,
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)

    # ── Segment C: Signalized Intersection + Crosswalk ────────────────────
    ic_x, ic_y = 175.0, 255.0  # intersection center

    # EW approach lanes (from ramp end, heading east)
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(130.0, ic_y + offset, ic_x - 20, ic_y + offset, n=40),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # EW departure lanes (heading east past intersection)
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(ic_x + 20, ic_y + offset, 220.0, ic_y + offset, n=40),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # NS lanes through intersection
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(ic_x + offset, ic_y - 30, ic_x + offset, ic_y + 30, n=50),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # Intersection box edges
    for edge_off in [LANE_W, -LANE_W]:
        _add(*_pts_line(ic_x + edge_off, ic_y - 20, ic_x + edge_off, ic_y + 20, n=30),
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    for edge_off in [20.0, -20.0]:
        _add(*_pts_line(ic_x - LANE_W, ic_y + edge_off, ic_x + LANE_W, ic_y + edge_off, n=15),
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    # Crosswalk stripes (two parallel lines spanning NS direction)
    for cx_off in [-3.0, 3.0]:
        _add(*_pts_line(ic_x + cx_off, ic_y - 5, ic_x + cx_off, ic_y + 5, n=20),
             datatypes.MapElementIds.CROSSWALK)
    # Stop sign cluster at EW approach
    eid += 1
    for sx in np.linspace(148.0, 152.0, 5):
        for sy in np.linspace(ic_y - 2, ic_y + 2, 5):
            pts.append((np.float32(sx), np.float32(sy),
                        np.float32(1.0), np.float32(0.0),
                        int(datatypes.MapElementIds.STOP_SIGN), eid))

    # ── Segment D: School / Construction Zone ─────────────────────────────
    sz_x = 195.0
    sz_y0, sz_y1 = 280.0, 370.0
    # 2 narrow lanes (3.2m each)
    for offset, direction in [(1.6, 1), (-1.6, -1)]:
        xs, ys, dx, dy = _pts_line(sz_x + offset, sz_y0, sz_x + offset, sz_y1, n=60)
        if direction == -1:
            ys = ys[::-1]
            dy = -dy
        _add(xs, ys, dx, dy, datatypes.MapElementIds.LANE_SURFACE_STREET)
    # Tight road edges
    for edge_off in [4.0, -4.0]:
        _add(*_pts_line(sz_x + edge_off, sz_y0, sz_x + edge_off, sz_y1, n=50),
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    # Speed bumps at y=300, 330, 350
    for bump_y in [300.0, 330.0, 350.0]:
        _add(*_pts_line(sz_x - 4, bump_y, sz_x + 4, bump_y, n=15),
             datatypes.MapElementIds.SPEED_BUMP)

    # Connecting road from intersection C to school zone D
    # Curve from (195, 270) to (195, 280) — short link
    _add(*_pts_line(195.0, 268.0, 195.0, 282.0, n=15),
         datatypes.MapElementIds.LANE_SURFACE_STREET)

    # ── Segment E: Roundabout ─────────────────────────────────────────────
    rcx, rcy = 225.0, 400.0
    R_inner, R_outer = 10.0, 18.0
    R_lane = (R_inner + R_outer) / 2.0  # 14.0m
    # Inner edge circle
    _add(*_pts_arc(rcx, rcy, R_inner, 0, 2 * np.pi, n=50),
         datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    # Driving lane (CCW)
    _add(*_pts_arc(rcx, rcy, R_lane, 0, 2 * np.pi, n=60),
         datatypes.MapElementIds.LANE_SURFACE_STREET)
    # Outer edge circle
    _add(*_pts_arc(rcx, rcy, R_outer, 0, 2 * np.pi, n=50),
         datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    # Approach spokes
    # South approach (from school zone)
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(rcx + offset, rcy - R_outer - 15, rcx + offset, rcy - R_outer, n=20),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # East departure (toward narrow street)
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(rcx + R_outer, rcy + offset, rcx + R_outer + 15, rcy + offset, n=20),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # North spoke
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(rcx + offset, rcy + R_outer, rcx + offset, rcy + R_outer + 15, n=20),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # West spoke
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(rcx - R_outer - 15, rcy + offset, rcx - R_outer, rcy + offset, n=20),
             datatypes.MapElementIds.LANE_SURFACE_STREET)

    # Connecting road from school zone to roundabout south spoke
    _add(*_pts_line(195.0, 370.0, 225.0, rcy - R_outer - 15, n=30),
         datatypes.MapElementIds.LANE_SURFACE_STREET)

    # ── Segment F: Narrow Street ──────────────────────────────────────────
    ns_x = 250.0
    ns_y0, ns_y1 = 430.0, 530.0
    # Two narrow lanes (3.5m each)
    for offset, direction in [(1.75, 1), (-1.75, -1)]:
        xs, ys, dx, dy = _pts_line(ns_x + offset, ns_y0, ns_x + offset, ns_y1, n=60)
        if direction == -1:
            ys = ys[::-1]
            dy = -dy
        _add(xs, ys, dx, dy, datatypes.MapElementIds.LANE_SURFACE_STREET)
    # Tight road edges
    for edge_off in [4.5, -4.5]:
        _add(*_pts_line(ns_x + edge_off, ns_y0, ns_x + edge_off, ns_y1, n=50),
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)

    # Connecting road from roundabout east to narrow street
    _add(*_pts_line(rcx + R_outer + 15, rcy, ns_x, ns_y0, n=30),
         datatypes.MapElementIds.LANE_SURFACE_STREET)

    # ── Segment G: Complex Final Intersection ─────────────────────────────
    fi_x, fi_y = 250.0, 560.0
    # NS through-road
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(fi_x + offset, fi_y - 30, fi_x + offset, fi_y + 30, n=40),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # EW cross-road
    for offset in [LANE_W / 2, -LANE_W / 2]:
        _add(*_pts_line(fi_x - 40, fi_y + offset, fi_x + 40, fi_y + offset, n=40),
             datatypes.MapElementIds.LANE_SURFACE_STREET)
    # Intersection edges
    for edge_off in [LANE_W, -LANE_W]:
        _add(*_pts_line(fi_x + edge_off, fi_y - 25, fi_x + edge_off, fi_y + 25, n=30),
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    for edge_off in [25.0, -25.0]:
        _add(*_pts_line(fi_x - LANE_W, fi_y + edge_off, fi_x + LANE_W, fi_y + edge_off, n=15),
             datatypes.MapElementIds.ROAD_EDGE_BOUNDARY)
    # Crosswalk at final intersection (north side)
    for cx_off in [-4.0, 4.0]:
        _add(*_pts_line(fi_x + cx_off, fi_y + 12, fi_x + cx_off, fi_y + 16, n=15),
             datatypes.MapElementIds.CROSSWALK)
    # Stop sign at NS approach (south)
    eid += 1
    for sx in np.linspace(fi_x - 2, fi_x + 2, 5):
        for sy in np.linspace(fi_y - 28, fi_y - 25, 5):
            pts.append((np.float32(sx), np.float32(sy),
                        np.float32(0.0), np.float32(1.0),
                        int(datatypes.MapElementIds.STOP_SIGN), eid))

    # Connecting road from narrow street to final intersection
    _add(*_pts_line(ns_x, ns_y1, fi_x, fi_y - 30, n=15),
         datatypes.MapElementIds.LANE_SURFACE_STREET)

    # ── Pack into RoadgraphPoints ─────────────────────────────────────────
    N = N_ROAD_PTS
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

    print(f"  Road network: {n_pts} points, {eid} elements")

    return datatypes.RoadgraphPoints(
        x=jnp.array(rg_x), y=jnp.array(rg_y), z=jnp.zeros(N),
        dir_x=jnp.array(rg_dx), dir_y=jnp.array(rg_dy), dir_z=jnp.zeros(N),
        types=jnp.array(rg_t), ids=jnp.array(rg_id), valid=jnp.array(rg_v),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Ego Trajectory — 7 Phases
# ═══════════════════════════════════════════════════════════════════════════════
def get_complex_ego_trajectory() -> List[Tuple[float, float, float, float]]:
    """Return waypoints (x, y, heading, speed) for ego through all segments."""
    wp: list[tuple[float, float, float, float]] = []

    # Phase 1 — Highway northbound (25 m/s, 8 s)
    for y in np.linspace(-30, 170, 25):
        wp.append((-30.0, float(y), math.pi / 2, 25.0))

    # Phase 2 — Exit ramp Bezier, decelerate 25→12 m/s
    p0 = np.array([-24.0, 200.0])
    p1 = np.array([60.0, 200.0])
    p2 = np.array([150.0, 250.0])
    for i, t in enumerate(np.linspace(0, 1, 25)):
        bx = float((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0])
        by = float((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1])
        bdx = float(2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0]))
        bdy = float(2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1]))
        h = math.atan2(bdy, bdx)
        speed = 25.0 - 13.0 * float(t)
        wp.append((bx, by, h, speed))

    # Phase 3 — Signalized intersection: approach, stop at red, go with turn
    # Approach to stop line
    for y in np.linspace(250, 253, 5):
        wp.append((155.0, float(y), math.pi / 2, 8.0))
    # Stop and wait (~3 s dwell)
    for _ in range(10):
        wp.append((155.0, 254.0, math.pi / 2, 0.0))
    # Turn right through intersection toward school zone
    tp0 = np.array([155.0, 254.0])
    tp1 = np.array([175.0, 262.0])
    tp2 = np.array([195.0, 272.0])
    for t in np.linspace(0, 1, 12):
        bx = float((1 - t) ** 2 * tp0[0] + 2 * (1 - t) * t * tp1[0] + t ** 2 * tp2[0])
        by = float((1 - t) ** 2 * tp0[1] + 2 * (1 - t) * t * tp1[1] + t ** 2 * tp2[1])
        bdx = float(2 * (1 - t) * (tp1[0] - tp0[0]) + 2 * t * (tp2[0] - tp1[0]))
        bdy = float(2 * (1 - t) * (tp1[1] - tp0[1]) + 2 * t * (tp2[1] - tp1[1]))
        h = math.atan2(bdy, bdx)
        wp.append((bx, by, h, 8.0))

    # Phase 4 — School / construction zone (5 m/s)
    for y in np.linspace(280, 370, 22):
        frac = (float(y) - 280.0) / 90.0
        x = 195.0 + frac * 10.0  # slight diagonal toward roundabout
        wp.append((x, float(y), math.pi / 2, 5.0))

    # Transition to roundabout approach
    for p in np.linspace(0, 1, 8):
        x = 205.0 + float(p) * 20.0  # 205 → 225
        y = 370.0 + float(p) * 12.0  # 370 → 382
        wp.append((x, y, math.atan2(12, 20), 5.0))

    # Phase 5 — Roundabout (CCW, enter from south, exit east, 6 m/s)
    rcx, rcy, R_path = 225.0, 400.0, 14.0
    for ang in np.linspace(-math.pi / 2, 0, 25):
        wx = rcx + R_path * math.cos(ang)
        wy = rcy + R_path * math.sin(ang)
        h = ang + math.pi / 2  # tangent for CCW motion
        wp.append((wx, wy, h, 6.0))

    # Transition roundabout exit → narrow street
    for p in np.linspace(0, 1, 8):
        x = 239.0 + float(p) * 11.0  # 239 → 250
        y = 400.0 + float(p) * 30.0  # 400 → 430
        wp.append((x, y, math.pi / 2, 5.0))

    # Phase 6 — Narrow street (4 m/s), weave around parked truck at y~475
    for y in np.linspace(430, 465, 10):
        wp.append((250.0, float(y), math.pi / 2, 4.0))
    # Swerve left to avoid parked truck (x=254), then back
    for y in np.linspace(468, 485, 8):
        frac = (float(y) - 468.0) / 17.0
        swerve = 4.0 * math.sin(math.pi * frac)  # max 4m left swerve
        wp.append((250.0 - swerve, float(y), math.pi / 2, 3.5))
    for y in np.linspace(488, 530, 10):
        wp.append((250.0, float(y), math.pi / 2, 4.0))

    # Phase 7 — Final intersection left turn (3 m/s)
    lp0 = np.array([250.0, 530.0])
    lp1 = np.array([250.0, 560.0])
    lp2 = np.array([210.0, 560.0])
    for t in np.linspace(0, 1, 20):
        bx = float((1 - t) ** 2 * lp0[0] + 2 * (1 - t) * t * lp1[0] + t ** 2 * lp2[0])
        by = float((1 - t) ** 2 * lp0[1] + 2 * (1 - t) * t * lp1[1] + t ** 2 * lp2[1])
        bdx = float(2 * (1 - t) * (lp1[0] - lp0[0]) + 2 * t * (lp2[0] - lp1[0]))
        bdy = float(2 * (1 - t) * (lp1[1] - lp0[1]) + 2 * t * (lp2[1] - lp1[1]))
        h = math.atan2(bdy, bdx)
        wp.append((bx, by, h, 3.0))

    return wp


def interpolate_trajectory(
    waypoints: List[Tuple[float, float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert (x, y, heading, speed) waypoints to dense arrays."""
    # Compute cumulative time from distances and speeds
    ts_pts = [0.0]
    for i in range(1, len(waypoints)):
        d = math.hypot(
            waypoints[i][0] - waypoints[i - 1][0],
            waypoints[i][1] - waypoints[i - 1][1],
        )
        v = (waypoints[i][3] + waypoints[i - 1][3]) / 2
        if v < 0.1:
            v = 0.1
        ts_pts.append(ts_pts[-1] + d / v)

    ts_sim = np.arange(N_TIMESTEPS) * DT

    x_vals = [w[0] for w in waypoints]
    y_vals = [w[1] for w in waypoints]
    h_vals = [w[2] for w in waypoints]
    v_vals = [w[3] for w in waypoints]

    x_out = np.interp(ts_sim, ts_pts, x_vals).astype(np.float32)
    y_out = np.interp(ts_sim, ts_pts, y_vals).astype(np.float32)
    h_unwrap = np.unwrap(h_vals)
    yaw_out = np.interp(ts_sim, ts_pts, h_unwrap).astype(np.float32)
    v_out = np.interp(ts_sim, ts_pts, v_vals).astype(np.float32)

    vx_out = (v_out * np.cos(yaw_out)).astype(np.float32)
    vy_out = (v_out * np.sin(yaw_out)).astype(np.float32)

    # Clamp after end of waypoints
    end_mask = ts_sim > ts_pts[-1]
    vx_out[end_mask] = 0
    vy_out[end_mask] = 0

    return x_out, y_out, vx_out, vy_out, yaw_out


# ═══════════════════════════════════════════════════════════════════════════════
# Background Actor Trajectory Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def make_straight_route(start_pos, heading, length, speed, t_offset=0.0):
    """Straight constant-velocity trajectory."""
    sx, sy = start_pos
    ts = np.arange(N_TIMESTEPS, dtype=np.float32) * DT - t_offset
    d = speed * np.maximum(ts, 0)
    d = np.minimum(d, length)

    x = (sx + d * math.cos(heading)).astype(np.float32)
    y = (sy + d * math.sin(heading)).astype(np.float32)
    yaw = np.full(N_TIMESTEPS, heading, dtype=np.float32)

    active = (ts >= 0) & (d < length)
    vx = np.where(active, speed * math.cos(heading), 0).astype(np.float32)
    vy = np.where(active, speed * math.sin(heading), 0).astype(np.float32)

    return x, y, vx, vy, yaw


def make_arc_route(center, radius, ang_start, ang_end, speed, t_offset=0.0):
    """Trajectory following a circular arc at constant speed."""
    arc_len = radius * abs(ang_end - ang_start)
    duration = arc_len / max(speed, 0.1)
    ts = np.arange(N_TIMESTEPS, dtype=np.float32) * DT - t_offset
    frac = np.clip(ts / duration, 0, 1)
    angles = ang_start + frac * (ang_end - ang_start)

    x = (center[0] + radius * np.cos(angles)).astype(np.float32)
    y = (center[1] + radius * np.sin(angles)).astype(np.float32)
    yaw = (angles + math.pi / 2).astype(np.float32)  # tangent (CCW)

    active = (ts >= 0) & (frac < 1.0)
    vx = np.where(active, speed * np.cos(yaw), 0).astype(np.float32)
    vy = np.where(active, speed * np.sin(yaw), 0).astype(np.float32)

    return x, y, vx, vy, yaw


def make_bezier_route(p0, p1, p2, speed, t_offset=0.0):
    """Trajectory following a quadratic Bezier curve at approx constant speed."""
    # Estimate arc length for timing
    n_sample = 100
    t_s = np.linspace(0, 1, n_sample)
    bx_s = (1 - t_s) ** 2 * p0[0] + 2 * (1 - t_s) * t_s * p1[0] + t_s ** 2 * p2[0]
    by_s = (1 - t_s) ** 2 * p0[1] + 2 * (1 - t_s) * t_s * p1[1] + t_s ** 2 * p2[1]
    arc_len = float(np.sum(np.hypot(np.diff(bx_s), np.diff(by_s))))
    duration = arc_len / max(speed, 0.1)

    ts = np.arange(N_TIMESTEPS, dtype=np.float32) * DT - t_offset
    frac = np.clip(ts / duration, 0, 1)

    x = ((1 - frac) ** 2 * p0[0] + 2 * (1 - frac) * frac * p1[0] + frac ** 2 * p2[0]).astype(np.float32)
    y = ((1 - frac) ** 2 * p0[1] + 2 * (1 - frac) * frac * p1[1] + frac ** 2 * p2[1]).astype(np.float32)
    dx = (2 * (1 - frac) * (p1[0] - p0[0]) + 2 * frac * (p2[0] - p1[0]))
    dy = (2 * (1 - frac) * (p1[1] - p0[1]) + 2 * frac * (p2[1] - p1[1]))
    yaw = np.arctan2(dy, dx).astype(np.float32)

    active = (ts >= 0) & (frac < 1.0)
    vx = np.where(active, speed * np.cos(yaw), 0).astype(np.float32)
    vy = np.where(active, speed * np.sin(yaw), 0).astype(np.float32)

    return x, y, vx, vy, yaw


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario Assembly
# ═══════════════════════════════════════════════════════════════════════════════
def create_city_scenario():
    """Build the complete SimulatorState with 32 actors, road network, and TLs."""
    print("Building road network...")
    roadgraph = build_road_network()

    print("Generating ego trajectory...")
    wp = get_complex_ego_trajectory()
    ego_x, ego_y, ego_vx, ego_vy, ego_yaw = interpolate_trajectory(wp)

    print("Generating background actor trajectories...")

    # Allocate trajectory arrays
    x = np.zeros((NUM_OBJECTS, N_TIMESTEPS), np.float32)
    y = np.zeros((NUM_OBJECTS, N_TIMESTEPS), np.float32)
    vx = np.zeros((NUM_OBJECTS, N_TIMESTEPS), np.float32)
    vy = np.zeros((NUM_OBJECTS, N_TIMESTEPS), np.float32)
    yaw = np.zeros((NUM_OBJECTS, N_TIMESTEPS), np.float32)
    valid = np.ones((NUM_OBJECTS, N_TIMESTEPS), bool)

    # ── Object 0: Ego ─────────────────────────────────────────────────────
    x[0], y[0], vx[0], vy[0], yaw[0] = ego_x, ego_y, ego_vx, ego_vy, ego_yaw

    # ── Objects 1-6: Highway vehicles (Segment A) ─────────────────────────
    x[1], y[1], vx[1], vy[1], yaw[1] = make_straight_route((-30, 10), math.pi / 2, 400, 20.0)
    x[2], y[2], vx[2], vy[2], yaw[2] = make_straight_route((-34, -10), math.pi / 2, 400, 28.0)
    x[3], y[3], vx[3], vy[3], yaw[3] = make_straight_route((-26, 50), math.pi / 2, 400, 22.0)
    x[4], y[4], vx[4], vy[4], yaw[4] = make_straight_route((-30, 80), math.pi / 2, 400, 18.0)
    x[5], y[5], vx[5], vy[5], yaw[5] = make_straight_route((-34, -50), math.pi / 2, 400, 26.0)
    # Object 6: takes exit ramp (Bezier trajectory)
    x[6], y[6], vx[6], vy[6], yaw[6] = make_bezier_route(
        np.array([-26.0, 150.0]), np.array([60.0, 200.0]), np.array([150.0, 250.0]),
        speed=18.0, t_offset=5.0,
    )

    # ── Objects 7-10: Intersection vehicles (Segment B-C) ─────────────────
    x[7], y[7], vx[7], vy[7], yaw[7] = make_straight_route((80, 248), 0, 200, 10.0, t_offset=8.0)
    x[8], y[8], vx[8], vy[8], yaw[8] = make_straight_route((175, 230), math.pi / 2, 80, 8.0, t_offset=10.0)
    x[9], y[9], vx[9], vy[9], yaw[9] = make_straight_route((175, 275), -math.pi / 2, 80, 7.0, t_offset=12.0)
    x[10], y[10], vx[10], vy[10], yaw[10] = make_straight_route((200, 255), math.pi, 100, 6.0, t_offset=11.0)

    # ── Objects 11-12: School zone vehicles (Segment D) ───────────────────
    x[11], y[11], vx[11], vy[11], yaw[11] = make_straight_route((195, 310), math.pi / 2, 80, 4.0, t_offset=16.0)
    x[12], y[12], vx[12], vy[12], yaw[12] = make_straight_route((195, 350), -math.pi / 2, 80, 3.0, t_offset=16.0)

    # ── Objects 13-14: Roundabout circulators (Segment E) ─────────────────
    rcx, rcy = 225.0, 400.0
    x[13], y[13], vx[13], vy[13], yaw[13] = make_arc_route(
        (rcx, rcy), 14.0, math.pi / 2, math.pi / 2 + 2 * math.pi, 5.0, t_offset=20.0)
    x[14], y[14], vx[14], vy[14], yaw[14] = make_arc_route(
        (rcx, rcy), 14.0, 0, 2 * math.pi, 6.0, t_offset=22.0)

    # ── Objects 15-17: Narrow street vehicles (Segment F) ─────────────────
    x[15], y[15], vx[15], vy[15], yaw[15] = make_straight_route((248, 460), math.pi / 2, 100, 5.0, t_offset=26.0)
    x[16], y[16], vx[16], vy[16], yaw[16] = make_straight_route((252, 510), -math.pi / 2, 100, 4.0, t_offset=26.0)
    x[17], y[17], vx[17], vy[17], yaw[17] = make_straight_route((252, 480), -math.pi / 2, 100, 4.0, t_offset=27.0)

    # ── Objects 18-20: Final intersection vehicles (Segment G) ────────────
    x[18], y[18], vx[18], vy[18], yaw[18] = make_straight_route((250, 580), -math.pi / 2, 80, 5.0, t_offset=30.0)
    x[19], y[19], vx[19], vy[19], yaw[19] = make_straight_route((230, 558), 0, 80, 6.0, t_offset=31.0)
    x[20], y[20], vx[20], vy[20], yaw[20] = make_straight_route((270, 562), math.pi, 80, 5.0, t_offset=31.0)

    # ── Objects 21-22: Pedestrians at Seg C crosswalk ─────────────────────
    x[21], y[21], vx[21], vy[21], yaw[21] = make_straight_route((172, 253), 0, 15, 1.5, t_offset=12.0)
    x[22], y[22], vx[22], vy[22], yaw[22] = make_straight_route((178, 257), math.pi, 15, 1.2, t_offset=13.0)

    # ── Objects 23-24: Pedestrians in school zone (Segment D) ─────────────
    x[23], y[23], vx[23], vy[23], yaw[23] = make_straight_route((191, 320), math.pi / 2, 30, 1.0, t_offset=18.0)
    x[24], y[24], vx[24], vy[24], yaw[24] = make_straight_route((199, 340), -math.pi / 2, 30, 1.3, t_offset=19.0)

    # ── Objects 25-26: Pedestrians at Seg G crosswalk ─────────────────────
    x[25], y[25], vx[25], vy[25], yaw[25] = make_straight_route((248, 572), 0, 12, 1.5, t_offset=32.0)
    x[26], y[26], vx[26], vy[26], yaw[26] = make_straight_route((252, 575), math.pi, 12, 1.4, t_offset=32.5)

    # ── Objects 27-28: Cyclists in roundabout (Segment E) ─────────────────
    x[27], y[27], vx[27], vy[27], yaw[27] = make_arc_route(
        (rcx, rcy), 14.0, math.pi, math.pi + 2 * math.pi, 7.0, t_offset=23.0)
    x[28], y[28], vx[28], vy[28], yaw[28] = make_straight_route((225, 382), math.pi / 2, 50, 6.0, t_offset=23.0)

    # ── Object 29: Cyclist on narrow street (Segment F) ───────────────────
    x[29], y[29], vx[29], vy[29], yaw[29] = make_straight_route((246, 440), math.pi / 2, 100, 5.0, t_offset=27.0)

    # ── Object 30: Cyclist in school zone (Segment D) ─────────────────────
    x[30], y[30], vx[30], vy[30], yaw[30] = make_straight_route((192, 290), math.pi / 2, 100, 8.0, t_offset=18.0)

    # ── Object 31: Parked truck on narrow street (Segment F) ──────────────
    x[31] = 254.0
    y[31] = 475.0
    yaw[31] = math.pi / 2  # facing north

    # ── Per-object dimensions ─────────────────────────────────────────────
    length = np.full((NUM_OBJECTS, N_TIMESTEPS), 4.5, np.float32)
    width = np.full((NUM_OBJECTS, N_TIMESTEPS), 2.0, np.float32)
    height = np.full((NUM_OBJECTS, N_TIMESTEPS), 1.5, np.float32)
    # Pedestrians (IDs 21-26)
    length[21:27] = 0.5;  width[21:27] = 0.5;  height[21:27] = 1.7
    # Cyclists (IDs 27-31 → 27-30)
    length[27:31] = 1.8;  width[27:31] = 0.6;  height[27:31] = 1.7
    # Parked truck (ID 31)
    length[31] = 12.0;    width[31] = 2.5;      height[31] = 3.5

    # ── Timestamps ────────────────────────────────────────────────────────
    ts_micros = (
        (np.arange(N_TIMESTEPS) * int(DT * 1e6))[None, :]
        .repeat(NUM_OBJECTS, 0)
        .astype(np.int32)
    )

    traj = datatypes.Trajectory(
        x=jnp.array(x), y=jnp.array(y), z=jnp.zeros_like(x),
        vel_x=jnp.array(vx), vel_y=jnp.array(vy),
        yaw=jnp.array(yaw), valid=jnp.array(valid),
        timestamp_micros=jnp.array(ts_micros),
        length=jnp.array(length), width=jnp.array(width), height=jnp.array(height),
    )

    # ── Object Metadata ───────────────────────────────────────────────────
    obj_types = np.ones(NUM_OBJECTS, np.int32)       # default: vehicle
    obj_types[21:27] = 2                             # pedestrians
    obj_types[27:31] = 3                             # cyclists

    metadata = datatypes.ObjectMetadata(
        ids=jnp.arange(NUM_OBJECTS, dtype=jnp.int32),
        object_types=jnp.array(obj_types),
        is_sdc=jnp.array([True] + [False] * (NUM_OBJECTS - 1)),
        is_modeled=jnp.ones(NUM_OBJECTS, bool),
        is_valid=jnp.ones(NUM_OBJECTS, bool),
        objects_of_interest=jnp.array([True] + [False] * (NUM_OBJECTS - 1)),
        is_controlled=jnp.ones(NUM_OBJECTS, bool),
    )

    # ── Traffic Lights (3 signals) ────────────────────────────────────────
    NUM_TL = 3
    tl_x = np.zeros((NUM_TL, N_TIMESTEPS), np.float32)
    tl_y = np.zeros((NUM_TL, N_TIMESTEPS), np.float32)
    tl_z = np.zeros((NUM_TL, N_TIMESTEPS), np.float32)
    tl_state = np.zeros((NUM_TL, N_TIMESTEPS), np.int32)
    tl_lane = np.zeros((NUM_TL, N_TIMESTEPS), np.int32)
    tl_valid = np.zeros((NUM_TL, N_TIMESTEPS), bool)

    # TL-0: Seg C EW direction (ego approach)
    #   Green 0-10s → Yellow 10-12s → Red 12-22s → Green 22+
    tl_x[0] = 155.0;  tl_y[0] = 254.0;  tl_valid[0] = True
    for t in range(N_TIMESTEPS):
        time_s = t * DT
        if time_s < 10.0:
            tl_state[0, t] = 6    # GO (green)
        elif time_s < 12.0:
            tl_state[0, t] = 5    # CAUTION (yellow)
        elif time_s < 22.0:
            tl_state[0, t] = 4    # STOP (red) — ego arrives ~13s
        else:
            tl_state[0, t] = 6    # GO

    # TL-1: Seg C NS direction (cross-traffic)
    #   Red 0-12s → Green 12-20s → Yellow 20-22s → Red 22+
    tl_x[1] = 175.0;  tl_y[1] = 235.0;  tl_valid[1] = True
    for t in range(N_TIMESTEPS):
        time_s = t * DT
        if time_s < 12.0:
            tl_state[1, t] = 4    # STOP
        elif time_s < 20.0:
            tl_state[1, t] = 6    # GO
        elif time_s < 22.0:
            tl_state[1, t] = 5    # CAUTION
        else:
            tl_state[1, t] = 4    # STOP

    # TL-2: Seg G NS direction
    #   Green 0-30s → Yellow 30-32s → Red 32+
    tl_x[2] = 250.0;  tl_y[2] = 540.0;  tl_valid[2] = True
    for t in range(N_TIMESTEPS):
        time_s = t * DT
        if time_s < 30.0:
            tl_state[2, t] = 6    # GO
        elif time_s < 32.0:
            tl_state[2, t] = 5    # CAUTION
        else:
            tl_state[2, t] = 4    # STOP — ego arrives ~33s

    traffic_lights = datatypes.TrafficLights(
        x=jnp.array(tl_x), y=jnp.array(tl_y), z=jnp.array(tl_z),
        state=jnp.array(tl_state),
        lane_ids=jnp.array(tl_lane),
        valid=jnp.array(tl_valid),
    )

    print(f"Scenario assembled: {NUM_OBJECTS} actors, {NUM_TL} traffic lights")

    return datatypes.SimulatorState(
        sim_trajectory=traj,
        log_trajectory=traj,
        log_traffic_light=traffic_lights,
        object_metadata=metadata,
        timestep=jnp.int32(0),
        roadgraph_points=roadgraph,
        sdc_paths=None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main — Simulation Loop
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("  Complex City Planning Simulation — 32 Actors, 7 Segments")
    print("=" * 65)

    scenario = create_city_scenario()

    dynamics_model = dynamics.StateDynamics()
    obj_idx = jnp.arange(NUM_OBJECTS)

    # ── 6 Actor Control Groups ────────────────────────────────────────────

    # Group 1: IDM for ego + highway vehicles (IDs 0-6)
    idm_actor = agents.IDMRoutePolicy(
        is_controlled_func=lambda state: state.object_metadata.ids <= 6,
    )

    # Group 2: Expert (log replay) for intersection & roundabout actors (IDs 7-14)
    expert_actor = agents.create_expert_actor(
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (state.object_metadata.ids >= 7)
        & (state.object_metadata.ids <= 14),
    )

    # Group 3: Constant speed for background vehicles (IDs 15-20)
    bg_actor = agents.create_constant_speed_actor(
        speed=5.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (state.object_metadata.ids >= 15)
        & (state.object_metadata.ids <= 20),
    )

    # Group 4: Constant speed for pedestrians (IDs 21-26)
    ped_actor = agents.create_constant_speed_actor(
        speed=1.3,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (state.object_metadata.ids >= 21)
        & (state.object_metadata.ids <= 26),
    )

    # Group 5: Constant speed for cyclists (IDs 27-30)
    cyc_actor = agents.create_constant_speed_actor(
        speed=6.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (state.object_metadata.ids >= 27)
        & (state.object_metadata.ids <= 30),
    )

    # Group 6: Static for parked truck (ID 31)
    parked_actor = agents.create_constant_speed_actor(
        speed=0.0,
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: state.object_metadata.ids == 31,
    )

    # ── Environment ───────────────────────────────────────────────────────
    env = _env.BaseEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(), max_num_objects=NUM_OBJECTS,
        ),
    )

    actors = [idm_actor, expert_actor, bg_actor, ped_actor, cyc_actor, parked_actor]

    # ── JIT compile ───────────────────────────────────────────────────────
    print("JIT compiling...")
    jit_step = jax.jit(env.step)
    jit_actions = [jax.jit(a.select_action) for a in actors]

    # ── Run simulation ────────────────────────────────────────────────────
    states = [env.reset(scenario)]
    print(f"Simulating {SIM_STEPS} steps ({SIM_STEPS * DT:.0f} seconds)...")

    for _ in tqdm(range(SIM_STEPS)):
        s = states[-1]
        outputs = [jit_act({}, s, None, None) for jit_act in jit_actions]
        action = agents.merge_actions(outputs)
        states.append(jit_step(s, action))

    # ── Render ────────────────────────────────────────────────────────────
    print("Rendering video...")
    viz_cfg = {
        "front_x": 80,
        "back_x": 60,
        "front_y": 80,
        "back_y": 60,
        "px_per_meter": 2.0,
        "show_agent_id": True,
    }

    imgs = [
        visualization.plot_simulator_state(s, viz_config=viz_cfg)
        for s in tqdm(states)
    ]
    mediapy.write_video("city_planning_sim.mp4", imgs, fps=10)
    print("Video saved as city_planning_sim.mp4")
    print(f"  Frames: {len(imgs)}, Duration: {len(imgs) / 10:.1f}s")
