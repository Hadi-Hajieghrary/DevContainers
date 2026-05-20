#!/usr/bin/env python3
"""Demo 13: run configurable IDM planner on a random scenario and record the simulation.

This demo:
1. Picks a random mini split scenario via the official nuPlan scenario filter
2. Runs closed-loop simulation with IDM planner (configurable parameters)
3. Renders the saved simulation log to MP4 for a fixed time window

Example (Random scenario, default parameters):
python demos/13_idm_simulation_and_record.py \
  --scenarios 3 \
  --target-velocity 10.0 \
  --accel-max 2.5 \
  --decel-max 2.5 \
  --headway-time 1.2 \
  --min-gap 1.5 \
  --record-seconds 30 \
  --fps 10

Reproducible version:
python demos/13_idm_simulation_and_record.py \
  --scenarios 3 \
  --target-velocity 10.0 \
  --accel-max 2.5 \
  --decel-max 2.5 \
  --headway-time 1.2 \
  --min-gap 1.5 \
  --record-seconds 30 \
  --fps 10 \
  --seed 7

"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

# Ensure we can import from the nuPlan devkit when running this demo standalone.
sys.path.insert(0, os.environ.get("NUPLAN_DEVKIT_ROOT", "/workspace/nuplan-devkit"))

from common.nuplan_demo_utils import add_common_args, ensure_output_dir, find_db_files, list_tables
from common.simulated_scenario_renderer import render_simulation_log_mp4




def db_is_readable(db_path: Path) -> bool:
    try:
        with sqlite3.connect(str(db_path)) as con:
            tables = list_tables(con)
            return "lidar_pc" in tables and "ego_pose" in tables
    except Exception:
        return False


def pick_random_log_name(data_root: Path, explicit_db: Path | None, seed: int | None) -> str:
    db_files = find_db_files(data_root, explicit_db)
    eligible = [db for db in db_files if db_is_readable(db)]
    if not eligible:
        raise RuntimeError("No readable mini DBs were found. Run scripts/bootstrap_nuplan.sh --profile planner_mini.")
    rng = random.Random(seed)
    return eligible[rng.randrange(len(eligible))].stem


def run_idm_simulation(
    *,
    experiment_name: str,
    scenario_filter_limit: int,
    log_name: str,
    target_velocity: float,
    accel_max: float,
    decel_max: float,
    headway_time: float,
    min_gap: float,
) -> Path:
    """Run simulation with IDM planner and return the experiment directory."""
    devkit_root = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", "/workspace/nuplan-devkit"))
    exp_root = Path(os.environ.get("NUPLAN_EXP_ROOT", "/workspace/exp"))

    command = [
        sys.executable,
        "nuplan/planning/script/run_simulation.py",
        "+simulation=closed_loop_nonreactive_agents",
        "planner=idm_planner",
        f"planner.idm_planner.target_velocity={target_velocity}",
        f"planner.idm_planner.accel_max={accel_max}",
        f"planner.idm_planner.decel_max={decel_max}",
        f"planner.idm_planner.headway_time={headway_time}",
        f"planner.idm_planner.min_gap_to_lead_agent={min_gap}",
        "scenario_builder=nuplan_mini",
        "scenario_filter=one_continuous_log",
        f"scenario_filter.log_names=[{log_name}]",
        f"scenario_filter.limit_total_scenarios={scenario_filter_limit}",
        "worker=sequential",
        f"experiment_name={experiment_name}",
        f"job_name={experiment_name}",
    ]

    print(f"[demo] simulation command: {' '.join(command)}")
    subprocess.run(command, cwd=str(devkit_root), check=True)
    return exp_root / "exp" / experiment_name / experiment_name


def find_latest_simulation_log(experiment_dir: Path) -> Path:
    """Find the newest serialized simulation log in an experiment directory."""
    candidates = list(experiment_dir.rglob("*.msgpack.xz")) + list(experiment_dir.rglob("*.pkl.xz"))
    if not candidates:
        raise FileNotFoundError(f"No simulation logs found under {experiment_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--target-velocity",
        type=float,
        default=8.0,
        help="IDM target velocity in m/s (free-traffic speed).",
    )
    parser.add_argument(
        "--accel-max",
        type=float,
        default=2.0,
        help="IDM max acceleration in m/s^2.",
    )
    parser.add_argument(
        "--decel-max",
        type=float,
        default=2.0,
        help="IDM max deceleration in m/s^2.",
    )
    parser.add_argument(
        "--headway-time",
        type=float,
        default=1.5,
        help="IDM headway time in seconds.",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=2.0,
        help="IDM minimum gap to lead agent in meters.",
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        default=1,
        help="Number of random scenarios to simulate (1 is recommended for recording).",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=30.0,
        help="How many seconds of the simulation log to render to MP4. Use 0 to render all available samples.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output MP4.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=40.0,
        help="Meters of view margin around ego in the rendered video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for scenario-log selection. Omit it to pick a different readable log each run.",
    )
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.output_dir)
    experiment_name = f"demo_13_idm_v{args.target_velocity:.1f}_a{args.accel_max:.1f}"
    video_path = out_dir / f"{experiment_name}.mp4"
    selected_log_name = pick_random_log_name(args.data_root, args.db, seed=args.seed)

    print("[demo] running IDM simulation with configurable parameters:")
    print(f"[demo]   target_velocity: {args.target_velocity} m/s")
    print(f"[demo]   accel_max: {args.accel_max} m/s^2")
    print(f"[demo]   decel_max: {args.decel_max} m/s^2")
    print(f"[demo]   headway_time: {args.headway_time} s")
    print(f"[demo]   min_gap: {args.min_gap} m")
    print(f"[demo]   scenarios: {args.scenarios}")
    print(f"[demo]   record_seconds: {args.record_seconds}")
    print(f"[demo]   fps: {args.fps}")
    print(f"[demo] experiment: {experiment_name}")
    if args.seed is not None:
        print(f"[demo]   seed: {args.seed}")
    print(f"[demo] selected log: {selected_log_name}")

    experiment_dir = run_idm_simulation(
        experiment_name=experiment_name,
        scenario_filter_limit=args.scenarios,
        log_name=selected_log_name,
        target_velocity=args.target_velocity,
        accel_max=args.accel_max,
        decel_max=args.decel_max,
        headway_time=args.headway_time,
        min_gap=args.min_gap,
    )

    simulation_log = find_latest_simulation_log(experiment_dir)
    print(f"[demo] selected simulation log: {simulation_log}")
    render_simulation_log_mp4(
        simulation_log,
        video_path,
        fps=args.fps,
        record_seconds=args.record_seconds,
        margin_m=args.margin,
    )

    print("\n[demo] simulation complete.")
    print(f"[demo] artifacts written to: {experiment_dir}")
    print(f"[demo] mp4 written to: {video_path}")
    print("[demo] next steps:")
    print("[demo]   - run: python demos/07_metrics_engine_summary.py")
    print("[demo]   - run: bash demos/08_launch_nuboard.sh")
    print("[demo] component demonstrated: IDM planner simulation + MP4 recording = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
