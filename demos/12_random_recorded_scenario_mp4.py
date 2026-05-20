#!/usr/bin/env python3
"""Demo 12: render a random recorded nuPlan scenario to MP4.

This demo selects a random DB log (or uses an explicitly provided DB) and
uses the shared recorded-scenario rendering utility.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from common.recorded_scenario_renderer import add_recorded_render_args, run_recorded_scenario_render
from common.nuplan_demo_utils import add_common_args, connect, ensure_output_dir, find_db_files, list_tables


REQUIRED_TABLES = {"lidar_pc", "ego_pose", "lidar_box"}


def db_has_required_tables(db_path: Path) -> bool:
    try:
        with connect(db_path) as con:
            return REQUIRED_TABLES.issubset(list_tables(con))
    except Exception:
        return False


def pick_random_db(data_root: Path, explicit_db: Path | None, seed: int | None) -> tuple[Path, int, int]:
    db_files = find_db_files(data_root, explicit_db)
    eligible = [db for db in db_files if db_has_required_tables(db)]
    if not eligible:
        raise RuntimeError(
            "No DB files with required tables were found. "
            "Run scripts/bootstrap_nuplan.sh --profile planner_mini or pass --db /path/to/log.db."
        )

    rng = random.Random(seed)
    picked_index = rng.randrange(len(eligible))
    return eligible[picked_index], picked_index, len(eligible)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_recorded_render_args(parser)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible DB selection.")
    args = parser.parse_args()

    selected_db, picked_index, candidate_count = pick_random_db(args.data_root, args.db, args.seed)
    out_dir = ensure_output_dir(args.output_dir)
    run_output_dir = ensure_output_dir(out_dir / f"12_random_{selected_db.stem}")

    print(f"[demo] randomly selected DB {picked_index + 1}/{candidate_count}: {selected_db}")
    if args.seed is not None:
        print(f"[demo] seed: {args.seed}")

    mp4_path = run_recorded_scenario_render(
        data_root=args.data_root,
        db=selected_db,
        output_dir=run_output_dir,
        show_schema=args.show_schema,
        frames=args.frames,
        fps=args.fps,
        tail_length=args.tail_length,
        track_tail=args.track_tail,
        margin=args.margin,
        output_name="12_random_recorded_scenario.mp4",
    )

    print(f"[demo] wrote: {mp4_path}")
    print("[demo] component demonstrated: Random recorded scenario playback = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())