#!/usr/bin/env python3
"""Demo 09: combine ego, agents, maps, and traffic lights into one planner input snapshot.

Component demonstrated: planner input assembly.

This is the bridge to your own Rulebook / W-SQP / MPC planner.  It creates a
compact JSON snapshot with:
  - ego state;
  - agent boxes around the same frame;
  - traffic-light annotations at the same frame if available;
  - map root metadata.

The output is intentionally simple and planner-agnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from common.nuplan_demo_utils import (
    DEFAULT_DATA_ROOT,
    add_common_args,
    choose_col,
    connect,
    ensure_output_dir,
    find_db_files,
    first_db_with_tables,
    list_tables,
    print_db_selection,
    table_columns,
)


def build_snapshot(db_path: Path, frame_index: int, data_root: Path, show_schema: bool) -> dict:
    with connect(db_path) as con:
        tables = list_tables(con)
        ego_cols = table_columns(con, "ego_pose")
        lp_cols = table_columns(con, "lidar_pc")
        lb_cols = table_columns(con, "lidar_box") if "lidar_box" in tables else []
        if show_schema:
            print("ego_pose columns:", ego_cols)
            print("lidar_pc columns:", lp_cols)
            print("lidar_box columns:", lb_cols)

        timestamp_col = choose_col(lp_cols, ["timestamp"])
        x_col = choose_col(ego_cols, ["x", "translation_x", "ego_pose_x"])
        y_col = choose_col(ego_cols, ["y", "translation_y", "ego_pose_y"])
        vx_col = choose_col(ego_cols, ["vx", "velocity_x", "ego_velocity_x"])
        vy_col = choose_col(ego_cols, ["vy", "velocity_y", "ego_velocity_y"])
        if not timestamp_col or not x_col or not y_col:
            raise RuntimeError("Could not identify required lidar_pc/ego_pose columns.")

        frame = con.execute(
            f"""
            SELECT lp.token AS lidar_pc_token, lp.{timestamp_col} AS timestamp_us,
                   ep.{x_col} AS x, ep.{y_col} AS y
                   {',' if vx_col else ''} {f'ep.{vx_col} AS vx' if vx_col else ''}
                   {',' if vy_col else ''} {f'ep.{vy_col} AS vy' if vy_col else ''}
            FROM lidar_pc AS lp
            JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token
            ORDER BY lp.{timestamp_col} ASC
            LIMIT 1 OFFSET ?
            """,
            (frame_index,),
        ).fetchone()
        if frame is None:
            raise RuntimeError(f"No lidar frame at index {frame_index}")

        lidar_pc_token = frame["lidar_pc_token"]
        agents = []
        if "lidar_box" in tables:
            lb_x = choose_col(lb_cols, ["x", "translation_x", "center_x"])
            lb_y = choose_col(lb_cols, ["y", "translation_y", "center_y"])
            lb_len = choose_col(lb_cols, ["length", "size_x"])
            lb_wid = choose_col(lb_cols, ["width", "size_y"])
            lb_yaw = choose_col(lb_cols, ["yaw", "heading"])
            lb_track = choose_col(lb_cols, ["track_token"])
            select = [f"{lb_x} AS x", f"{lb_y} AS y"] if lb_x and lb_y else ["*"]
            for name, col in [("length", lb_len), ("width", lb_wid), ("yaw", lb_yaw), ("track_token", lb_track)]:
                if col:
                    select.append(f"{col} AS {name}")
            rows = con.execute(
                f"SELECT {', '.join(select)} FROM lidar_box WHERE lidar_pc_token = ? LIMIT 200",
                (lidar_pc_token,),
            ).fetchall()
            agents = [dict(row) for row in rows]

        traffic_lights = []
        if "traffic_light_status" in tables:
            tl_cols = table_columns(con, "traffic_light_status")
            tl_lp = choose_col(tl_cols, ["lidar_pc_token"])
            if tl_lp:
                rows = con.execute(
                    "SELECT * FROM traffic_light_status WHERE lidar_pc_token = ? LIMIT 200",
                    (lidar_pc_token,),
                ).fetchall()
                traffic_lights = [dict(row) for row in rows]

    map_root = Path(os.environ.get("NUPLAN_MAPS_ROOT", str(data_root / "maps")))
    snapshot = {
        "source_db": str(db_path),
        "frame_index": frame_index,
        "timestamp_us": frame["timestamp_us"],
        "ego_state": dict(frame),
        "agents": agents,
        "traffic_lights": traffic_lights,
        "map_root": str(map_root),
        "map_available": map_root.exists(),
        "notes": [
            "This JSON is a compact planner-input snapshot, not a replacement for the full nuPlan API.",
            "It is sufficient to start wiring a Rulebook / W-SQP / MPC planner prototype.",
        ],
    }
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--frame-index", type=int, default=20)
    args = parser.parse_args()

    db_files = find_db_files(args.data_root, args.db)
    db = first_db_with_tables(db_files, ["lidar_pc", "ego_pose"])
    print_db_selection(db.path, db.tables)

    out_dir = ensure_output_dir(args.output_dir)
    snapshot = build_snapshot(db.path, args.frame_index, args.data_root, args.show_schema)
    out_path = out_dir / "09_planner_input_snapshot.json"
    out_path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "source_db": snapshot["source_db"],
        "frame_index": snapshot["frame_index"],
        "timestamp_us": snapshot["timestamp_us"],
        "ego_state_keys": list(snapshot["ego_state"].keys()),
        "num_agents": len(snapshot["agents"]),
        "num_traffic_light_records": len(snapshot["traffic_lights"]),
        "map_available": snapshot["map_available"],
    }, indent=2))
    print(f"[demo] wrote: {out_path}")
    print("[demo] components demonstrated: ego + agents + maps + traffic lights + route/scenario-ready planner input = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
