#!/usr/bin/env python3
"""Demo 05: lightweight scenario extraction/mining from planner-level DB data.

Component demonstrated: Scenario extraction.

Official nuPlan scenario extraction is normally used through the devkit's
scenario builders and Hydra configs.  This lightweight demo mines simple
candidate windows directly from the DB so that you can see the principle before
running a full simulation:
  - stop/slow ego windows;
  - high nearby-agent-count windows;
  - windows with traffic-light annotations.

For the official extraction path, run:
    demos/scripts/05_official_scenario_extraction_smoke.sh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from common.nuplan_demo_utils import (
    add_common_args,
    choose_col,
    connect,
    ensure_output_dir,
    find_db_files,
    first_db_with_tables,
    list_tables,
    maybe_float,
    print_db_selection,
    table_columns,
)


def mine_candidates(db_path: Path, limit: int, show_schema: bool) -> pd.DataFrame:
    with connect(db_path) as con:
        tables = list_tables(con)
        ego_cols = table_columns(con, "ego_pose")
        lp_cols = table_columns(con, "lidar_pc")
        if show_schema:
            print("ego_pose columns:", ego_cols)
            print("lidar_pc columns:", lp_cols)

        vx_col = choose_col(ego_cols, ["vx", "velocity_x", "ego_velocity_x"])
        vy_col = choose_col(ego_cols, ["vy", "velocity_y", "ego_velocity_y"])
        x_col = choose_col(ego_cols, ["x", "translation_x", "ego_pose_x"])
        y_col = choose_col(ego_cols, ["y", "translation_y", "ego_pose_y"])
        timestamp_col = choose_col(lp_cols, ["timestamp"])
        if not timestamp_col:
            raise RuntimeError("Could not identify lidar_pc timestamp column.")

        select_parts = [f"lp.token AS lidar_pc_token", f"lp.{timestamp_col} AS timestamp_us"]
        if x_col:
            select_parts.append(f"ep.{x_col} AS ego_x")
        if y_col:
            select_parts.append(f"ep.{y_col} AS ego_y")
        if vx_col:
            select_parts.append(f"ep.{vx_col} AS vx")
        if vy_col:
            select_parts.append(f"ep.{vy_col} AS vy")

        rows = con.execute(
            f"""
            SELECT {', '.join(select_parts)}
            FROM lidar_pc AS lp
            JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token
            ORDER BY lp.{timestamp_col} ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        frames = pd.DataFrame([dict(row) for row in rows])

        if frames.empty:
            raise RuntimeError("No frames found for scenario mining.")

        # Agent count per frame.
        if "lidar_box" in tables:
            counts = con.execute(
                """
                SELECT lidar_pc_token, COUNT(*) AS agent_box_count
                FROM lidar_box
                GROUP BY lidar_pc_token
                """
            ).fetchall()
            count_df = pd.DataFrame([dict(row) for row in counts])
            frames = frames.merge(count_df, on="lidar_pc_token", how="left")
        else:
            frames["agent_box_count"] = 0

        # Traffic light presence per frame.
        tl_table = "traffic_light_status" if "traffic_light_status" in tables else None
        if tl_table:
            tl_cols = table_columns(con, tl_table)
            tl_lp_col = choose_col(tl_cols, ["lidar_pc_token"])
            if tl_lp_col:
                tl_counts = con.execute(
                    f"""
                    SELECT {tl_lp_col} AS lidar_pc_token, COUNT(*) AS traffic_light_count
                    FROM {tl_table}
                    GROUP BY {tl_lp_col}
                    """
                ).fetchall()
                tl_df = pd.DataFrame([dict(row) for row in tl_counts])
                frames = frames.merge(tl_df, on="lidar_pc_token", how="left")
            else:
                frames["traffic_light_count"] = 0
        else:
            frames["traffic_light_count"] = 0

    frames["agent_box_count"] = frames["agent_box_count"].fillna(0).astype(int)
    frames["traffic_light_count"] = frames["traffic_light_count"].fillna(0).astype(int)
    frames["t_s"] = (frames["timestamp_us"] - frames["timestamp_us"].iloc[0]) / 1e6

    if {"vx", "vy"}.issubset(frames.columns):
        frames["ego_speed_mps"] = (frames["vx"] ** 2 + frames["vy"] ** 2).pow(0.5)
    else:
        frames["ego_speed_mps"] = float("nan")

    # Simple labels.  These are not official nuPlan scenario types; they are
    # deliberately transparent examples of how scenario extraction uses low-level
    # attributes.
    agent_threshold = max(5, int(frames["agent_box_count"].quantile(0.90)))
    labels = []
    for row in frames.itertuples(index=False):
        row_labels: list[str] = []
        speed = maybe_float(getattr(row, "ego_speed_mps", float("nan")))
        if speed == speed and speed < 0.5:
            row_labels.append("ego_stop_or_creep")
        if int(getattr(row, "agent_box_count")) >= agent_threshold:
            row_labels.append("dense_agent_context")
        if int(getattr(row, "traffic_light_count")) > 0:
            row_labels.append("traffic_light_context")
        labels.append(";".join(row_labels) if row_labels else "ordinary_context")
    frames["demo_scenario_label"] = labels
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    db_files = find_db_files(args.data_root, args.db)
    db = first_db_with_tables(db_files, ["lidar_pc", "ego_pose"])
    print_db_selection(db.path, db.tables)

    out_dir = ensure_output_dir(args.output_dir)
    df = mine_candidates(db.path, args.limit, args.show_schema)
    csv_path = out_dir / "05_scenario_extraction_candidates.csv"
    df.to_csv(csv_path, index=False)

    print("[demo] mined scenario-label counts:")
    print(df["demo_scenario_label"].value_counts().to_string())
    print("\n[demo] candidate rows:")
    print(df[df["demo_scenario_label"] != "ordinary_context"].head(25).to_string(index=False))
    print(f"[demo] wrote: {csv_path}")
    print("[demo] component demonstrated: Scenario extraction = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
