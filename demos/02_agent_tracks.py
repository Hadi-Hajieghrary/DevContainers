#!/usr/bin/env python3
"""Demo 02: read tracked agents/object boxes from nuPlan SQLite log DBs.

Component demonstrated: Agent tracks.

This script reads tracked boxes attached to lidar_pc frames. These are the
planner-level object tracks/obstacles you usually want for trajectory planning.
No raw point cloud or image archive is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.nuplan_demo_utils import (
    add_common_args,
    choose_col,
    connect,
    ensure_output_dir,
    find_db_files,
    first_db_with_tables,
    print_db_selection,
    table_columns,
)


def load_agent_boxes(db_path: Path, frame_limit: int, show_schema: bool) -> pd.DataFrame:
    with connect(db_path) as con:
        lb_cols = table_columns(con, "lidar_box")
        lp_cols = table_columns(con, "lidar_pc")
        tr_cols = table_columns(con, "track") if "track" in {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()} else []
        if show_schema:
            print("lidar_box columns:", lb_cols)
            print("lidar_pc columns:", lp_cols)
            print("track columns:", tr_cols)

        x_col = choose_col(lb_cols, ["x", "translation_x", "center_x"])
        y_col = choose_col(lb_cols, ["y", "translation_y", "center_y"])
        yaw_col = choose_col(lb_cols, ["yaw", "heading"])
        length_col = choose_col(lb_cols, ["length", "size_x"])
        width_col = choose_col(lb_cols, ["width", "size_y"])
        track_col = choose_col(lb_cols, ["track_token"])
        token_col = choose_col(lb_cols, ["token"])
        timestamp_col = choose_col(lp_cols, ["timestamp"])

        if not x_col or not y_col or not timestamp_col:
            raise RuntimeError(f"Could not identify lidar_box x/y/timestamp columns. lidar_box={lb_cols}, lidar_pc={lp_cols}")

        select_parts = [
            f"lidar_pc.{timestamp_col} AS timestamp_us",
            f"lidar_box.{x_col} AS x",
            f"lidar_box.{y_col} AS y",
        ]
        for name, col in [("yaw", yaw_col), ("length", length_col), ("width", width_col), ("track_token", track_col), ("box_token", token_col)]:
            if col:
                select_parts.append(f"lidar_box.{col} AS {name}")

        # Restrict to a small number of lidar frames first, then collect all boxes
        # in those frames.  This is much faster and easier to inspect than scanning
        # the entire log.
        query = f"""
            WITH selected_frames AS (
                SELECT token, {timestamp_col} AS timestamp_us
                FROM lidar_pc
                ORDER BY {timestamp_col} ASC
                LIMIT ?
            )
            SELECT {', '.join(select_parts)}
            FROM lidar_box
            JOIN selected_frames ON lidar_box.lidar_pc_token = selected_frames.token
            JOIN lidar_pc ON lidar_pc.token = selected_frames.token
            ORDER BY lidar_pc.{timestamp_col} ASC
        """
        rows = con.execute(query, (frame_limit,)).fetchall()

    if not rows:
        raise RuntimeError("No agent/box rows found in selected DB.")
    df = pd.DataFrame([dict(row) for row in rows])
    df["t_s"] = (df["timestamp_us"] - df["timestamp_us"].iloc[0]) / 1e6
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--frames", type=int, default=20, help="Number of lidar_pc frames to inspect.")
    args = parser.parse_args()

    db_files = find_db_files(args.data_root, args.db)
    db = first_db_with_tables(db_files, ["lidar_pc", "lidar_box"])
    print_db_selection(db.path, db.tables)

    out_dir = ensure_output_dir(args.output_dir)
    df = load_agent_boxes(db.path, args.frames, args.show_schema)
    csv_path = out_dir / "02_agent_tracks.csv"
    png_path = out_dir / "02_agent_tracks_xy.png"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))
    # Plot at most 2000 boxes to keep the figure readable.
    plot_df = df.head(2000)
    plt.scatter(plot_df["x"], plot_df["y"], s=10, alpha=0.55)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Tracked agent boxes from first {args.frames} lidar frames")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)

    print(df.head(20).to_string(index=False))
    print(f"[demo] frames inspected: {args.frames}")
    print(f"[demo] boxes loaded: {len(df)}")
    print(f"[demo] wrote: {csv_path}")
    print(f"[demo] wrote: {png_path}")
    print("[demo] component demonstrated: Agent tracks = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
