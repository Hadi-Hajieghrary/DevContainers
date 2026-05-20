#!/usr/bin/env python3
"""Demo 01: read and plot ego trajectory from nuPlan SQLite log DBs.

Component demonstrated: Ego trajectories / ego pose-state.

What this proves:
  - You do not need camera/LiDAR blobs to access ego state.
  - Ego pose is reachable through lidar_pc.ego_pose_token -> ego_pose.token.
  - The output can directly feed path/trajectory planners, MPC, SQP, or rulebooks.
"""

from __future__ import annotations

import argparse
import math
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
    maybe_float,
    print_db_selection,
    table_columns,
    yaw_from_quaternion,
)


def load_ego_trajectory(db_path: Path, limit: int, show_schema: bool) -> pd.DataFrame:
    with connect(db_path) as con:
        ego_cols = table_columns(con, "ego_pose")
        lidar_cols = table_columns(con, "lidar_pc")
        if show_schema:
            print("ego_pose columns:", ego_cols)
            print("lidar_pc columns:", lidar_cols)

        x_col = choose_col(ego_cols, ["x", "translation_x", "ego_pose_x"])
        y_col = choose_col(ego_cols, ["y", "translation_y", "ego_pose_y"])
        qw_col = choose_col(ego_cols, ["qw", "rotation_w"])
        qx_col = choose_col(ego_cols, ["qx", "rotation_x"])
        qy_col = choose_col(ego_cols, ["qy", "rotation_y"])
        qz_col = choose_col(ego_cols, ["qz", "rotation_z"])
        vx_col = choose_col(ego_cols, ["vx", "velocity_x", "ego_velocity_x"])
        vy_col = choose_col(ego_cols, ["vy", "velocity_y", "ego_velocity_y"])
        timestamp_col = choose_col(lidar_cols, ["timestamp"])

        if not x_col or not y_col:
            raise RuntimeError(f"Could not identify ego x/y columns in ego_pose: {ego_cols}")
        if not timestamp_col:
            raise RuntimeError(f"Could not identify timestamp column in lidar_pc: {lidar_cols}")

        select_parts = [
            f"lidar_pc.{timestamp_col} AS timestamp_us",
            f"ego_pose.{x_col} AS x",
            f"ego_pose.{y_col} AS y",
        ]
        for name, col in [("qw", qw_col), ("qx", qx_col), ("qy", qy_col), ("qz", qz_col), ("vx", vx_col), ("vy", vy_col)]:
            if col:
                select_parts.append(f"ego_pose.{col} AS {name}")

        query = f"""
            SELECT {', '.join(select_parts)}
            FROM lidar_pc
            JOIN ego_pose ON lidar_pc.ego_pose_token = ego_pose.token
            ORDER BY lidar_pc.timestamp ASC
            LIMIT ?
        """
        rows = con.execute(query, (limit,)).fetchall()

    if not rows:
        raise RuntimeError("No ego trajectory rows found in selected DB.")

    df = pd.DataFrame([dict(row) for row in rows])
    if {"qw", "qx", "qy", "qz"}.issubset(df.columns):
        df["yaw_rad"] = [
            yaw_from_quaternion(maybe_float(r.qw), maybe_float(r.qx), maybe_float(r.qy), maybe_float(r.qz))
            for r in df.itertuples(index=False)
        ]
    else:
        df["yaw_rad"] = float("nan")

    if {"vx", "vy"}.issubset(df.columns):
        df["speed_mps"] = (df["vx"] ** 2 + df["vy"] ** 2).pow(0.5)
    else:
        df["speed_mps"] = float("nan")

    df["t_s"] = (df["timestamp_us"] - df["timestamp_us"].iloc[0]) / 1e6
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    db_files = find_db_files(args.data_root, args.db)
    db = first_db_with_tables(db_files, ["lidar_pc", "ego_pose"])
    print_db_selection(db.path, db.tables)

    out_dir = ensure_output_dir(args.output_dir)
    df = load_ego_trajectory(db.path, args.limit, args.show_schema)

    csv_path = out_dir / "01_ego_trajectory.csv"
    png_path = out_dir / "01_ego_trajectory_xy.png"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(df["x"], df["y"], linewidth=2)
    plt.scatter([df["x"].iloc[0]], [df["y"].iloc[0]], label="start")
    plt.scatter([df["x"].iloc[-1]], [df["y"].iloc[-1]], label="end")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("nuPlan ego trajectory from SQLite log DB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)

    print(df.head(10).to_string(index=False))
    print(f"[demo] wrote: {csv_path}")
    print(f"[demo] wrote: {png_path}")
    print("[demo] component demonstrated: Ego trajectories = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
