#!/usr/bin/env python3
"""Demo 04: read traffic-light states from nuPlan SQLite log DBs.

Component demonstrated: Traffic light states.

Traffic-light annotations are planner-level data in nuPlan DBs. They do not
require downloading camera images or training a traffic-light detector.
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
    print_db_selection,
    table_columns,
)


def load_traffic_lights(db_path: Path, limit: int, show_schema: bool) -> pd.DataFrame:
    with connect(db_path) as con:
        tables = list_tables(con)
        candidates = [t for t in ["traffic_light_status", "traffic_light_statuses"] if t in tables]
        if not candidates:
            raise RuntimeError(
                "No traffic-light status table was found. Available tables: " + ", ".join(sorted(tables))
            )
        table = candidates[0]
        tl_cols = table_columns(con, table)
        lp_cols = table_columns(con, "lidar_pc") if "lidar_pc" in tables else []
        if show_schema:
            print(f"{table} columns:", tl_cols)
            print("lidar_pc columns:", lp_cols)

        status_col = choose_col(tl_cols, ["status", "traffic_light_status_type", "type"])
        lane_col = choose_col(tl_cols, ["lane_connector_id", "lane_id", "traffic_light_lane_connector_id"])
        lp_token_col = choose_col(tl_cols, ["lidar_pc_token"])
        timestamp_col = choose_col(lp_cols, ["timestamp"])

        select_parts = []
        if status_col:
            select_parts.append(f"tl.{status_col} AS status")
        if lane_col:
            select_parts.append(f"tl.{lane_col} AS lane_connector_id")
        if lp_token_col:
            select_parts.append(f"tl.{lp_token_col} AS lidar_pc_token")
        if timestamp_col and lp_token_col:
            select_parts.append(f"lp.{timestamp_col} AS timestamp_us")

        if not select_parts:
            select_parts = ["tl.*"]

        if lp_token_col and timestamp_col:
            query = f"""
                SELECT {', '.join(select_parts)}
                FROM {table} AS tl
                LEFT JOIN lidar_pc AS lp ON tl.{lp_token_col} = lp.token
                ORDER BY lp.{timestamp_col} ASC
                LIMIT ?
            """
        else:
            query = f"SELECT {', '.join(select_parts)} FROM {table} AS tl LIMIT ?"

        rows = con.execute(query, (limit,)).fetchall()

    if not rows:
        raise RuntimeError("Traffic-light table exists, but no rows were returned for this DB/log.")
    return pd.DataFrame([dict(row) for row in rows])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    db_files = find_db_files(args.data_root, args.db)
    db = first_db_with_tables(db_files, ["traffic_light_status"])
    print_db_selection(db.path, db.tables)

    out_dir = ensure_output_dir(args.output_dir)
    df = load_traffic_lights(db.path, args.limit, args.show_schema)
    csv_path = out_dir / "04_traffic_light_states.csv"
    df.to_csv(csv_path, index=False)

    print(df.head(30).to_string(index=False))
    if "status" in df.columns:
        print("\n[demo] status counts:")
        print(df["status"].value_counts(dropna=False).to_string())
    print(f"[demo] wrote: {csv_path}")
    print("[demo] component demonstrated: Traffic light states = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
