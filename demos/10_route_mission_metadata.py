#!/usr/bin/env python3
"""Demo 10: inspect route / mission-goal metadata available in nuPlan DBs.

Component demonstrated: Route / mission metadata.

In the official nuPlan framework, route information is most commonly consumed
through scenario objects.  This DB-level demo is intentionally diagnostic: it
finds route/mission/goal-related tables or columns in the selected DB and dumps
a small sample so you can see what exists before wiring planner code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common.nuplan_demo_utils import (
    add_common_args,
    connect,
    ensure_output_dir,
    find_db_files,
    first_db_with_tables,
    list_tables,
    print_db_selection,
    table_columns,
)

KEYWORDS = ("route", "mission", "goal", "roadblock")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()

    db_files = find_db_files(args.data_root, args.db)
    db = first_db_with_tables(db_files, ["lidar_pc", "ego_pose"])
    print_db_selection(db.path, db.tables)

    result: dict = {"source_db": str(db.path), "matching_tables": {}, "matching_columns": {}}
    with connect(db.path) as con:
        tables = sorted(list_tables(con))
        for table in tables:
            cols = table_columns(con, table)
            table_matches = any(k in table.lower() for k in KEYWORDS)
            col_matches = [c for c in cols if any(k in c.lower() for k in KEYWORDS)]
            if table_matches:
                try:
                    rows = con.execute(f"SELECT * FROM {table} LIMIT ?", (args.limit,)).fetchall()
                    result["matching_tables"][table] = [dict(row) for row in rows]
                except Exception as exc:
                    result["matching_tables"][table] = f"Could not read rows: {type(exc).__name__}: {exc}"
            if col_matches:
                result["matching_columns"][table] = col_matches

    out_dir = ensure_output_dir(args.output_dir)
    out_path = out_dir / "10_route_mission_metadata.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print("[demo] route/mission/goal-related tables found:")
    for table, rows in result["matching_tables"].items():
        print(f"  - {table}: {len(rows) if isinstance(rows, list) else rows}")
    print("[demo] route/mission/goal-related columns found:")
    for table, cols in result["matching_columns"].items():
        print(f"  - {table}: {cols}")
    if not result["matching_tables"] and not result["matching_columns"]:
        print("[demo] No direct route/mission DB fields were discovered in this DB version.")
        print("[demo] Use official scenario objects in the devkit for route-roadblock access during simulation.")
    print(f"[demo] wrote: {out_path}")
    print("[demo] component demonstrated: Route / mission metadata inspection = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
