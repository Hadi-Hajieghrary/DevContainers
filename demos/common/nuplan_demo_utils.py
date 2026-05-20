#!/usr/bin/env python3
"""Utility functions shared by the nuPlan planner-input demos.

The goal of these demos is to expose the planner-level data that matters for
behavior/path/trajectory-planning research without requiring raw camera/LiDAR
sensor blobs.  We therefore read the SQLite log DBs directly for lightweight
examples, and we use official nuPlan scripts for simulation, metrics, and
nuBoard.

These helpers intentionally avoid hard-coding too much of the nuPlan schema.
The public nuPlan DB schema is stable, but small differences exist across
versions.  The helpers inspect available tables/columns and fail with useful
messages instead of crashing deep in a query.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

DEFAULT_DATA_ROOT = Path(os.environ.get("NUPLAN_DATA_ROOT", "/workspace/data"))
DEFAULT_EXP_ROOT = Path(os.environ.get("NUPLAN_EXP_ROOT", "/workspace/exp"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("NUPLAN_DEMO_OUTPUT_ROOT", "/workspace/nuplan-project/demos/outputs"))


@dataclass(frozen=True)
class DbInfo:
    """Small descriptor for one nuPlan SQLite DB file."""

    path: Path
    tables: set[str]


def default_db_roots(data_root: Path = DEFAULT_DATA_ROOT) -> list[Path]:
    """Return likely places where nuPlan DB files live."""

    return [
        data_root / "nuplan-v1.1" / "splits" / "mini",
        data_root / "nuplan-v1.1" / "splits" / "trainval",
        data_root / "nuplan-v1.1" / "splits" / "train",
        data_root / "nuplan-v1.1" / "splits" / "val",
        data_root / "nuplan-v1.1" / "splits" / "test",
        data_root,
    ]


def find_db_files(data_root: Path = DEFAULT_DATA_ROOT, explicit_db: Optional[Path] = None) -> list[Path]:
    """Find candidate nuPlan SQLite DB files.

    Parameters
    ----------
    data_root:
        Root of the mounted nuPlan dataset.
    explicit_db:
        Optional user-specified DB path.  If present, only this DB is returned.
    """

    if explicit_db is not None:
        db_path = explicit_db.expanduser().resolve()
        if not db_path.exists():
            raise FileNotFoundError(f"Requested DB does not exist: {db_path}")
        return [db_path]

    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in default_db_roots(data_root):
        if not root.exists():
            continue
        for db in root.rglob("*.db"):
            db = db.resolve()
            if db not in seen:
                candidates.append(db)
                seen.add(db)

    return sorted(candidates)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite DB with row access by column name."""

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def list_tables(con: sqlite3.Connection) -> set[str]:
    rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {str(row[0]) for row in rows}


def table_columns(con: sqlite3.Connection, table: str) -> list[str]:
    rows = con.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
    return [str(row[1]) for row in rows]


def quote_ident(identifier: str) -> str:
    """Quote a SQLite identifier safely."""

    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def qcol(table: str, col: str) -> str:
    return f"{quote_ident(table)}.{quote_ident(col)}"


def choose_col(columns: Sequence[str], options: Sequence[str]) -> Optional[str]:
    """Choose the first available column from an ordered preference list."""

    colset = set(columns)
    for opt in options:
        if opt in colset:
            return opt
    return None


def require_tables(con: sqlite3.Connection, required: Iterable[str]) -> None:
    tables = list_tables(con)
    missing = [t for t in required if t not in tables]
    if missing:
        raise RuntimeError(
            "The selected DB is missing required tables: "
            + ", ".join(missing)
            + f". Available tables: {', '.join(sorted(tables))}"
        )


def first_db_with_tables(db_files: Sequence[Path], required_tables: Iterable[str]) -> DbInfo:
    """Pick the first DB that has all required tables."""

    required = set(required_tables)
    if not db_files:
        raise FileNotFoundError(
            "No nuPlan .db files were found. Run `scripts/bootstrap_nuplan.sh --profile planner_mini` "
            "or pass `--db /path/to/log.db`."
        )

    inspected: list[str] = []
    for db in db_files:
        try:
            with connect(db) as con:
                tables = list_tables(con)
                inspected.append(f"{db.name}: {len(tables)} tables")
                if required.issubset(tables):
                    return DbInfo(path=db, tables=tables)
        except sqlite3.DatabaseError as exc:
            inspected.append(f"{db.name}: not readable as SQLite ({exc})")

    raise RuntimeError(
        "Could not find a DB with required tables "
        + ", ".join(sorted(required))
        + ". Inspected: "
        + "; ".join(inspected[:12])
    )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Mounted nuPlan dataset root.")
    parser.add_argument("--db", type=Path, default=None, help="Optional explicit SQLite log DB.")
    parser.add_argument("--limit", type=int, default=300, help="Maximum rows/frames to load.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory for plots/CSVs.")
    parser.add_argument("--show-schema", action="store_true", help="Print relevant table columns before running.")
    return parser


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def yaw_from_quaternion(w: float, x: float, y: float, z: float) -> float:
    """Compute yaw from a quaternion using the common w,x,y,z convention."""

    # yaw around z-axis
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def maybe_float(value: object, default: float = float("nan")) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def print_db_selection(db: Path, tables: Iterable[str]) -> None:
    print(f"[demo] selected DB: {db}")
    print(f"[demo] available tables: {len(list(tables))}")
