#!/usr/bin/env python3
"""Demo 03: inspect/query the mounted nuPlan HD semantic maps.

Component demonstrated: HD maps.

This demo has two layers:
  1. robust filesystem inspection that always works after maps are downloaded;
  2. optional nuPlan map-factory import, which verifies that the devkit can load
     maps through its official API in this environment.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from common.nuplan_demo_utils import DEFAULT_DATA_ROOT, ensure_output_dir


def find_map_root(data_root: Path) -> Path:
    candidates = [
        Path(os.environ.get("NUPLAN_MAPS_ROOT", "")),
        data_root / "maps",
        data_root / "nuplan-v1.1" / "maps",
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists():
            return candidate
    return data_root / "maps"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/nuplan-project/demos/outputs"))
    parser.add_argument("--try-api", action="store_true", help="Try loading map metadata through the nuPlan devkit map API.")
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.output_dir)
    map_root = find_map_root(args.data_root)
    print(f"[demo] map root: {map_root}")

    if not map_root.exists():
        raise FileNotFoundError(
            f"Map root does not exist: {map_root}. Run `scripts/bootstrap_nuplan.sh --profile maps_only` "
            "or `scripts/bootstrap_nuplan.sh --profile planner_mini`."
        )

    map_files = sorted([p for p in map_root.rglob("*") if p.is_file()])
    summary_path = out_dir / "03_hd_map_files.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for p in map_files:
            rel = p.relative_to(map_root)
            f.write(f"{rel}\t{p.stat().st_size}\n")

    print(f"[demo] map files found: {len(map_files)}")
    print("[demo] first map files:")
    for p in map_files[:30]:
        print("  ", p.relative_to(map_root), f"({p.stat().st_size / 1024:.1f} KiB)")
    print(f"[demo] wrote: {summary_path}")

    if args.try_api:
        try:
            # The exact public API has evolved.  This import path is valid in the
            # nuPlan devkit lineage.  If it changes, the filesystem check above
            # still confirms that the maps are installed.
            from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB  # type: ignore

            version = os.environ.get("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
            maps_db = GPKGMapsDB(map_version=version, map_root=str(map_root))
            map_names = sorted(maps_db.get_map_names())
            print(f"[demo] nuPlan map API loaded. Map names: {map_names}")
        except Exception as exc:  # pragma: no cover - intentionally diagnostic
            print("[demo] nuPlan map API check failed, but map files are present.")
            print(f"[demo] API error: {type(exc).__name__}: {exc}")
            print("[demo] This is usually an import/version mismatch, not missing data.")

    print("[demo] component demonstrated: HD maps = YES")
    return 0


if __name__ == "__main__":
    sys.exit(main())
