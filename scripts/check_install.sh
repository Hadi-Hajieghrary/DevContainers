#!/usr/bin/env bash
set -euo pipefail
source /home/vscode/.venv/bin/activate

echo "Python: $(python --version)"
echo "NUPLAN_DATA_ROOT=${NUPLAN_DATA_ROOT}"
echo "NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT}"
echo "NUPLAN_EXP_ROOT=${NUPLAN_EXP_ROOT}"
python - <<'PY'
import importlib, os
from pathlib import Path
for pkg in ["nuplan", "torch", "hydra", "bokeh", "shapely"]:
    m = importlib.import_module(pkg)
    print(f"{pkg}: OK ({getattr(m, '__version__', 'no __version__')})")
data_root = Path(os.environ["NUPLAN_DATA_ROOT"])
print("mini db count:", len(list(data_root.glob("nuplan-v1.1/splits/mini/*.db"))))
print("map json exists:", (data_root / "maps" / "nuplan-maps-v1.0.json").exists())
print("active manifest exists:", (data_root / "nuplan-v1.1" / "splits" / "mini_active_manifest.txt").exists())
PY
nuplan_cli --help | head -40 || true