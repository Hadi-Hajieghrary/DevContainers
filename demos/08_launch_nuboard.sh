#!/usr/bin/env bash
set -euo pipefail

# Demo 08: launch nuBoard for the newest available .nuboard file.
# Component demonstrated: nuBoard support.

# Prefer conda env if available; otherwise use current shell Python.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate nuplan || echo "[demo] warning: could not activate conda env 'nuplan'; using current Python."
  fi
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[demo] error: python is not available in PATH."
  exit 2
fi

cd "${NUPLAN_DEVKIT_ROOT:-/workspace/nuplan-devkit}"

EXP_ROOT="${NUPLAN_EXP_ROOT:-/workspace/exp}"
PORT="${NUBOARD_PORT:-5006}"
NUBOARD_FILE="${1:-}"

if [[ -z "${NUBOARD_FILE}" ]]; then
  NUBOARD_FILE="$(find "${EXP_ROOT}" -name '*.nuboard' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi

if [[ -z "${NUBOARD_FILE}" || ! -f "${NUBOARD_FILE}" ]]; then
  echo "No .nuboard file found. Run a simulation first, for example:"
  echo "  demos/06_closed_loop_simulation.sh 3"
  echo "Then rerun this script."
  exit 2
fi

echo "[demo] launching nuBoard file: ${NUBOARD_FILE}"
echo "[demo] open http://localhost:${PORT} in your browser"
python nuplan/planning/script/run_nuboard.py \
  simulation_path="${NUBOARD_FILE}" \
  port_number="${PORT}"
