#!/usr/bin/env bash
set -euo pipefail

# Official scenario-extraction smoke test.
# It asks the nuPlan scenario builder to select one_of_each_scenario_type from
# the mini split, then runs a tiny official simulation.  This exercises the same
# scenario-builder/scenario-filter machinery used in nuPlan evaluations.

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

export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/workspace/data}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NUPLAN_DATA_ROOT}/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/workspace/exp}"

EXPECTED_MINI_DIR="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini"
if [[ ! -d "${EXPECTED_MINI_DIR}" ]]; then
  mkdir -p "${EXPECTED_MINI_DIR}"
fi
if ! compgen -G "${EXPECTED_MINI_DIR}/*.db" >/dev/null; then
  echo "[demo] no mini DBs found in ${EXPECTED_MINI_DIR}" >&2
  echo "[demo] run: scripts/bootstrap_nuplan.sh --profile planner_mini" >&2
  exit 2
fi

cd "${NUPLAN_DEVKIT_ROOT:-/workspace/nuplan-devkit}"

LIMIT_TOTAL_SCENARIOS="${1:-3}"
EXPERIMENT_NAME="demo_05_official_scenario_extraction"

python nuplan/planning/script/run_simulation.py \
  +simulation=open_loop_boxes \
  planner=idm_planner \
  scenario_builder=nuplan_mini \
  scenario_filter=one_of_each_scenario_type \
  scenario_filter.limit_total_scenarios="${LIMIT_TOTAL_SCENARIOS}" \
  worker=sequential \
  experiment_name="${EXPERIMENT_NAME}" \
  job_name="${EXPERIMENT_NAME}"

echo "[demo] component demonstrated: official scenario extraction/filtering = YES"
