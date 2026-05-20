#!/usr/bin/env bash
set -euo pipefail
source /home/vscode/.venv/bin/activate

export NUPLAN_DATA_ROOT="${NUPLAN_DATA_ROOT:-/workspace/data}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NUPLAN_DATA_ROOT}/maps}"
export NUPLAN_EXP_ROOT="${NUPLAN_EXP_ROOT:-/workspace/exp}"

EXPECTED_MINI_DIR="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/mini"
if [[ ! -d "${EXPECTED_MINI_DIR}" ]]; then
  mkdir -p "${EXPECTED_MINI_DIR}"
fi
if ! compgen -G "${EXPECTED_MINI_DIR}/*.db" >/dev/null; then
  echo "[sim] no mini DBs found in ${EXPECTED_MINI_DIR}" >&2
  echo "[sim] run: scripts/bootstrap_nuplan.sh --profile planner_mini" >&2
  exit 2
fi

cd "${NUPLAN_DEVKIT_ROOT:-/workspace/nuplan-devkit}"

LIMIT_TOTAL_SCENARIOS="${1:-${NUPLAN_SIM_LIMIT_TOTAL_SCENARIOS:-10}}"
EXPERIMENT_NAME="${NUPLAN_EXPERIMENT_NAME:-idm_mini_smoke_test}"

# This uses the official mini scenario builder and limits the number of scenarios for a fast smoke test.
# Increase LIMIT_TOTAL_SCENARIOS only after the first run succeeds.
python nuplan/planning/script/run_simulation.py \
  +simulation=open_loop_boxes \
  planner=idm_planner \
  scenario_builder=nuplan_mini \
  scenario_filter=one_of_each_scenario_type \
  scenario_filter.limit_total_scenarios="${LIMIT_TOTAL_SCENARIOS}" \
  worker=sequential \
  experiment_name="${EXPERIMENT_NAME}" \
  job_name="${EXPERIMENT_NAME}"