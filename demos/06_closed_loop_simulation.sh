#!/usr/bin/env bash
set -euo pipefail

# Demo 06: closed-loop/open-loop simulation smoke test through official nuPlan script.
# Component demonstrated: Closed-loop simulation support.
#
# Default mode is closed_loop_nonreactive_agents to exercise closed-loop
# execution without requiring learned reactive agents. If your installed devkit
# lacks this Hydra config, set NUPLAN_SIMULATION_CONFIG=open_loop_boxes for a
# conservative smoke test.

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

LIMIT_TOTAL_SCENARIOS="${1:-${NUPLAN_SIM_LIMIT_TOTAL_SCENARIOS:-3}}"
SIMULATION_CONFIG="${NUPLAN_SIMULATION_CONFIG:-closed_loop_nonreactive_agents}"
PLANNER="${NUPLAN_PLANNER:-idm_planner}"
SCENARIO_FILTER="${NUPLAN_SCENARIO_FILTER:-one_of_each_scenario_type}"
EXPERIMENT_NAME="${NUPLAN_EXPERIMENT_NAME:-demo_06_${SIMULATION_CONFIG}_${PLANNER}}"

printf '[demo] simulation config: %s\n' "${SIMULATION_CONFIG}"
printf '[demo] planner: %s\n' "${PLANNER}"
printf '[demo] scenarios: %s\n' "${LIMIT_TOTAL_SCENARIOS}"
printf '[demo] experiment: %s\n' "${EXPERIMENT_NAME}"

python nuplan/planning/script/run_simulation.py \
  +simulation="${SIMULATION_CONFIG}" \
  planner="${PLANNER}" \
  scenario_builder=nuplan_mini \
  scenario_filter="${SCENARIO_FILTER}" \
  scenario_filter.limit_total_scenarios="${LIMIT_TOTAL_SCENARIOS}" \
  worker=sequential \
  experiment_name="${EXPERIMENT_NAME}" \
  job_name="${EXPERIMENT_NAME}"

printf '\n[demo] component demonstrated: Closed-loop/simulation framework = YES\n'
printf '[demo] generated files under: %s\n' "${NUPLAN_EXP_ROOT:-/workspace/exp}"
find "${NUPLAN_EXP_ROOT:-/workspace/exp}" -maxdepth 8 -type f -path "*${EXPERIMENT_NAME}*" | tail -50 || true
