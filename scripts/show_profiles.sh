#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_DIR="$(cd "${SCRIPT_DIR}/../configs/data_profiles" && pwd)"

echo "Available nuPlan data profiles:"
for f in "${PROFILE_DIR}"/*.env; do
  name="$(basename "$f" .env)"
  first_comment="$(grep -m 1 '^#' "$f" | sed 's/^# *//' || true)"
  printf '  %-20s %s\n' "$name" "$first_comment"
done

echo
cat <<'HELP'
Preview a profile without downloading:
  scripts/bootstrap_nuplan.sh --profile planner_mini --print-plan

Use a profile:
  scripts/bootstrap_nuplan.sh --profile maps_only
  scripts/bootstrap_nuplan.sh --profile planner_mini
  scripts/bootstrap_nuplan.sh --profile mini_with_sensors

Override the active mini-DB count:
  scripts/bootstrap_nuplan.sh --profile planner_mini --max-active-mini-dbs 3
HELP