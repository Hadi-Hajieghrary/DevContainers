#!/usr/bin/env bash
set -euo pipefail

# Run only the DB/map demos.  This avoids long simulation jobs and does not
# require raw camera/LiDAR sensor archives.

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

cd /workspace/nuplan-project

python demos/01_ego_trajectories.py --limit "${DEMO_LIMIT:-300}"
python demos/02_agent_tracks.py --frames "${DEMO_FRAMES:-20}"
python demos/03_hd_map_query.py
python demos/04_traffic_light_states.py --limit "${DEMO_LIMIT:-300}" || true
python demos/05_scenario_extraction.py --limit "${DEMO_LIMIT:-300}"
python demos/10_route_mission_metadata.py --limit 20 || true
python demos/09_end_to_end_planning_inputs.py --frame-index "${DEMO_FRAME_INDEX:-20}"

echo "[demo] lightweight demos complete. Outputs: demos/outputs"
