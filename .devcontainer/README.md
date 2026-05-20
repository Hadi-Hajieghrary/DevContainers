# nuPlan DevContainer

This folder contains the VS Code devcontainer configuration for the repository.
The container is planner-first: it installs the nuPlan devkit, mounts host-side
`data/` and `exp/` directories to matching container paths (`/workspace/data`
and `/workspace/exp`), and bootstraps the default `planner_mini` profile on
first create. The container runtime wiring now lives in
`docker-compose.yaml` in this same folder.

Dataset downloads land in the host `data/` folder and appear in the container
at `/workspace/data`.

See the root [README.md](../README.md) for the
full workflow, dataset profiles, and helper scripts.
