# nuPlan DevContainer Repository

This repository contains a configurable VS Code devcontainer for nuPlan planning
research. It is set up for planner-centric work by default: maps plus mini SQLite
log DBs, no raw camera/LiDAR downloads, Jupyter, and nuBoard support.

## What you get

- A devcontainer based on the official nuPlan devkit.
- Host-mounted dataset and experiment directories at `./data` and `./exp`.
- Dataset profiles for maps only, planner-only mini data, mini data with sensors,
  no-download setups, and custom overrides.
- Helper scripts for bootstrapping the dataset, running a small IDM simulation,
  launching nuBoard, starting JupyterLab, and checking the installation.

## Quick start

1. Open the repository in VS Code.
2. Reopen in container when prompted.
3. Wait for the post-create bootstrap to finish.
4. Verify the install with `scripts/check_install.sh`.

## Default workflow

The default profile is `planner_mini` and downloads:

- HD semantic maps;
- mini SQLite log DBs.

It does not download raw camera or LiDAR blobs.

Useful commands:

```bash
scripts/show_profiles.sh
scripts/bootstrap_nuplan.sh --profile planner_mini
scripts/run_simulation_idm_mini.sh 3
scripts/run_nuboard.sh /workspace/exp/path/to/file.nuboard
scripts/start_jupyter.sh
```

## Dataset layout

The bootstrap downloads dataset files into the host `./data` folder and the
container mounts that same folder at `/workspace/data` through
`.devcontainer/docker-compose.yaml`:

- `./data` -> `/workspace/data`
- `./exp` -> `/workspace/exp`

The NuPlan devkit itself is cloned into `/workspace/nuplan-devkit` during the
image build.

## Changing profiles

The available profiles live in `configs/data_profiles/`:

- `planner_mini`
- `planner_mini_small`
- `maps_only`
- `mini_with_sensors`
- `no_download`
- `custom`

Edit `NUPLAN_DATA_PROFILE` in `.devcontainer/docker-compose.yaml` if you want a
different default.

## More details

- Devcontainer internals: `.devcontainer/README.md`
- Planner-input demos: `demos/README.md`
- Data profile quick help: `scripts/show_profiles.sh`

## License

See `LICENSE`.
