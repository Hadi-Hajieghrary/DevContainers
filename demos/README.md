# nuPlan Planner-Input Demos

This folder contains small, inspectable demos for the planner-level components you asked for:

| Component | Demo | Requires raw camera/LiDAR blobs? | Output |
|---|---|---:|---|
| Ego trajectories | `01_ego_trajectories.py` | No | CSV + XY plot |
| Agent tracks | `02_agent_tracks.py` | No | CSV + XY scatter |
| HD maps | `03_hd_map_query.py` | No | Map file inventory; optional API check |
| Traffic-light states | `04_traffic_light_states.py` | No | CSV + status counts |
| Scenario extraction | `05_scenario_extraction.py` + `scripts/05_official_scenario_extraction_smoke.sh` | No | Candidate scenario CSV; optional official scenario-filter run |
| Closed-loop / open-loop simulation | `06_closed_loop_simulation.sh` | No | nuPlan experiment artifacts |
| Metrics engine | `07_metrics_engine_summary.py` | No | JSON summary of metrics/simulation artifacts |
| nuBoard support | `08_launch_nuboard.sh` | No | Browser dashboard |
| Combined planner input | `09_end_to_end_planning_inputs.py` | No | JSON planner-input snapshot |
| Route / mission metadata | `10_route_mission_metadata.py` | No | JSON route/mission metadata inspection |
| Recorded scenario playback | `11_recorded_scenario_mp4.py` | No | MP4 video with full infrastructure overlay |
| Random recorded scenario playback | `12_random_recorded_scenario_mp4.py` | No | MP4 video from a randomly selected mini log |
| Configurable IDM simulation | `13_idm_simulation_and_record.py` | No | Run IDM planner on scenarios with tunable parameters |

The important design choice is that these examples operate on **planner-level nuPlan data**: SQLite DB logs plus HD maps. They intentionally avoid raw sensor blobs unless you explicitly choose the `mini_with_sensors` profile.

---

## 1. Prepare the planner-mini data

From the devcontainer terminal:

```bash
scripts/bootstrap_nuplan.sh --profile planner_mini
```

This downloads:

- mini SQLite DB logs;
- HD semantic maps.

It does **not** download:

- camera images;
- LiDAR point-cloud blobs.

The DB logs are enough for ego pose/state, tracked agents, traffic-light state, route/mission metadata, and scenario extraction.

---

## 2. Run all lightweight demos

```bash
bash demos/scripts/run_all_lightweight_demos.sh
```

This runs the DB/map demos and writes outputs under:

```text
demos/outputs/
```

This command does not launch nuBoard and does not run a long simulation job.

---

## 3. Run each demo individually

### 3.1 Ego trajectories

```bash
python demos/01_ego_trajectories.py --limit 300
```

What it does:

- finds a mini `.db` log;
- reads `lidar_pc -> ego_pose` records;
- exports ego trajectory samples;
- plots the ego path.

Output:

```text
demos/outputs/01_ego_trajectory.csv
demos/outputs/01_ego_trajectory_xy.png
```

Use this data as the minimal ego-state stream for controllers, MPC, SQP, W-SQP, and rulebook evaluation.

---

### 3.2 Agent tracks

```bash
python demos/02_agent_tracks.py --frames 20
```

What it does:

- reads tracked object boxes from `lidar_box` rows;
- groups them over a small number of `lidar_pc` frames;
- plots a quick XY scatter of surrounding objects.

Output:

```text
demos/outputs/02_agent_tracks.csv
demos/outputs/02_agent_tracks_xy.png
```

This is the right abstraction for planning. You do not need raw LiDAR to start collision checking or obstacle-aware trajectory optimization.

---

### 3.3 HD maps

```bash
python demos/03_hd_map_query.py
```

Optional map API check:

```bash
python demos/03_hd_map_query.py --try-api
```

What it does:

- verifies that the map archive is mounted;
- prints the installed map files;
- optionally tries to instantiate the nuPlan map API.

Output:

```text
demos/outputs/03_hd_map_files.txt
```

---

### 3.4 Traffic-light states

```bash
python demos/04_traffic_light_states.py --limit 300
```

What it does:

- reads the traffic-light status table if present in the selected DB;
- joins with timestamps when possible;
- prints status counts.

Output:

```text
demos/outputs/04_traffic_light_states.csv
```

Some logs may have few or no traffic-light rows. If this happens, pass another DB explicitly:

```bash
python demos/04_traffic_light_states.py --db /workspace/data/nuplan-v1.1/splits/mini/YOUR_LOG.db
```

---

### 3.5 Scenario extraction

Lightweight transparent extraction:

```bash
python demos/05_scenario_extraction.py --limit 500
```

What it does:

- mines simple candidate windows from low-level DB attributes;
- labels frames such as `ego_stop_or_creep`, `dense_agent_context`, and `traffic_light_context`.

Output:

```text
demos/outputs/05_scenario_extraction_candidates.csv
```

Official nuPlan scenario-filter smoke test:

```bash
bash demos/scripts/05_official_scenario_extraction_smoke.sh 3
```

This uses the official `run_simulation.py` Hydra path with:

```text
scenario_builder=nuplan_mini
scenario_filter=one_of_each_scenario_type
```

That exercises the official scenario-builder/scenario-filter machinery.

---

### 3.6 Closed-loop / open-loop simulation

Fast smoke test:

```bash
bash demos/06_closed_loop_simulation.sh 3
```

By default this uses:

```text
simulation=closed_loop_nonreactive_agents
planner=idm_planner
scenario_builder=nuplan_mini
```

If your installed devkit version does not expose that Hydra config, use the conservative open-loop smoke test:

```bash
NUPLAN_SIMULATION_CONFIG=open_loop_boxes bash demos/06_closed_loop_simulation.sh 3
```

Depending on your installed nuPlan devkit version, valid simulation config names may differ.

---

### 3.7 Metrics engine

After running a simulation:

```bash
python demos/07_metrics_engine_summary.py
```

What it does:

- searches `NUPLAN_EXP_ROOT` for metric, simulation, and nuBoard artifacts;
- writes a compact JSON inventory.

Output:

```text
demos/outputs/07_metrics_engine_artifacts.json
```

---

### 3.8 nuBoard

After running a simulation:

```bash
bash demos/08_launch_nuboard.sh
```

Or pass a specific `.nuboard` file:

```bash
bash demos/08_launch_nuboard.sh /workspace/exp/path/to/file.nuboard
```

Open:

```text
http://localhost:5006
```

VS Code devcontainer forwards port `5006` by default.

---

### 3.9 Combined planner input snapshot

```bash
python demos/09_end_to_end_planning_inputs.py --frame-index 20
```

What it does:

- reads one ego state;
- reads agents at the same `lidar_pc` frame;
- reads traffic-light rows for the same frame if available;
- records map-root availability;
- writes a compact JSON snapshot.

Output:

```text
demos/outputs/09_planner_input_snapshot.json
```

This is the best starting point for plugging in your own Rulebook / W-SQP / MPC planner.

---

### 3.10 Route / mission metadata

```bash
python demos/10_route_mission_metadata.py --limit 20
```

What it does:

- inspects DB tables and columns that contain route/mission/goal/roadblock keywords;
- writes a JSON diagnostic.

Output:

```text
demos/outputs/10_route_mission_metadata.json
```

In the official nuPlan simulation framework, route information is usually consumed through scenario objects rather than hand-parsing the DB directly. This demo helps you inspect what exists in your installed DB version.

---

### 3.11 Recorded scenario playback

```bash
python demos/11_recorded_scenario_mp4.py --fps 10
```

Optional longer playback with custom frame count:

```bash
python demos/11_recorded_scenario_mp4.py --frames 480 --fps 15
```

What it does:

- reads ego pose and agent box history from the DB;
- queries the HD map GeoPackage for all nearby infrastructure (roads, lanes, intersections, crosswalks, traffic lights, baseline paths);
- renders an MP4 video with:
  - **Ego trajectory tail** (blue line showing recent path)
      - **Ego footprint + heading** (oriented red vehicle box and heading vector)
      - **Agent oriented boxes** (length/width/yaw from tracked boxes)
      - **Agent class coloring** (vehicle/pedestrian/bicycle/bus/generic where available)
      - **Agent motion vectors + short per-track trails** (velocity and short history)
  - **Traffic-light highlights** (lane segments colored green/yellow/red by signal status)
  - **Route goal marker** (golden star at destination)
  - **Lane centerlines** (blue dashed lines showing baseline paths)
      - **Static map features** (drivable area, roads, lanes, lane groups, boundaries, connectors, intersections, crosswalks, walkways, carpark areas)
      - **Telemetry HUD** (time, frame index, ego speed/acceleration/yaw, agent class counts, traffic-light counts, scene name, roadblock ids)
- uses the shared utility `demos/common/recorded_scenario_renderer.py`, so other demos/scripts can reuse the same renderer.

Output:

```text
demos/outputs/11_recorded_scenario.mp4
```

Default parameters:

- **Frames**: all available frames in the selected log (use `--frames N` to cap for faster previews)
- **FPS**: 10
- **Tail length**: 30 frames (shows last 3 seconds of ego trajectory)
- **Map margin**: 40 meters (spatial query buffer around ego)

This demo is useful for:

- Verifying that recorded scenarios match your expectations
- Inspecting agent interactions in rich visual context
- Exporting scenario playback for reports or presentations
- Validating that your planner's predicted trajectory fits the recorded infrastructure

Supported map locations:

- `sg-one-north` (Singapore)
- `us-ma-boston` (Boston, MA)
- `us-nv-las-vegas-strip` (Las Vegas, NV)
- `us-pa-pittsburgh-hazelwood` (Pittsburgh, PA)

If the selected DB log uses a different map location, the demo will raise a clear error.

### 3.12 Random recorded scenario playback

```bash
python demos/12_random_recorded_scenario_mp4.py
```

Reproducible random selection with a fixed seed:

```bash
python demos/12_random_recorded_scenario_mp4.py --seed 7 --frames 360 --fps 12
```

What it does:

- scans available nuPlan DB logs under your data root;
- filters to logs that contain planner-level required tables (`lidar_pc`, `ego_pose`, `lidar_box`);
- picks one DB at random (or deterministic with `--seed`);
- calls the shared renderer utility with full map/agent/traffic-light overlays.

Output:

```text
demos/outputs/12_random_<db_name>/12_random_recorded_scenario.mp4
```

### 3.13 Configurable IDM simulation

```bash
python demos/13_idm_simulation_and_record.py --scenarios 3
```

Customize IDM planner behavior:

```bash
python demos/13_idm_simulation_and_record.py \
  --scenarios 3 \
  --target-velocity 10.0 \
  --accel-max 2.5 \
  --decel-max 2.5 \
  --headway-time 1.2 \
  --min-gap 1.5 \
  --seed 7
```

What it does:

- runs closed-loop simulation on a configurable number of random scenarios;
- uses the IDM (Intelligent Driver Model) planner to control ego;
- supports tunable IDM parameters:
  - **target_velocity**: desired speed in free traffic (m/s)
  - **accel_max**: maximum acceleration (m/s²)
  - **decel_max**: maximum deceleration (m/s²)
  - **headway_time**: time gap to lead vehicle (s)
  - **min_gap**: minimum distance to lead vehicle (m)
- writes simulation artifacts to `NUPLAN_EXP_ROOT`.

Output:

Simulation logs and metrics under `/workspace/exp/demo_13_idm_*/<timestamp>/`.
MP4 recording under `demos/outputs/demo_13_idm_*.mp4`.
The selected log is random by default; pass `--seed` to make it reproducible.

View results:

```bash
python demos/07_metrics_engine_summary.py
bash demos/08_launch_nuboard.sh
```

This demo is useful for:

- establishing baseline vehicle behavior with tunable parameters;
- validating simulation infrastructure;
- benchmarking planner performance against IDM baseline;
- understanding scenario difficulty through IDM success/failure modes.

---

## 4. Recommended workflow for your Rulebook / W-SQP planner

Use this pipeline:

```text
01 ego trajectory
      +
02 agent tracks
      +
03 HD map
      +
04 traffic lights
      +
10 route / mission metadata
      ↓
09 planner-input snapshot
      ↓
your Rulebook / W-SQP / MPC planner
      ↓
06 nuPlan simulation
      ↓
07 metrics summary
      ↓
08 nuBoard
      +
11 recorded scenario playback
     (validation visualization)
```

Do not start by downloading raw sensors. First make the planner work with the structured state abstraction.

---

## 5. Debugging tips

### No DB files found

Run:

```bash
scripts/bootstrap_nuplan.sh --profile planner_mini
find /workspace/data -name '*.db' | head
```

### Map root missing

Run:

```bash
scripts/bootstrap_nuplan.sh --profile maps_only
find /workspace/data/maps -maxdepth 3 -type f | head
```

### Traffic-light demo finds no rows

That can happen for a particular log. Try another DB:

```bash
find /workspace/data/nuplan-v1.1/splits/mini -name '*.db'
python demos/04_traffic_light_states.py --db /path/to/another.db
```

### nuBoard cannot find a `.nuboard` file

Run a simulation first:

```bash
bash demos/06_closed_loop_simulation.sh 3
find /workspace/exp -name '*.nuboard'
```

---

## 6. Why no raw sensors are required here

For planning research, the normal abstraction is not raw pixels or point clouds. It is:

```text
ego state + tracked agents + map + traffic lights + route/context
```

The demos are built around exactly that abstraction. Raw sensors are only needed if your research question includes perception, sensor fusion, detection, or end-to-end sensor-to-control learning.
