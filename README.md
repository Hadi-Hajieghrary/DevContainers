# DevContainers Repository

This repository contains production-ready DevContainer configurations optimized for GPU-accelerated development workflows.

## Available Branches

| Branch | Description | Simulation Framework |
|--------|-------------|---------------------|
| `WayMax` | Waymax + SQP-Based Rulebook MPC | Waymax (Waymo) |
| `Mujoco` | Physics simulation with MuJoCo | MuJoCo 3.4.0 |
| `Drake` | Robotics simulation with Drake | Drake (MIT) |

---

## Waymax Branch (WayMax) - SQP-Based Rulebook MPC Development

The `WayMax` branch provides a comprehensive DevContainer for developing **Weighted SQP-Based Rulebook MPC** for autonomous vehicle planning using Waymax.

### Features

**Core Frameworks:**
- **Waymax** - Waymo's lightweight simulator for autonomous driving research
- **JAX** - High-performance numerical computing with GPU acceleration
- **CasADi** - Symbolic framework for automatic differentiation and optimal control
- **CVXPY + OSQP** - Convex optimization and fast QP solvers
- **TensorFlow** - Required for Waymo Open Dataset

**Optimization Stack (for SQP/MPC):**
- CasADi with multiple solver backends
- CVXPY with 15+ solver options
- OSQP, ECOS, Clarabel, ProxSuite QP solvers
- SciPy SLSQP for nonlinear optimization
- do-mpc for Model Predictive Control
- python-control for control systems analysis

**Environment Configuration:**
- **Base:** Ubuntu 22.04 + NVIDIA CUDA 12.4.1
- **GPU Support:** Full NVIDIA GPU passthrough for JAX acceleration
- **Python:** 3.11 with dedicated virtual environment at `/home/vscode/.venv`
- **Shell:** Bash with automatic venv activation

**Development Tools:**
- VS Code extensions: Python, Pylance, Jupyter, GitHub Copilot, LaTeX Workshop
- Black code formatter (88-character line length)
- Jupyter notebooks for interactive development
- SSH key mounting for secure Git operations

### Quick Start

1. Clone this repository
2. Open in VS Code with Dev Containers extension
3. Rebuild container when prompted
4. Wait for automatic package verification

### Testing Installation

```bash
# Comprehensive installation test
python scripts/test_waymax_installation.py
```

### Multi-Agent Simulation Demo (`multi_agent_sim.py`)

```bash
python scripts/multi_agent_sim.py
```

A minimal simulation on a **straight 3-lane highway** demonstrating Waymax's multi-agent environment.
Since no Waymo Open Motion Dataset (WOMD) data is needed, the scenario is built
programmatically with `create_synthetic_scenario()`, which constructs a full
`datatypes.SimulatorState` — trajectories, road graph, object metadata, and
traffic lights — entirely from NumPy/JAX arrays.

**Vehicles (8 objects):**

| Objects | Policy | Behaviour |
|---------|--------|-----------|
| 0, 1 | `IDMRoutePolicy` | Car-following with safe distance in the centre lane |
| 2 | `create_constant_speed_actor` (5 m/s) | Cruises in the right lane |
| 3, 4 | `create_expert_actor` (log replay) | Replays logged trajectory |
| 5, 6, 7 | `create_constant_speed_actor` (0 m/s) | Parked on the road |

Outputs `simulation.mp4` (81 frames @ 10 fps, ~8 s).

---

### City Planning Simulation (`city_planning_sim.py`)

```bash
python scripts/city_planning_sim.py
```

A complex urban driving scenario designed to test advanced planning capabilities.
This simulation replaces the standard grid with a custom road network featuring:
*   **High-Speed Highway:** A multi-lane freeway replacing the western avenue.
*   **Exit Ramp:** A quadratic Bezier curve connecting the highway to the city grid.
*   **Roundabout:** A circular traffic junction handling multi-way intersections.
*   **Explicit Goal:** A visualized target area for the ego vehicle.
*   **Mixed Traffic:** Highway cruisers, merging vehicles, roundabout navigators, and parked cars.

**Scenario Details:**
*   **Ego Vehicle:** Starts on the highway, must take the exit ramp, navigate the roundabout, and park at the goal.
*   **Obstacles:** A slow-moving truck is placed directly in the Ego's path on the highway to force IDM braking behavior.
*   **Control:** Uses `IDMRoutePolicy` for collision avoidance while following a complex pre-calculated trajectory.
*   **Map Generation:** Procedurally generated geometry using `datatypes.RoadgraphPoints`.

Outputs `city_planning_sim.mp4` (~20s duration).

#### Working Without WOMD Data

Both simulations construct `datatypes.SimulatorState` from scratch — no TFRecord
files or Google Cloud access required. The key dataclasses:

```python
datatypes.Trajectory(x, y, z, vel_x, vel_y, yaw, valid,
                     timestamp_micros, length, width, height)
                     # shape: (num_objects, num_timesteps)

datatypes.RoadgraphPoints(x, y, z, dir_x, dir_y, dir_z,
                          types, ids, valid)
                          # shape: (30000,)

datatypes.ObjectMetadata(ids, object_types, is_sdc, is_modeled,
                         is_valid, objects_of_interest, is_controlled)
                         # shape: (num_objects,)

datatypes.SimulatorState(sim_trajectory, log_trajectory,
                         log_traffic_light, object_metadata,
                         timestep, roadgraph_points, sdc_paths)
```

For real-world scenarios, use `dataloader.simulator_state_generator()` with a
`DatasetConfig` pointing to WOMD TFRecord files instead.

---

### City Grid with Ego Path Planning (`scripts/city_planning_sim.py`)

This script demonstrates **ego-vehicle path planning** in a dense urban grid with background traffic. It uses the A* algorithm to find a route through the road network and generates smooth trajectories that respect lane geometry and turning constraints.

**Key Features:**
- **6×6 City Grid**: A larger urban environment with 6 avenues and 6 streets, plus a diagonal "Express Avenue" cutting through the grid.
- **A* Planner**: Computes the optimal sequence of intersections from start `(0,0)` to goal `(5,4)`.
- **Smooth Trajectory Generation**:
  - Converts A* node sequence into lane-centre waypoints.
  - Generates circular arcs for turns at intersections.
  - Applies speed profiling (slow down for turns, cruise on straights).
- **Collision Avoidance**: All vehicles (Ego + Background) use the **IDM (Intelligent Driver Model)** policy (`waymax.agents.IDMRoutePolicy`) to follow their planned paths while reacting to other vehicles (braking to prevent collisions).
- **Goal Visualization**: The goal location is marked with yellow dots (rendered as simulated speed bumps).

**Scenario Details:**

| Feature | Details |
| :--- | :--- |
| **Map** | 6 east-west streets × 6 north-south avenues. Block size = 50m. Lane width = 4m. |
| **Ego Vehicle** | Object 0. Route: (0,0) → (5,4). |
| **Background** | 11 vehicles with diverse fixed routes (straight, turning, snaking, diagonal). |
| **Parked** | 4 vehicles statically positioned on shoulders. |
| **Simulation** | 20 seconds (200 steps @ 10 Hz). |

**Algorithm:**
1. **Graph Abstraction**: The city grid is abstracted into a graph where nodes are intersections and edges are road segments.
2. **Path Search**: A* finds the shortest path of nodes.
3. **Geometry Generation**: Detailed (x,y,yaw) trajectory is interpolated from lane centerlines.
4. **Reactive Control**: During simulation, the `IDMRoutePolicy` controls acceleration to maintain safety distances, effectively preventing collisions even if paths intersect.

#### Running the simulation
```bash
python scripts/city_planning_sim.py
```
Outputs `city_planning_sim.mp4`.

### Project Structure for Rulebook MPC Development

```
.
├── .devcontainer/           # DevContainer configuration
│   ├── Dockerfile           # Container build instructions
│   ├── devcontainer.json    # VS Code configuration
│   ├── requirements.txt     # Python dependencies
│   └── post-create.sh       # Post-creation setup script
├── scripts/
│   ├── test_waymax_installation.py  # Installation verification
│   ├── multi_agent_sim.py           # Multi-agent driving simulation demo
│   ├── complex_road_sim.py          # Complex road network simulation
│   └── city_planning_sim.py         # City grid with A* ego path planning
└── References/              # Research documentation
    ├── ref/                 # Paper LaTeX sources
    ├── sim/                 # Highway traffic simulator
    └── sqp/                 # SQP algorithm notes
```

### Key Dependencies for Rulebook MPC

| Package | Purpose |
|---------|---------|
| `waymax` | Waymo's simulator with real-world driving scenarios |
| `jax` | Hardware-accelerated numerical computing |
| `casadi` | Symbolic MPC formulation and automatic differentiation |
| `cvxpy` | Convex optimization modeling |
| `osqp` | Fast embedded QP solver |
| `python-control` | LQR, pole placement, state-space analysis |
| `do-mpc` | High-level MPC framework |

### Downloading Waymo Open Dataset

To use Waymax with real driving scenarios:

```bash
# Install gsutil
pip install gsutil

# Download sample scenarios (requires Google Cloud account)
gsutil -m cp -r gs://waymo_open_dataset_motion_v_1_2_0/scenario/training.tfrecord-* ./data/
```

See [Waymax documentation](https://waymo-research.github.io/waymax/) for detailed instructions.

---

## Drake Branch - Robotics Simulation & Planning Environment

The `Drake` branch provides a comprehensive DevContainer for GPU-accelerated robotics simulation, planning, and control development.

### Features

**Core Frameworks:**
- **Drake** - MIT's robotics toolkit for simulation, planning, and control
- **PyTorch 2.6.0+** - Deep learning with CUDA 12.4 support
- **Meshcat** - Web-based 3D visualization
- **OpenCV, NumPy, SciPy, Matplotlib** - Scientific computing stack

**Environment Configuration:**
- **Base:** Ubuntu 22.04 + NVIDIA CUDA 12.4.1
- **GPU Support:** Full NVIDIA GPU passthrough with X11 forwarding
- **Python:** Dedicated virtual environment at `/home/vscode/.venv`
- **Shell:** Bash with automatic venv activation

**Development Tools:**
- VS Code extensions: Python, Pylance, Jupyter, GitHub Copilot, GitLens, Autodocstring
- GitHub CLI for repository operations
- Black code formatter (88-character line length)
- SSH key mounting for secure Git operations

### Quick Start

1. Clone this repository
2. Switch to the `Drake` branch: `git checkout Drake`
3. Open in VS Code with Dev Containers extension
4. Rebuild container when prompted
5. Wait for automatic package verification

### Testing Installation

```bash
# Test PyTorch and GPU support
python scripts/test_pytorch_instalation.py

# Test Drake installation
python scripts/test_drake_instalation.py
```

---

## MuJoCo Branch - Physics Simulation & Deep Learning Environment

The `Mujoco` branch provides a comprehensive DevContainer for GPU-accelerated physics simulation and deep learning development.

### Features

**Core Frameworks:**
- **MuJoCo 3.4.0** - Multi-Joint dynamics with Contact physics simulation
- **PyTorch 2.6.0+** - Deep learning with CUDA 12.4 support
- **OpenCV, NumPy, SciPy, Matplotlib** - Scientific computing stack

**Environment Configuration:**
- **Base:** Ubuntu 22.04 + NVIDIA CUDA 12.4.1
- **GPU Support:** Full NVIDIA GPU passthrough with X11 forwarding
- **Python:** Dedicated virtual environment at `/home/vscode/.venv`
- **Shell:** Bash with automatic venv activation

**Development Tools:**
- VS Code extensions: Python, Pylance, Jupyter, GitHub Copilot, GitLens, Autodocstring
- GitHub CLI for repository operations
- Black code formatter (88-character line length)
- SSH key mounting for secure Git operations

### Quick Start

1. Clone this repository
2. Switch to the `Mujoco` branch: `git checkout Mujoco`
3. Open in VS Code with Dev Containers extension
4. Rebuild container when prompted
5. Wait for automatic package verification

### Testing Installation

```bash
# Test PyTorch and GPU support
python scripts/test_pytorch_instalation.py

# Test MuJoCo (headless rendering)
python scripts/test_mujoco_instalation.py --headless

# Test MuJoCo (interactive GUI - requires X11)
python scripts/test_mujoco_instalation.py
```

---

## Common Prerequisites

- Docker with NVIDIA GPU support (nvidia-container-toolkit)
- NVIDIA drivers compatible with CUDA 12.4.1
- X11 server for GUI applications
- VS Code with Dev Containers extension
- NVIDIA GPU with CUDA support
- 4GB+ RAM recommended (8GB+ for Drake)

## Container Configuration

All branches share similar GPU configuration:
- 2GB shared memory allocation
- IPC host mode for optimal performance
- NVIDIA driver capabilities: compute, utility, graphics, display
- Automatic virtual environment activation

### Documentation

For comprehensive setup instructions, troubleshooting, and customization options, see `.devcontainer/README.md` in each branch.

## License

This DevContainer configuration is provided as-is for development purposes. Check individual package licenses for usage restrictions.
