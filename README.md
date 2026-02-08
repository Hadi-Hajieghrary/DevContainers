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

# Run example SQP-MPC demo
python scripts/example_sqp_mpc.py
```

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
│   └── example_sqp_mpc.py           # Example MPC implementation
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
