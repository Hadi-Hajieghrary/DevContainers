# DevContainers Repository

This repository contains production-ready DevContainer configurations optimized for GPU-accelerated development workflows.

## Available Branches

| Branch | Description | Simulation Framework |
|--------|-------------|---------------------|
| `Mujoco` | Physics simulation with MuJoCo | MuJoCo 3.4.0 |
| `Drake` | Robotics simulation with Drake | Drake (MIT) |

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
