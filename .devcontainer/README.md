# Waymax + SQP-Based Rulebook MPC DevContainer

This DevContainer provides a complete development environment for implementing **Weighted SQP-Based Rulebook MPC** for autonomous vehicle planning.

## Overview

The container includes:
- **Waymax**: Waymo's lightweight simulator for behavior planning research
- **JAX + CUDA**: GPU-accelerated numerical computing
- **Optimization Stack**: CasADi, CVXPY, OSQP, and more for SQP/MPC
- **Control Libraries**: python-control, do-mpc

## Prerequisites

- Docker with NVIDIA GPU support (`nvidia-container-toolkit`)
- NVIDIA drivers compatible with CUDA 12.4.1
- VS Code with Dev Containers extension
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- 16GB+ RAM recommended

## Quick Start

1. **Open in VS Code**: Open the repository folder in VS Code
2. **Reopen in Container**: Press `F1` → "Dev Containers: Reopen in Container"
3. **Wait for Build**: First build takes 10-15 minutes (downloading CUDA, JAX, etc.)
4. **Verify Installation**: Run `python scripts/test_waymax_installation.py`

## Container Structure

```
.devcontainer/
├── Dockerfile           # Multi-stage build with CUDA + Python + packages
├── devcontainer.json    # VS Code settings, extensions, mounts
├── requirements.txt     # Python packages for optimization/control
├── post-create.sh       # Post-build verification script
└── README.md            # This file
```

## GPU Configuration

The container is configured for full GPU access:

```json
"runArgs": [
    "--gpus", "all",
    "--ipc=host",
    "--shm-size=4g"
]
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `waymax` | Waymo's simulator with real-world driving scenarios |
| `jax` | Hardware-accelerated numerical computing |
| `casadi` | Symbolic MPC formulation and automatic differentiation |
| `cvxpy` | Convex optimization modeling |
| `osqp` | Fast embedded QP solver |
| `python-control` | LQR, pole placement, state-space analysis |
| `do-mpc` | High-level MPC framework |

## Development Workflow

1. **Simulation**: Use Waymax for scenario-based testing
2. **Modeling**: Formulate MPC problems with CasADi/CVXPY
3. **Optimization**: Solve with OSQP or other QP solvers
4. **Validation**: Test controllers in simulated environments

## Testing

Run the installation test:
```bash
python scripts/test_waymax_installation.py
```

Run the MPC example:
```bash
python scripts/example_sqp_mpc.py
```

## Troubleshooting

- **GPU Issues**: Ensure NVIDIA drivers and container toolkit are installed
- **Memory**: Increase Docker memory limit if builds fail
- **Permissions**: Check SSH key mounting for Git operations

## License

This DevContainer configuration is provided as-is for development purposes.
