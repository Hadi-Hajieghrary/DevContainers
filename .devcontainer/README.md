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

JAX memory settings (adjustable in `devcontainer.json`):
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`: Dynamic memory allocation
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`: Use 75% of GPU memory max

## Installed Packages

### Core Simulation
| Package | Version | Purpose |
|---------|---------|---------|
| `waymax` | latest | Waymo's AV planning simulator |
| `jax[cuda12]` | latest | GPU-accelerated computing |
| `tensorflow` | 2.15.0 | Waymo Open Dataset support |

### Optimization (SQP/MPC)
| Package | Purpose |
|---------|---------|
| `casadi` | Symbolic optimal control, auto-diff |
| `cvxpy` | Convex optimization modeling |
| `osqp` | Fast QP solver (embedded-friendly) |
| `clarabel` | Interior-point conic solver |
| `proxsuite` | Proximal QP solver |
| `qpsolvers` | Unified QP interface |

### Control
| Package | Purpose |
|---------|---------|
| `python-control` | LQR, state-space, transfer functions |
| `do-mpc` | High-level MPC framework |

## Development Workflow

### 1. Formulating MPC Problems with CasADi

```python
import casadi as ca

# Symbolic variables
x = ca.SX.sym('x', 4)  # State
u = ca.SX.sym('u', 2)  # Control

# Dynamics (kinematic bicycle)
x_next = f(x, u)

# Cost function
cost = ca.mtimes([x.T, Q, x]) + ca.mtimes([u.T, R, u])

# NLP formulation
nlp = {'x': ca.vertcat(x, u), 'f': cost, 'g': constraints}
solver = ca.nlpsol('mpc', 'ipopt', nlp)
```

### 2. Using Waymax for Scenario Simulation

```python
import waymax
from waymax import config as waymax_config

# Load scenario from Waymo Open Dataset
config = waymax_config.WaymaxConfig(...)

# Run simulation with your MPC controller
for step in range(horizon):
    action = mpc_controller.solve(state)
    state = waymax.step(state, action)
```

### 3. Solving QPs with OSQP

```python
import osqp
import numpy as np
from scipy import sparse

# QP: min 0.5 x'Px + q'x s.t. l <= Ax <= u
P = sparse.csc_matrix(...)
q = np.array(...)
A = sparse.csc_matrix(...)

solver = osqp.OSQP()
solver.setup(P, q, A, l, u, verbose=False)
result = solver.solve()
```

## Troubleshooting

### JAX not detecting GPU

```bash
# Check CUDA visibility
nvidia-smi

# Verify JAX devices
python -c "import jax; print(jax.devices())"
```

If only CPU is detected:
1. Ensure `nvidia-container-toolkit` is installed on host
2. Verify Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`
3. Rebuild container

### Import errors for Waymax

Waymax requires TensorFlow for dataset loading:
```bash
pip install waymo-open-dataset-tf-2-12-0
```

### OSQP solver failures

For numerically challenging QPs:
```python
solver.setup(P, q, A, l, u,
    eps_abs=1e-5,
    eps_rel=1e-5,
    max_iter=10000,
    polish=True
)
```

### Memory issues

Reduce JAX memory fraction:
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

## Customization

### Adding packages

Edit `requirements.txt` and rebuild:
```bash
pip install -r .devcontainer/requirements.txt
```

### Changing CUDA version

Modify the base image in `Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
```

### VS Code extensions

Add to `devcontainer.json`:
```json
"customizations": {
    "vscode": {
        "extensions": [
            "your.extension-id"
        ]
    }
}
```

## References

- [Waymax Documentation](https://waymo-research.github.io/waymax/)
- [CasADi Documentation](https://web.casadi.org/docs/)
- [OSQP Documentation](https://osqp.org/docs/)
- [JAX Documentation](https://jax.readthedocs.io/)
