#!/bin/bash
# Post-create script for Waymax-SQP-MPC DevContainer

set -e

echo "=============================================="
echo "Waymax-SQP-MPC DevContainer Post-Create Setup"
echo "=============================================="

# Create virtual environment if it doesn't exist
if [ ! -d "/home/vscode/.venv" ]; then
    echo "Creating virtual environment..."
    python -m venv /home/vscode/.venv
fi

# Activate virtual environment
source /home/vscode/.venv/bin/activate

# Set LD_LIBRARY_PATH for pip-installed nvidia CUDA libraries (needed by JAX GPU)
NVIDIA_SITE_PKGS="/home/vscode/.venv/lib/python3.11/site-packages/nvidia"
if [ -d "$NVIDIA_SITE_PKGS" ]; then
    for lib_dir in cublas/lib cuda_cupti/lib cuda_nvrtc/lib cuda_runtime/lib cudnn/lib cufft/lib cusolver/lib cusparse/lib nccl/lib nvjitlink/lib nvshmem/lib; do
        [ -d "$NVIDIA_SITE_PKGS/$lib_dir" ] && export LD_LIBRARY_PATH="$NVIDIA_SITE_PKGS/$lib_dir:${LD_LIBRARY_PATH:-}"
    done
fi

# Ensure pip is up to date
pip install --upgrade pip

# Fix SSH key permissions if mounted
if [ -d "/home/vscode/.ssh" ]; then
    echo "Fixing SSH key permissions..."
    sudo chown -R vscode:vscode /home/vscode/.ssh 2>/dev/null || true
    chmod 700 /home/vscode/.ssh 2>/dev/null || true
    chmod 600 /home/vscode/.ssh/* 2>/dev/null || true
fi

# Configure git to use SSH (may fail if .gitconfig is bind-mounted read-only)
git config --global url."git@github.com:".insteadOf "https://github.com/" 2>/dev/null || true

# Create useful directories
mkdir -p /home/vscode/data
mkdir -p /home/vscode/experiments
mkdir -p /home/vscode/logs

# Print verification info
echo ""
echo "=============================================="
echo "Installation Verification"
echo "=============================================="

echo ""
echo "Python version:"
python --version

echo ""
echo "JAX version and devices:"
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(f'Devices: {jax.devices()}')" 2>/dev/null || echo "JAX not available"

echo ""
echo "Waymax version:"
python -c "import waymax; print(f'Waymax version: {waymax.__version__}')" 2>/dev/null || echo "Waymax not available"

echo ""
echo "CasADi version:"
python -c "import casadi; print(f'CasADi version: {casadi.__version__}')" 2>/dev/null || echo "CasADi not available"

echo ""
echo "CVXPY version:"
python -c "import cvxpy; print(f'CVXPY version: {cvxpy.__version__}')" 2>/dev/null || echo "CVXPY not available"

echo ""
echo "OSQP version:"
python -c "import osqp; print(f'OSQP version: {osqp.__version__}')" 2>/dev/null || echo "OSQP not available"

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "No GPU detected"

echo ""
echo "=============================================="
echo "DevContainer setup complete!"
echo "Run 'python scripts/test_waymax_installation.py' to verify full installation."
echo "=============================================="
