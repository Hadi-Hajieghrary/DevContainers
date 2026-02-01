#!/usr/bin/env bash
set -euo pipefail

echo "=== Devcontainer postCreateCommand ==="
echo "Testing Python packages..."
/home/vscode/.venv/bin/python -c "import torch; import pydrake; import numpy as np; print('✓ PyTorch', torch.__version__, '(CUDA available:', str(torch.cuda.is_available()) + ')'); print('✓ Drake imported successfully'); print('✓ NumPy', np.__version__)"
echo "All packages verified successfully!"
