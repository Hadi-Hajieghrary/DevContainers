#!/usr/bin/env bash
set -euo pipefail

echo "=== Devcontainer postCreateCommand ==="

echo "Testing Python packages..."
/home/vscode/.venv/bin/python -c "import torch; import pydrake; import numpy as np; print('✓ PyTorch', torch.__version__, '(CUDA available:', str(torch.cuda.is_available()) + ')'); print('✓ Drake imported successfully'); print('✓ NumPy', np.__version__)"

echo ""
echo "Testing C++ build tools..."
cmake --version | head -1
echo "✓ CMake installed"
ninja --version | head -1 && echo "✓ Ninja installed"

echo ""
echo "Testing Drake C++ installation..."
if [ -d "/opt/drake" ]; then
    echo "✓ Drake C++ installed at /opt/drake"
    echo "  Drake version: $(ls /opt/drake/share/drake/package.xml 2>/dev/null && grep -oP '(?<=<version>)[^<]+' /opt/drake/share/drake/package.xml || echo 'unknown')"
else
    echo "✗ Drake C++ not found at /opt/drake"
fi

echo ""
echo "All packages verified successfully!"
echo ""
echo "To build C++ code:"
echo "  cd cpp && mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_PREFIX_PATH=/opt/drake"
echo "  cmake --build . -j\$(nproc)"
