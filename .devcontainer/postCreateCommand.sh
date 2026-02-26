#!/usr/bin/env bash
set -euo pipefail

echo "=== Devcontainer postCreateCommand ==="

echo "Testing Python packages..."
/home/vscode/.venv/bin/python -c "
import torch; import pydrake; import numpy as np
import pandas, matplotlib, scipy, PIL
print('✓ PyTorch', torch.__version__, '(CUDA available:', str(torch.cuda.is_available()) + ')')
print('✓ Drake imported successfully')
print('✓ NumPy', np.__version__)
print('✓ pandas', pandas.__version__)
print('✓ matplotlib', matplotlib.__version__)
print('✓ scipy', scipy.__version__)
print('✓ Pillow', PIL.__version__)
"

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
echo "Testing video tools..."
ffmpeg -version 2>/dev/null | head -1 && echo "✓ ffmpeg installed" || echo "✗ ffmpeg not found"
ffprobe -version 2>/dev/null | head -1 && echo "✓ ffprobe installed" || echo "✗ ffprobe not found"

echo ""
echo "Testing LaTeX tools..."
pdflatex --version 2>/dev/null | head -1 && echo "✓ pdflatex installed" || echo "✗ pdflatex not found"
biber --version 2>/dev/null | head -1 && echo "✓ biber installed" || echo "✗ biber not found"
latexmk --version 2>/dev/null | head -1 && echo "✓ latexmk installed" || echo "✗ latexmk not found"

echo ""
echo "All packages verified successfully!"
echo ""
echo "To build C++ code:"
echo "  cd Research/cpp && mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_PREFIX_PATH=/opt/drake"
echo "  cmake --build . -j\$(nproc)"
echo ""
echo "To compile the paper:"
echo "  cd IEEE_IROS_2026 && latexmk -pdf Main.tex"
