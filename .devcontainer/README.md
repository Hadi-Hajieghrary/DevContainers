# MuJoCo GPU GUI DevContainer

This DevContainer provides a complete development environment for working with MuJoCo (Multi-Joint dynamics with Contact) physics simulation library and PyTorch deep learning framework, optimized for GPU-accelerated GUI rendering on Ubuntu 22.04.

## Overview

The container (`mujoco-gpu-gui-ubuntu22`) is based on NVIDIA CUDA 12.4.1 development image and includes:
- MuJoCo 3.4.0 with GLFW backend for interactive visualization
- PyTorch 2.4+ with CUDA 12.4 support for GPU-accelerated deep learning
- OpenCV, NumPy, SciPy, and Matplotlib for scientific computing and visualization
- GPU acceleration support with proper X11 forwarding
- Git and GitHub CLI for version control and repository management
- SSH client for secure Git operations
- VS Code extensions for Python development, GitHub integration, and code formatting

## Prerequisites

- Docker with NVIDIA GPU support
- NVIDIA drivers compatible with CUDA 12.4.1
- X11 server (for GUI display)
- VS Code with Dev Containers extension

### Hardware Requirements
- NVIDIA GPU with CUDA support
- At least 4GB RAM recommended
- Sufficient disk space for container image (~5GB)

## Features

### GPU and GUI Support
- Full NVIDIA GPU passthrough with `--gpus=all`
- X11 forwarding for interactive MuJoCo viewer
- Shared memory size set to 2GB for optimal performance
- Environment variables configured for GLFW backend

### Development Environment
- Python 3.10 virtual environment at `/home/vscode/.venv`
- Pre-installed packages: mujoco, glfw, numpy, opencv-python, scipy, matplotlib, torch, torchvision, torchaudio
- Git and GitHub CLI for version control and repository operations
- VS Code Python extension with Pylance for intelligent code completion
- GitHub Copilot and Pull Request extensions for enhanced development workflow
- GitLens for advanced Git capabilities
- Autodocstring for automatic Python docstring generation
- Jupyter notebook support
- Black code formatting with 88-character line length
- Editor rulers at 88 and 100 characters

### Container Configuration
- Non-root user (`vscode`) with sudo privileges
- Persistent workspace at `/workspaces`
- Automatic package verification during build

## Setup and Usage

1. **Clone the repository** (if not already done)
2. **Open in VS Code** and ensure Dev Containers extension is installed
3. **Rebuild the container** when prompted or via Command Palette: `Dev Containers: Rebuild Container`
4. **Wait for post-create setup** to complete (verifies package installation)

### First Time Setup
The container will automatically:
- Create a Python virtual environment
- Install required packages
- Verify package imports

## Configuration Details

### Dockerfile
- **Base Image**: `nvidia/cuda:12.4.1-devel-ubuntu22.04`
- **User**: `vscode` (UID/GID: 1000)
- **Virtual Environment**: `/home/vscode/.venv`
- **Installed Packages**:
  - mujoco==3.4.0
  - glfw
  - numpy
  - opencv-python
  - scipy
  - matplotlib
  - torch (2.4+ with CUDA 12.4)
  - torchvision
  - torchaudio

### devcontainer.json
- **Container Args**:
  - `--gpus=all`: GPU passthrough
  - `--ipc=host`: Shared IPC namespace
  - `--shm-size=2g`: 2GB shared memory
  - X11 forwarding volumes and environment variables
- **Environment Variables**:
  - `MUJOCO_GL=glfw`: GLFW backend for MuJoCo
  - `PYTHONUNBUFFERED=1`: Unbuffered Python output
  - `PIP_DISABLE_PIP_VERSION_CHECK=1`: Disable pip version checks
  - `__GLX_VENDOR_LIBRARY_NAME=nvidia`: NVIDIA GLX vendor for proper GPU rendering
- **VS Code Extensions**:
  - ms-python.python
  - ms-toolsai.jupyter
  - ms-python.vscode-pylance
  - GitHub.copilot
  - GitHub.vscode-pull-request-github
  - GitHub.copilot-chat
  - eamodio.gitlens
  - njpwerner.autodocstring
- **Python Interpreter**: `/home/vscode/.venv/bin/python`

### Post-Create Command
Runs `postCreateCommand.sh` which verifies that all required Python packages (torch, mujoco, glfw, numpy) can be imported successfully and checks CUDA availability.

## File Structure
```
.devcontainer/
├── devcontainer.json    # DevContainer configuration
├── Dockerfile          # Container build instructions
├── postCreateCommand.sh # Post-build setup script
└── README.md          # This documentation
```

## Testing the Setup

After the container builds successfully, you can test the installations:

```bash
# Test PyTorch GPU support
python scripts/test_pytorch_instalation.py

# Test MuJoCo with headless rendering
python scripts/test_mujoco_instalation.py --headless

# Or for interactive GUI (requires X11 forwarding)
python scripts/test_mujoco_instalation.py
```

The test scripts verify GPU acceleration, package imports, and rendering capabilities.

## Troubleshooting

### GUI Not Displaying
- Ensure X11 server is running on host
- Check that `DISPLAY` environment variable is set correctly
- Verify NVIDIA drivers are installed and working

### GPU Not Detected
- Run `nvidia-smi` inside container to verify GPU access
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi`

### Import Errors
- Ensure virtual environment is activated or use full path: `/home/vscode/.venv/bin/python`
- Check that all packages installed: `pip list`

### Build Failures
- Clear Docker cache: `docker system prune -a`
- Rebuild without cache: `Dev Containers: Rebuild Container` with cache clear option

### Performance Issues
- Increase shared memory: modify `--shm-size` in devcontainer.json
- Check GPU memory usage with `nvidia-smi`

## Customization

### Adding Packages
To add more Python packages, modify the Dockerfile:
```dockerfile
RUN /home/vscode/.venv/bin/pip install \
      mujoco==3.4.0 \
      glfw \
      numpy \
      opencv-python \
      scipy \
      matplotlib \
      torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
      your-package-here
```

### Changing CUDA Version
Update the base image in Dockerfile:
```dockerfile
FROM nvidia/cuda:X.Y.Z-devel-ubuntu22.04
```

### Modifying Extensions
Add or remove VS Code extensions in devcontainer.json:
```json
"extensions": [
  "ms-python.python",
  "ms-toolsai.jupyter",
  "ms-python.vscode-pylance",
  "your-extension-here"
]
```

## Contributing

When modifying this DevContainer:
1. Update this README with any configuration changes
2. Test the build process thoroughly
3. Verify GPU and GUI functionality
4. Update version numbers and dependencies as needed

## License

This DevContainer configuration is provided as-is for development purposes. Check individual package licenses for usage restrictions.</content>
<parameter name="filePath">/workspaces/DevContainers/.devcontainer/README.md