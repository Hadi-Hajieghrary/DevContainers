# Waymax + SQP-Based Rulebook MPC Development Container
# Base: NVIDIA CUDA 12.4.1 + Ubuntu 22.04
# Python: 3.11 with JAX GPU support

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build tools
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    # Git and development tools
    git \
    git-lfs \
    curl \
    wget \
    unzip \
    # Scientific computing dependencies
    libopenblas-dev \
    liblapack-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    # Visualization dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    # Media processing
    ffmpeg \
    # X11 for GUI applications
    x11-apps \
    xauth \
    # Utilities
    htop \
    vim \
    nano \
    tmux \
    tree \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create non-root user for development
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER $USERNAME
WORKDIR /home/$USERNAME

# Create virtual environment
ENV VIRTUAL_ENV=/home/$USERNAME/.venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install Waymax first — it pins compatible JAX, jaxlib, and waymo-open-dataset versions
# Waymax is not on PyPI; install from GitHub
RUN pip install git+https://github.com/waymo-research/waymax.git

# Install JAX with CUDA 12 support — brings in jax-cuda12-plugin, jax-cuda12-pjrt,
# and nvidia-* packages (cublas, cudnn, cufft, cusolver, cusparse, nccl, nvjitlink)
RUN pip install "jax[cuda12]==0.7.1"

# Install TensorFlow (required for Waymo Open Dataset)
RUN pip install tensorflow==2.15.0

# Install Waymo Open Dataset (compatible workaround for TF 2.15 / Py 3.11)
RUN pip install pyarrow==23.0.0 \
    && pip install --no-deps waymo-open-dataset-tf-2-12-0

# Copy and install requirements for SQP/MPC development
COPY --chown=$USERNAME:$USERNAME requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Set up JAX to use GPU by default
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
# Include pip-installed nvidia libraries so JAX can find cuSPARSE, cuBLAS, cuDNN, etc.
ENV NVIDIA_SITE_PKGS=$VIRTUAL_ENV/lib/python3.11/site-packages/nvidia
ENV LD_LIBRARY_PATH=$NVIDIA_SITE_PKGS/cublas/lib:$NVIDIA_SITE_PKGS/cuda_cupti/lib:$NVIDIA_SITE_PKGS/cuda_nvrtc/lib:$NVIDIA_SITE_PKGS/cuda_runtime/lib:$NVIDIA_SITE_PKGS/cudnn/lib:$NVIDIA_SITE_PKGS/cufft/lib:$NVIDIA_SITE_PKGS/cusolver/lib:$NVIDIA_SITE_PKGS/cusparse/lib:$NVIDIA_SITE_PKGS/nccl/lib:$NVIDIA_SITE_PKGS/nvjitlink/lib:$NVIDIA_SITE_PKGS/nvshmem/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Configure matplotlib backend
ENV MPLBACKEND=Agg

# Set working directory
WORKDIR /workspaces

# Activate venv in bashrc
RUN echo "source $VIRTUAL_ENV/bin/activate" >> ~/.bashrc

# Default command
CMD ["/bin/bash"]
