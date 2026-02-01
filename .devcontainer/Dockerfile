# Ubuntu 22.04 + CUDA dev image (works well with RTX 4060 Ti)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# -----------------------------
# Base OS packages + GUI deps
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    curl \
    ca-certificates \
    build-essential \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    openssh-client \
    # X11 + OpenGL + GLFW runtime deps for MuJoCo viewer
    libgl1 \
    libgl1-mesa-dri \
    libglvnd0 \
    libegl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    libglfw3 \
    mesa-utils \
    x11-apps \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install GitHub CLI
# -----------------------------
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
 && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
 && apt-get update \
 && apt-get install -y gh \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Create non-root user (Standard DevContainer practice)
# -----------------------------
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd -s /bin/bash --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}

# -----------------------------
# Python: Setup Virtual Env & Deep Learning Stack
# -----------------------------
USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.venv/bin:$PATH"

RUN python3 -m venv /home/${USERNAME}/.venv \
 && pip install --no-cache-dir --upgrade pip setuptools wheel \
 # Install PyTorch 2.4+ specifically for CUDA 12.4
 && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
 # Install MuJoCo and scientific stack
 && pip install --no-cache-dir \
      mujoco==3.4.0 \
      glfw \
      numpy \
      opencv-python \
      scipy \
      matplotlib \
 # Auto-activate virtual environment in new shells
 && echo 'source /home/vscode/.venv/bin/activate' >> /home/${USERNAME}/.bashrc

WORKDIR /workspace