FROM python:3.9-bullseye

ARG DEBIAN_FRONTEND=noninteractive
ARG NUPLAN_DEVKIT_REPO=https://github.com/motional/nuplan-devkit.git
ARG NUPLAN_DEVKIT_REF=master

ENV SHELL=/bin/bash \
    VIRTUAL_ENV=/home/vscode/.venv \
    PATH=/home/vscode/.venv/bin:$PATH \
    NUPLAN_DEVKIT_ROOT=/workspace/nuplan-devkit \
    NUPLAN_DATA_ROOT=/workspace/data \
    NUPLAN_MAPS_ROOT=/workspace/data/maps \
    NUPLAN_EXP_ROOT=/workspace/exp \
    MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash build-essential ca-certificates curl git git-lfs wget unzip rsync \
    ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 libgeos-dev libspatialindex-dev \
    libsqlite3-dev sqlite3 graphviz vim nano htop tmux jq sudo \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash vscode && echo "vscode ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/vscode

RUN python -m venv ${VIRTUAL_ENV} \
    && ${VIRTUAL_ENV}/bin/python -m pip install --upgrade 'pip<24.1' setuptools wheel

RUN mkdir -p /workspace ${NUPLAN_DATA_ROOT} ${NUPLAN_EXP_ROOT} /workspace/nuplan-project \
    && chown -R vscode:vscode /workspace ${VIRTUAL_ENV}

USER vscode
WORKDIR /workspace

RUN git clone ${NUPLAN_DEVKIT_REPO} ${NUPLAN_DEVKIT_ROOT} \
    && cd ${NUPLAN_DEVKIT_ROOT} \
    && git checkout ${NUPLAN_DEVKIT_REF}

RUN bash -lc "source ${VIRTUAL_ENV}/bin/activate && \
    cd ${NUPLAN_DEVKIT_ROOT} && \
    python -m pip install --upgrade 'pip<24.1' setuptools wheel && \
    if [ -f requirements_torch.txt ]; then \
        grep -Ev '^(--find-links|torch==|torchvision==|torch_scatter==)' requirements_torch.txt > /tmp/requirements_torch.filtered.txt && \
        pip install \
            'https://download.pytorch.org/whl/cu111/torch-1.9.0%2Bcu111-cp39-cp39-linux_x86_64.whl' \
            'https://download.pytorch.org/whl/cu111/torchvision-0.10.0%2Bcu111-cp39-cp39-linux_x86_64.whl' \
            'https://data.pyg.org/whl/torch-1.9.0%2Bcu111/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl' && \
        pip install -r /tmp/requirements_torch.filtered.txt; \
    fi && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi && \
    pip install -e . && \
    pip install jupyterlab ipykernel notebook ipywidgets pandas matplotlib scipy scikit-learn tqdm rich ruff black pytest && \
    python -m ipykernel install --user --name nuplan --display-name 'Python (nuPlan)'"

RUN echo 'source /home/vscode/.venv/bin/activate' >> /home/vscode/.bashrc \
    && echo 'export NUPLAN_DATA_ROOT=/workspace/data' >> /home/vscode/.bashrc \
    && echo 'export NUPLAN_MAPS_ROOT=/workspace/data/maps' >> /home/vscode/.bashrc \
    && echo 'export NUPLAN_EXP_ROOT=/workspace/exp' >> /home/vscode/.bashrc \
    && echo 'export NUPLAN_DEVKIT_ROOT=/workspace/nuplan-devkit' >> /home/vscode/.bashrc \
    && echo 'alias cdnuplan="cd /workspace/nuplan-devkit"' >> /home/vscode/.bashrc \
    && echo 'alias cdproj="cd /workspace/nuplan-project"' >> /home/vscode/.bashrc

WORKDIR /workspace/nuplan-project
