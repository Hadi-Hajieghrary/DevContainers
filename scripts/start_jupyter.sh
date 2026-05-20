#!/usr/bin/env bash
set -euo pipefail
source /home/vscode/.venv/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''