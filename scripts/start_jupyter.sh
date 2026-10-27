#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gnn_project
export DGLBACKEND=pytorch
exec jupyter lab --no-browser --ip=127.0.0.1 --port=8888 \
    --ServerApp.port_retries=0 --ServerApp.root_dir="$PROJECT_DIR"
