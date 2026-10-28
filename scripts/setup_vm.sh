#!/usr/bin/env bash
# Run from any directory: bash /path/to/project/scripts/setup_vm.sh [--download-data]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_DATA=false
if [[ $# -gt 1 ]] || [[ $# -eq 1 && "$1" != "--download-data" ]]; then
    echo "Usage: bash scripts/setup_vm.sh [--download-data]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then DOWNLOAD_DATA=true; fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This setup targets Linux x86_64 NVIDIA VMs." >&2
    exit 1
fi
command -v conda >/dev/null || { echo "Install Miniconda and initialize conda first." >&2; exit 1; }
if "$DOWNLOAD_DATA"; then
    for cmd in curl unzip; do
        command -v "$cmd" >/dev/null || { echo "Missing $cmd; see VM_SETUP.md." >&2; exit 1; }
    done
    if ! command -v 7z >/dev/null && ! command -v 7zz >/dev/null; then
        echo "Install p7zip-full; see VM_SETUP.md." >&2
        exit 1
    fi
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if conda run -n gnn_project python --version >/dev/null 2>&1; then
    echo "Reusing gnn_project (existing dependencies are not changed)."
elif conda run -n gnn_course python --version >/dev/null 2>&1; then
    conda create -n gnn_project --clone gnn_course -y
    conda run -n gnn_project python -m pip install -r "$PROJECT_DIR/requirements.txt"
else
    conda env create -f "$PROJECT_DIR/environment-vm.yml"
fi
conda activate gnn_project
python -m pip install --no-deps --no-build-isolation -e "$PROJECT_DIR"
python -m ipykernel install --user --name gnn_project --display-name gnn_project
export DGLBACKEND=pytorch

DATA_DIR="$PROJECT_DIR/data/MFCADDataset"
if "$DOWNLOAD_DATA"; then
    if [[ -d "$DATA_DIR/graph" && -d "$DATA_DIR/labels" && -f "$DATA_DIR/split.json" ]]; then
        echo "Dataset already unpacked; skipping download."
    else
        bash "$PROJECT_DIR/data/download_mfcad.sh"
    fi
fi

# Imports, GPU operation and one-sample inference only; no training.
python "$PROJECT_DIR/scripts/check_vm.py"
echo "Ready. Launch Jupyter: bash $PROJECT_DIR/scripts/start_jupyter.sh"
