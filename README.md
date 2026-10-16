## GNN-IN-CAD

### Environment

Ensure the `gnn_course` environment is created as described in the `CS5284_2026` repository. Then clone it and install this project's extra packages; `gnn_course` stays unchanged.

```sh
conda create --name gnn_project --clone gnn_course
conda activate gnn_project
pip install -r requirements.txt
pip install -e .
```

### STEP visualization environment

OCCWL requires a separate Python 3.10 environment and is intentionally kept
apart from `gnn_project`. Create the reproducible environment and register its
notebook kernel from the project root:

```sh
conda env create --file environment-occwl.yml
conda run --name occwl_viewer python -m ipykernel install --user --name occwl_viewer
```

Open `notebooks/visualize.ipynb` and select **occwl_viewer** as its
kernel. If the new kernel is not listed immediately, restart Cursor or refresh
the kernel picker.

To update an existing environment from the tracked specification:

```sh
conda env update --name occwl_viewer --file environment-occwl.yml --prune
```

### Download MFCAD

From the project root, download and unpack MFCAD into `data/MFCADDataset`:

```sh
bash data/download_mfcad.sh
```

The script requires `7z` or `7zz`. On macOS, install it with `brew install p7zip` if needed.

### Load MFCAD in a notebook

The editable project installation makes the loader available to notebooks without modifying `sys.path`:

```python
from load_mfcad import load_mfcad

samples = load_mfcad("train", head=5)
```

Omit `head` to load the entire selected split.

```python
train, test = load_mfcad("train"), load_mfcad("test")
```

### Train the three UV-grid baselines

The models predict one of 16 machining-feature classes for each B-rep face:

| Model flag | Information passed between faces |
| --- | --- |
| `cnn` | None; independent surface CNN and per-face classifier |
| `complete_gcn` | All faces of the same solid, with residual graph convolutions |
| `adjacency_gcn` | Binary, bidirectional face adjacency, with residual graph convolutions |

All models share the same surface encoder and classifier architecture. The two
GCNs have identical parameter counts. Complete propagation uses DGL mean readout
and broadcast, mathematically equivalent to GraphConv on a complete graph with
self-loops. No information passes between different solids in a batch.
GroupNorm and LayerNorm keep normalization independent of other faces and solids.

The local baseline implementation has been checked with Python 3.11, PyTorch
2.1.2, DGL 1.0.2, NumPy 1.24.3, and SciPy 1.10.1. Use CPU on Apple Silicon;
the trainer also accepts `--device cuda` for a CUDA-enabled DGL installation.

```sh
conda activate gnn_project
pip install -e .
python -m unittest discover -s tests -v
python scripts/audit_mfcad.py --output results/data_audit.json
python scripts/overfit_baselines.py --output results/overfit.json

# Quick pipeline check; these subsets are not a final benchmark.
train-baselines --model all --epochs 5 --train-limit 256 --val-limit 64 \
  --output results/pilot-256-seed42

# Complete official splits, 20 epochs per model, test best validation-loss checkpoint.
train-baselines --model all --epochs 20 --batch-size 16 --threads 4 --seed 42 \
  --test --output results/full-seed42
```

Alternatively run `python scripts/train_baselines.py` with the same arguments.
Output directories must be new, to prevent overwriting experiments. `--model all`
runs the models sequentially with the same seeded data order. Each model directory
contains `config.json` (including split IDs, versions and source hashes),
`history.jsonl`, `best.pt`, and `result.json`; `summary.json` collects completed
models. Metrics include face accuracy, dataset-level macro IoU, class IoUs, and
the confusion matrix. Classes with zero union are excluded from macro IoU and
reported as `null`. Test data is evaluated only after selecting the checkpoint.
Data and training artifacts are ignored by Git.
