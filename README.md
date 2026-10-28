## GNN-IN-CAD

Face-level machining-feature classification on MFCAD, with the UV-Net architecture and
ablations of the information sources it consumes (surface geometry, curve geometry,
B-rep topology).

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

### Repository layout

```
src/uvnet/
  cli.py              # uvnet-train entry point (argparse, model selection)
  audit.py            # dataset audit (shapes, labels, adjacency, masks)
  overfit.py          # single-solid sanity check for every model
  data/
    loading.py        # MFCAD .bin/.json loading and official splits
    graphs.py         # UV normalization, B-rep graph construction, dataset, collate
    features.py       # 15-dim graphlet degree vectors and Laplacian positional encodings
  models/
    encoders.py       # surface UV-grid CNN, curve 1D CNN, topology MLP
    convs.py          # edge-conditioned convolution (UV-Net graph operator)
    face_model.py     # all per-face models behind one API
  training/
    metrics.py        # confusion-matrix metrics (accuracy, macro IoU, macro F1)
    loop.py           # seeded training/evaluation loop and run records
tests/                # unit tests (data, features, models)
```

### Models

Every model predicts one of 16 machining-feature classes per B-rep face:

| `--model` | Surface geometry | Curve geometry | Topology | Notes |
| --- | --- | --- | --- | --- |
| `cnn` | ✔ (10×10×7 UV grid) | – | none | per-face CNN, no message passing |
| `complete_gcn` | ✔ | – | all faces form a complete graph | global context, no B-rep edges |
| `adjacency_gcn` | ✔ | – | B-rep face adjacency | binary edges |
| `uvnet` | ✔ | ✔ (10×6 curve grid) | B-rep face adjacency | curve CNN + edge-conditioned convolution |
| `uvnet_lape` | ✔ | ✔ | B-rep adjacency + 8 Laplacian eigenvectors | UV-Net plus positional encoding |
| `adjacency_gcn_lape` | ✔ | – | B-rep adjacency + 8 Laplacian eigenvectors | positional-encoding ablation |
| `topo_gdv` | – | – | 15-dim graphlet degree vectors | topology-only, no geometry |

Shared pieces: the surface encoder (three conv blocks, global average pooling, linear
projection), the classifier head, the optimizer and the data pipeline are identical
wherever they apply, so differences come from the information each model receives.

### Data representation

* **Faces**: `(10, 10, 7)` UV grids — xyz, normal xyz, binary trimming mask.
* **Curves**: `(10, 6)` per directed edge — 10 samples along the trim curve, each with
  xyz and normal xyz.
* **Normalization**: each solid is centered on the bounding box of its valid surface
  samples and scaled to longest extent 2; the same transform is applied to curve points.
  No dataset statistics are used, so nothing leaks from train to validation/test.
* **Graph**: directed trim-curve edges are kept as they are (each direction carries its
  own curve), exact duplicate edges are dropped, and each face gets a self-loop with zero
  curve features so the edge-conditioned convolution can learn a self term.

### Usage

```sh
conda activate gnn_project
pip install -e .
python -m unittest discover -s tests -v

# Dataset audit and single-solid overfit check.
uvnet-audit --output results/data_audit.json
uvnet-overfit --output results/overfit.json

# One model on the official splits (test evaluated once, after checkpoint selection).
uvnet-train --model uvnet --epochs 20 --batch-size 16 --threads 4 --seed 42 \
  --test --output results/uvnet-seed42

# All six models, sequentially, same seeded data order.
uvnet-train --model all --epochs 20 --batch-size 16 --threads 4 --seed 42 \
  --test --output results/all-seed42

# Quick pipeline check on subsets (not a benchmark).
uvnet-train --model uvnet --epochs 2 --train-limit 256 --val-limit 64 \
  --test --test-limit 64 --output results/pilot-uvnet
```

`python -m uvnet.cli`, `python -m uvnet.audit` and `python -m uvnet.overfit` are
equivalent to the console scripts. `--model all` runs the models sequentially with the
same seeded data order. Output directories must be new, to prevent overwriting
experiments. Each run directory contains `config.json` (including split IDs, package
versions and source hashes), `history.jsonl` (per-epoch metrics), `best.pt` (lowest
validation loss) and `result.json`; `summary.json` collects the completed models.

Metrics are computed from a dataset-level confusion matrix: face accuracy, macro IoU over
classes with non-empty union, per-class IoU, macro F1 and balanced accuracy. Test data is
evaluated only after the checkpoint has been selected on validation loss. Data and
training artifacts are ignored by Git.

### Reports

* `results/REPORT.md` — baselines: CNN vs complete GCN vs B-rep adjacency GCN, with
  per-class analysis and paired tests.
* `results/REPORT_EXPERIMENTS.md` — seven-model comparison including full UV-Net, the
  topology-only GDV model and the LapPE ablation. Regenerate it with
  `python analysis/make_report.py`.
