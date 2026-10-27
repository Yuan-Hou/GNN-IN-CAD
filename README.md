# GNN-IN-CAD

Face-level machining-feature classification on MFCAD, with the UV-Net architecture and
ablations of the information sources it consumes (surface geometry, curve geometry,
B-rep topology).

Two parallel implementations share the MFCAD dataset and environment:

- [Yong Jun workflow](#yong-jun-workflow): notebook-based model ladder using `src/uvnet_project/`.
- [YuanSen workflow](#yuansen-workflow): CLI experiments using `src/uvnet/`.

Model settings, graph preprocessing, checkpoints and report formats are described
within their respective workflows.

## Shared setup

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

## Yong Jun workflow

Run the notebooks in `notebooks/`, backed by reusable modules in
`src/uvnet_project/`. Start with `baseline.ipynb` to train the face CNN and
independent classifier, then run the shallow-topology and message-passing
notebooks below. Training starts when their training cells are executed.

### VM setup

For teammates setting up a Linux GPU VM, see [VM_SETUP.md](VM_SETUP.md).
Run `bash scripts/setup_vm.sh --download-data`, then
`bash scripts/start_jupyter.sh`. These scripts do not run training.

### Model notebooks and encoders

`uvnet_project.edge_cnn.EdgeCNN` provides the UV-Net curve encoder:
6 input channels → Conv1d 64 → 128 → 256 → average pooling → 64D embedding.
Its architecture follows the [released UV-Net encoder](https://github.com/AutodeskAILab/UV-Net/blob/main/uvnet/encoders.py).

`notebooks/laplacian.ipynb` implements Stage 1.5 with `training_mode="frozen"`
(cached baseline CNN embeddings, classifier only) or `training_mode="joint"`
(fine-tune CNN + classifier on face grids). Both concatenate per-solid
normalized-Laplacian eigenvectors, defaulting to 8 columns, and use random
signs per solid/eigenvector during training. No message passing is used. It requires
`results/baseline_run_01/epoch_020/checkpoint.pt`; adjust that notebook path
if needed. Features are cached under `data/feature_cache`, and results go
to `results/laplacian_run_01` (frozen) or `results/laplacian_joint_run_01`
(joint). The final comparison chart includes baseline, frozen and joint
histories whenever available.

`notebooks/graphlet.ipynb` adds per-face graphlet degree vectors, following
Shallow Learning Lab 01's node orbits and normalization and Lab 02's
bit-mask lookup. Set `max_graphlet_nodes=4` (15 features) or `5` (73), and
`training_mode="frozen"` or `"joint"`. Counts are exact on simple undirected
graphs; standardization is fitted on training faces only. Results are
separated as `results/graphlet4_frozen_run_01`, `graphlet4_joint_run_01`,
`graphlet5_frozen_run_01`, and `graphlet5_joint_run_01`. Comparisons include
the geometry-only baseline and variants within the same theme. No graph-level
SVM kernel is used for this face-classification experiment.

`notebooks/wl.ipynb` adapts Shallow Learning Lab 03's 1-WL refinement to
per-face colour histories. Set `wl_iterations=3` (rounds 0–3 by default)
and `training_mode="frozen"` or `"joint"`. Colours start uniformly, without
target labels; a training-only vocabulary maps unseen validation colours
to zero-contribution unknowns. Sparse categorical projection is equivalent
to concatenated one-hot WL features in the classifier's first linear layer.
Results are `results/wl3_frozen_run_01` / `wl3_joint_run_01` for default
settings. Comparisons include the baseline and WL runs only. Both modes
support the same per-epoch seed/RNG/optimizer checkpointing.

`notebooks/chebyshev.ipynb` follows Graph Convnets Lab 01's sparse Chebyshev
recurrence for per-face classification. Choose `cheb_hops=2` or `3` and
`training_mode="frozen"` or `"joint"`. One graph convolution at width 64,
ReLU and a pointwise two-layer MLP keep the total graph reach at 2 or 3 hops.
Polynomial degree is the hop count; the tutorial's K is the number of terms
(3 or 4). Simple undirected normalized Laplacians are scaled with the bound
lambda_max=2 and batched as sparse block diagonal matrices. No coarsening,
pooling, eigenvectors or edge geometry are used. Results go to
`results/cheb2_frozen_run_01` etc.; comparisons include baseline + Chebyshev
variants only. Both modes retain epoch checkpoints, exact epoch-boundary
RNG resume, batch loss logging and final-only prediction export.

`notebooks/anisotropic.ipynb` uses Graph Convnets Lab 04/05's residual
GatedGCN pattern, with exactly two graph layers and no global pooling.
Run `edge_mode="static"` (curve features gate face messages), then
`"updated"` (endpoint-conditioned gates and recurrent edge updates).
The UV-Net EdgeCNN trains in both variants; static describes graph-layer
state, not encoder training. `training_mode="joint"` fine-tunes FaceCNN
by default; `"frozen"` caches baseline face embeddings while still training
EdgeCNN. Original DGL edge IDs and curve-grid alignment are preserved,
including parallel B-rep edges. Results are `results/anisotropic_static_joint_run_01`
etc. Comparisons include baseline + anisotropic only. Epoch checkpoints,
RNG resume, loss/accuracy logging and final prediction export are retained.

### Load MFCAD in a notebook

The editable project installation makes the loader available to notebooks without modifying `sys.path`:

```python
from uvnet_project.load_mfcad import load_mfcad

samples = load_mfcad("train", head=5)
```

Omit `head` to load the entire selected split.

```python
train, test = load_mfcad("train"), load_mfcad("test")
```

### Classifier and evaluation

The classifier maps face embeddings to raw logits with two linear layers
and a ReLU between them. Both dimensions are explicit inputs:

```python
from uvnet_project.classifier import Classifier
from uvnet_project.evaluation import evaluate

classifier = Classifier(embedding_dim=64, hidden_dim=128, num_classes=16)
# evaluation_loader yields (embedding_batch, label_batch).
results = evaluate(classifier, evaluation_loader, "results/validation.pt")
print(results["loss"], results["accuracy"])
```

The dimensions above are examples. This follows the paper's two-layer MLP
description; the released UV-Net implementation uses a three-layer head with
batch normalization and dropout instead. See
[UV-Net Section 4.2](https://openaccess.thecvf.com/content/CVPR2021/papers/Jayaraman_UV-Net_Learning_From_Boundary_Representations_CVPR_2021_paper.pdf)
and [the released classifier](https://github.com/AutodeskAILab/UV-Net/blob/main/uvnet/models.py).

For a complete encoder-plus-classifier model, pass that model and batches of
its inputs instead. Keep the model on the evaluation device; the evaluator
moves inputs and labels to that device. It disables gradients and restores
the model's previous training mode afterward.

The saved dictionary contains mean cross-entropy loss per face, accuracy
(0 to 1), a confusion matrix (true labels in rows, predictions in columns),
and all softmax probabilities, labels, and predictions as CPU tensors.
Rows follow loader order; use a non-shuffled loader to preserve face order.
Load the dictionary with `torch.load("results/validation.pt", weights_only=True)`.

## YuanSen workflow

Use the `uvnet` package and its command-line entry points for the model
comparisons below. This workflow defines its own graph preprocessing,
checkpoint selection and report generation.

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
