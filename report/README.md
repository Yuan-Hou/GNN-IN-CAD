# Report setup

With conda installed, create a fresh environment. Run these commands from
this report folder:

```bash
conda create -n gnn_report python=3.11 pip
conda activate gnn_report
python -m pip install -r requirement.txt
python -m ipykernel install --sys-prefix --name gnn_report --display-name "Python (gnn_report)"
jupyter lab
```

Open `main.ipynb`, select the Python kernel from the `gnn_report` environment,
and follow the links through the report sections.

## Dataset extraction

The submission includes `data/MFCADDataset.zip`. Run the extraction cell in
`sections/01_data_preparation.ipynb` to unpack it within this folder.

The nested `.7z` archives require 7-Zip, with either `7zz` or `7z` available
on your PATH. Run all notebooks and commands with the report folder as the
working directory, including notebooks stored in `sections/` and `working/`.
The extraction script resolves data paths relative to its own location.
You can also run it from the report folder with `python -c "from scripts.data import unpack_data; unpack_data()"`.

## Folder structure

- `main.ipynb`: introduction and contents.
- `sections/`: report section notebooks.
- `working/`: working notebooks.
- `scripts/`: reusable helpers. Model helpers are grouped into `model_components.py`
  (CNN and classifier), `model_evaluation.py` (evaluation and cached results),
  and `model_utils.py` (architecture diagrams).
- `data/`: raw data and cached computed results.

Further setup instructions will be added as the report develops.

## Model runs

`working/model_baseline.ipynb` contains the baseline setup, training loop, cache
handling, and result display. It loads the submitted baseline results by default.
No training or inference runs, and no GPU is required to inspect the cached
predictions and loss history. The bundled cache is copied from our existing
20-epoch baseline run; file checksums detect missing or changed artifacts.

For optional reproduction, set `retrain = True` in that notebook on a machine
with a CUDA GPU. This starts a fresh 20-epoch run and overwrites the cached baseline
in `data/model_runs/baseline/`. A missing or invalid cache in default mode raises
an error and never starts training automatically.

`data/model_runs/` is tracked by Git; large checkpoints and arrays use Git LFS.
The bundled dataset ZIP and t-SNE coordinate archive also use LFS. Small cache
metadata and the shared validation index use regular Git. Submission packaging
includes the actual local files, not LFS pointers.
Numeric class IDs are preserved; displayed names use the reviewed mapping.

## Clean and package the submission

From the report folder:

```bash
python scripts/package_submission.py --clean
```

The ZIP is written outside this folder to `../dist/report_submission.zip`.
Use `--output /path/to/submission.zip` to choose another destination.

The packager includes the raw dataset ZIP, notebooks, scripts, figures, and completed model runs, even when Git ignores them. For each completed
model run it includes logs, summaries, and final validation probabilities.
Non-WL runs retain their final checkpoint, including weights. WL runs instead
retain validation geometry, compact colour-history IDs, and first-classifier-layer
activations after ReLU; these replace the large WL weight matrices.
Validation labels, solid IDs, and face offsets are shared in `data/validation_index/`. Training performance is retained as compact confusion
counts and loss, without per-face training predictions. Training saves weights once, inside the final checkpoint, without resume state;
incomplete runs are excluded. Re-run the
result-display cell after training to save the summary tables.

Extracted raw data, Python/editor caches, temporary files, local editor settings,
and standalone HTML inspection exports are excluded. Cleanup removes only
Python/editor caches and `.DS_Store` files; it preserves datasets and model runs.
The ZIP contains a file manifest with SHA-256 checksums and is integrity-checked.

Shallow-topology working notebooks are `working/model_laplacian.ipynb`,
`working/model_wl.ipynb`, and `working/model_graphlet.ipynb`. Each defaults to
loading cached results; optional retraining requires `retrain=True` and a CUDA
GPU. The bundled shallow-topology models use frozen geometry: eight Laplacian
features, WL with three or four rounds, and graphlets with up to four or five nodes.
The main comparisons retain three-round WL and four-node graphlets.

`working/model_chebyshev.ipynb` supports cached two-/three-hop ChebNet runs
with frozen or joint geometry, with the same explicit retraining switch.

EDA statistics are computed directly from the bundled dataset on each notebook
run. EDA does not read or write persistent caches.

`working/model_edge.ipynb` provides cached static/updated edge-gated results.
Both variants use joint geometry training; optional fresh training follows the
same `retrain=True` switch.

Non-WL final checkpoints retain weights, training history, configuration, and
model metadata. WL history and configuration accompany the cached embeddings.
Optimizer, AMP scaler, and RNG resume state are removed.


`working/analysis_tsne.ipynb` contains the sampling, reconstruction, PCA, t-SNE
and cache-writing code, and loads the three aligned embedding views by default.
`scripts/analysis_tsne.py` only provides cache I/O, integrity checks and plotting.
Coordinates in `data/tsne/` are included in the submission; the report only loads
and plots them. Set `recompute=True` in the working notebook to reconstruct the
selected embeddings from the report artifacts and rerun PCA/t-SNE explicitly.
This requires the bundled dataset to be unpacked and scikit-learn installed;
CPU is supported, and no retraining is performed. Missing or stale caches raise
an error rather than starting computation automatically.


Class labels in analysis tables, plot axes, heatmaps and legends follow the EDA
order in `data/class_order.json`. Reuse `CLASS_ORDER` or `ordered_class_indices`
from `scripts/data.py`; avoid numeric-ID, alphabetical or subset-frequency order.
