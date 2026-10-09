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
