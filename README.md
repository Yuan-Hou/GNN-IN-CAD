## GNN-IN-CAD

### Environment

Ensure the `gnn_course` environment is created as described in the `CS5284_2026` repository. Then clone it and install this project's extra packages; `gnn_course` stays unchanged.

```sh
conda create --name gnn_project --clone gnn_course
conda activate gnn_project
pip install -r requirements.txt
pip install -e .
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
