# Run the notebooks on your own VM

Use a Linux x86_64 VM with an NVIDIA GPU and working driver (`nvidia-smi`).
Install Miniconda first. The setup creates `gnn_project`; it does not change
system Python or install GPU drivers. It never starts notebook training.

Upload and unpack the project package in your home folder, or clone this
repository. Run the following on the VM from its project directory:

```bash
sudo apt-get update
sudo apt-get install -y curl unzip p7zip-full tmux
bash scripts/setup_vm.sh --download-data
```

To rebuild the shareable package locally, run `python scripts/package_vm.py`.
Share `dist/project-vm.zip`. It includes the notebooks (with outputs cleared),
modules, setup scripts and project plan; datasets download during setup.

The script reuses an existing `gnn_project`, otherwise clones `gnn_course`
if available, otherwise creates the pinned environment in `environment-vm.yml`.
It installs the local Python package and registers the notebook kernel.
The checks print package versions and test CUDA and both encoders. Existing
environments must already have compatible dependencies; a failed check is
reported rather than silently replacing their stack.

The NVIDIA environment pins Python 3.11, PyTorch 2.1.2 with CUDA 12.1,
CPU DGL 1.0.2, and NumPy 1.24.3, matching the course's core versions.
CPU DGL reads graph files; the face/edge CNNs execute on the GPU.
Future GPU message passing may need a matching GPU DGL installation.

`--download-data` downloads the published MFCAD archive and extracts it to
`data/MFCADDataset`. It skips this if the extracted folders already exist.
Allow several GB of free disk space for extraction and checkpoints.
Without that option, setup checks synthetic grids if data is absent.
Dataset downloads require network access to the published UV-Net S3 URL.

## Start JupyterLab

On the VM:

```bash
tmux new -s jupyter
bash scripts/start_jupyter.sh
```

Detach with Ctrl-B then D to keep Jupyter running. Reattach using
`tmux attach -t jupyter`. Jupyter prints a URL with a generated token.

In a terminal on your own computer, open an SSH tunnel (replace USER and VM_IP):

```bash
ssh -N -L 8888:127.0.0.1:8888 USER@VM_IP
```

For Google Cloud with gcloud authentication instead:

```bash
gcloud compute ssh USER@VM_NAME --project=PROJECT_ID --zone=ZONE -- \
  -N -L 8888:127.0.0.1:8888
```

Open `http://127.0.0.1:8888/lab` on your computer and enter the printed token.
Select the **gnn_project** kernel and open `notebooks/baseline.ipynb`.
The baseline training cell starts training only when you execute it. It
overwrites matching files in `results/baseline_run_01`; change `run_dir`
for a separate experiment. The final IoU cell expects epoch 20 prediction
files; adjust its path if you change the run length.

This package contains code/notebooks, not trained checkpoints. Rerun training
to generate predictions before running the IoU cell. STEP visualization uses
the separate OCCWL environment described in README.md.

For `notebooks/laplacian.ipynb`, first run the baseline or copy its final
`results/baseline_run_01/epoch_020/checkpoint.pt` into this project. The
notebook freezes that checkpoint's CNN, caches per-solid geometry/Laplacian
features, and trains a new classifier when you run its training cell.

`notebooks/graphlet.ipynb` requires the same baseline checkpoint. Choose
`max_graphlet_nodes=4` or `5` and `training_mode="frozen"` or `"joint"`.
Run each desired combination separately; each saves to its own results
folder. Counting is exact and may take time on larger graphs, but results
are cached. Feature scaling uses training data only. The notebook retains
the same per-epoch seed, RNG-state, weights and optimizer checkpointing.

`notebooks/wl.ipynb` uses the same completed baseline checkpoint and kernel.
Choose `wl_iterations` (default 3) and frozen/joint CNN mode. The training
colour vocabulary is fixed before validation is encoded, and is saved with
the checkpoints. The notebook reports unseen validation colours by round.
No dense all-pairs kernel or one-hot matrix is allocated. The extra colour
projection parameters are included in the reported model size.

`notebooks/chebyshev.ipynb` needs the same completed baseline checkpoint.
Run `cheb_hops=2` and `3`, each in frozen/joint CNN mode if desired. There is
one Chebyshev convolution, followed by a pointwise classifier, so the hop
setting is also the total graph receptive field. The sparse operator cache
is reused across hop counts. No extra dependencies or eigenvalue solver are
needed. Run the training cell yourself; it defaults to 20 epochs. Results
and comparison plots stay within the Chebyshev theme plus baseline.

`notebooks/anisotropic.ipynb` adds curve geometry through a learned EdgeCNN.
Use `edge_mode="static"` first, then `"updated"`; both have exactly two
graph layers. The default joint mode trains both CNNs. Frozen mode freezes
only the baseline FaceCNN because no pre-trained EdgeCNN is available.
Original directed edges and curve arrays are cached in matching order.
No extra dependencies are required. Training starts only when you execute
the training cell, with epoch checkpoints and final-only predictions.

## Seeded runs and resume

All model notebooks expose `seed`, `deterministic`, and `resume_checkpoint`
in their settings. `resume_checkpoint=None` starts a fresh seeded model.
Every completed epoch saves `model_weights.pt` and a full `checkpoint.pt`
with optimizer, RNG states, seed, metrics and AMP scaler where used.
To continue after epoch 10, set `resume_checkpoint` to its `checkpoint.pt`
and set `num_epochs=20` for a total of 20 epochs. Run preparation cells and
then the training cell. Resume does not reset the random sequence each epoch.
Keep the original Laplacian feature caches, dataset order, hardware/software
and training settings for reproducible replay. Older checkpoints without
RNG states can provide weights but cannot reproduce an exact continuation.

Stop Jupyter with Ctrl-C in its tmux session, close the SSH tunnel, and stop
the VM from your cloud console when finished.

## Edge encoder

```python
from uvnet_project.edge_cnn import EdgeCNN

edge_cnn = EdgeCNN(embedding_dim=64, in_channels=6)
edge_grids = graph.edata["x"].permute(0, 2, 1)  # edges, channels, samples
edge_embeddings = edge_cnn(edge_grids)
```

The channels are xyz and unit tangents; there is no trimming-mask channel.
Training mode needs more than one edge for the final BatchNorm1d layer.
Use `edge_cnn.eval()` and `torch.no_grad()` for inference checks.
