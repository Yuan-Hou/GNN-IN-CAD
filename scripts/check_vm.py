"""Check VM dependencies, GPU, and encoders without training."""

import torch
import dgl
import matplotlib
import pandas
import tqdm

from uvnet_project.face_cnn import FaceCNN
from uvnet_project.edge_cnn import EdgeCNN
from uvnet_project.load_mfcad import DATA_DIR, load_mfcad


print("PyTorch:", torch.__version__, "DGL:", dgl.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU available. Check the NVIDIA driver and PyTorch installation.")
device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))
face_cnn = FaceCNN().to(device).eval()
edge_cnn = EdgeCNN().to(device).eval()
if (DATA_DIR / "split.json").is_file() and (DATA_DIR / "graph").is_dir():
    sample = load_mfcad("train", head=1)[0]
    face_grids = sample["graph"].ndata["x"].permute(0, 3, 1, 2).to(device)
    edge_grids = sample["graph"].edata["x"].permute(0, 2, 1).to(device)
    print("Sample:", sample["id"])
else:
    print("Dataset not unpacked; checking synthetic grids. Run setup_vm.sh --download-data.")
    face_grids = torch.zeros(2, 7, 10, 10, device=device)
    edge_grids = torch.zeros(2, 6, 10, device=device)
with torch.no_grad():
    faces = face_cnn(face_grids)
    edges = edge_cnn(edge_grids)
assert faces.shape == (len(face_grids), 64) and torch.isfinite(faces).all()
assert edges.shape == (len(edge_grids), 64) and torch.isfinite(edges).all()
print("Face embeddings:", tuple(faces.shape), "Edge embeddings:", tuple(edges.shape))
print("Checks passed. No training was run.")
