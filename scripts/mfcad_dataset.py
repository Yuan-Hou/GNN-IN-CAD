"""Training data for per-face MFCAD segmentation (CPU DGL graphs)."""

from __future__ import annotations

from pathlib import Path

import dgl
import torch
from torch.utils.data import Dataset

from load_mfcad import DATA_DIR, _load_sample, _load_splits

NUM_CLASSES = 16


def adjacency_graph(graph):
    """A separate binary, bidirectional propagation graph with one self-loop."""
    src, dst = graph.edges()
    result = dgl.graph((torch.cat((src, dst)), torch.cat((dst, src))),
                       num_nodes=graph.num_nodes())
    return dgl.add_self_loop(dgl.remove_self_loop(dgl.to_simple(result)))


def normalize_uv(face_uv, edge_uv):
    """Normalize a whole solid, using only valid surface samples for its bbox."""
    face_uv, edge_uv = face_uv.float().clone(), edge_uv.float().clone()
    if not torch.isfinite(face_uv).all() or not torch.isfinite(edge_uv).all():
        raise ValueError("UV grids contain NaN or Inf")
    mask = face_uv[..., 6]
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Expected a binary trimming mask")
    points = face_uv[..., :3][mask == 1]
    if points.numel() == 0:
        raise ValueError("Solid has no valid surface samples")
    lower, upper = points.amin(dim=0), points.amax(dim=0)
    extent = (upper - lower).max()
    if extent <= 0:
        raise ValueError("Solid has a zero-size bounding box")
    center, scale = (lower + upper) / 2, 2 / extent
    face_uv[..., :3] = (face_uv[..., :3] - center) * scale
    edge_uv[..., :3] = (edge_uv[..., :3] - center) * scale
    return face_uv, edge_uv


class MFCADDataset(Dataset):
    """Load on demand and cache normalized samples; no in-place augmentation."""

    def __init__(self, split="train", data_dir=DATA_DIR, limit=None, sample_ids=None):
        self.data_dir = Path(data_dir)
        split = "validation" if split == "val" else split
        splits = _load_splits(self.data_dir)
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        self.ids = list(splits[split] if sample_ids is None else sample_ids)
        if not set(self.ids).issubset(splits[split]):
            raise ValueError("Sample selection contains IDs outside the split")
        if limit is not None:
            self.ids = self.ids[:limit]
        if not self.ids:
            raise ValueError(f"Empty split: {split}")
        self.cache = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if index not in self.cache:
            sample = _load_sample(self.ids[index], self.data_dir)
            raw = sample["graph"]
            face_uv, edge_uv = raw.ndata["x"], raw.edata["x"]
            if face_uv.shape != (raw.num_nodes(), 10, 10, 7):
                raise ValueError(f"{sample['id']}: unexpected face UV shape")
            if edge_uv.shape != (raw.num_edges(), 10, 6):
                raise ValueError(f"{sample['id']}: unexpected curve UV shape")
            labels = torch.tensor([face["segment"]["index"] for face in sample["faces"]],
                                  dtype=torch.long)
            if labels.numel() == 0 or labels.min() < 0 or labels.max() >= NUM_CLASSES:
                raise ValueError(f"{sample['id']}: invalid face labels")
            face_uv, _ = normalize_uv(face_uv, edge_uv)
            graph = adjacency_graph(raw)
            graph.ndata["x"], graph.ndata["y"] = face_uv, labels
            self.cache[index] = {"id": sample["id"], "graph": graph}
        return self.cache[index]


def collate_samples(samples):
    return {"ids": [sample["id"] for sample in samples],
            "graph": dgl.batch([sample["graph"] for sample in samples])}
