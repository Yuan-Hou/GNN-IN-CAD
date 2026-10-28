"""Per-solid B-rep graphs, UV normalization and the MFCAD dataset."""

from __future__ import annotations

from pathlib import Path

import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import graphlet_degree_vectors, laplacian_positional_encoding
from .loading import DATA_DIR, _load_sample, _load_splits

NUM_CLASSES = 16
DATASET_FEATURES = ("gdv", "pe", "edge")


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


def adjacency_graph(graph, edge_uv=None):
    """Bidirectional B-rep face-adjacency graph with one self-loop per face.

    Directed (trim-curve) edges are kept as they are, so each direction of an adjacency
    carries its own curve grid; exact duplicates are dropped (first occurrence wins),
    because DGL message passing would otherwise weight repeated edges more heavily.
    ``edge_uv`` must be aligned with ``graph.edges()`` and is stored as ``edata["xs"]``;
    self-loops get zero curve features so the ECC layer can learn a self term from them.
    """
    node_count = graph.num_nodes()
    src, dst = graph.edges()
    src, dst = src.numpy().astype(np.int64), dst.numpy().astype(np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    features = None
    if edge_uv is not None:
        if edge_uv.shape[0] != graph.num_edges():
            raise ValueError("Curve features are not aligned with the graph edges")
        features = edge_uv[torch.from_numpy(keep)]
    key = src * node_count + dst
    _, first = np.unique(key, return_index=True)
    src, dst = src[first], dst[first]
    if features is not None:
        features = features[torch.from_numpy(first)]

    nodes = torch.arange(node_count, dtype=torch.int64)
    result = dgl.graph((torch.cat([torch.from_numpy(src), nodes]),
                        torch.cat([torch.from_numpy(dst), nodes])), num_nodes=node_count)
    if features is not None:
        self_loops = torch.zeros(node_count, *features.shape[1:], dtype=features.dtype)
        result.edata["xs"] = torch.cat([features, self_loops])
    return result


class MFCADDataset(Dataset):
    """Load on demand and cache normalized samples; no in-place augmentation."""

    def __init__(self, split="train", data_dir=DATA_DIR, limit=None, sample_ids=None,
                 features=(), pe_dim=8):
        self.data_dir = Path(data_dir)
        split = "validation" if split == "val" else split
        splits = _load_splits(self.data_dir)
        if split not in splits:
            raise ValueError(f"Unknown split {split!r}")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        unknown = set(features) - set(DATASET_FEATURES)
        if unknown:
            raise ValueError(f"Unknown dataset features {sorted(unknown)}")
        self.features = tuple(features)
        self.pe_dim = pe_dim
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
            face_uv, edge_uv = normalize_uv(face_uv, edge_uv)
            graph = adjacency_graph(raw, edge_uv if "edge" in self.features else None)
            graph.ndata["x"], graph.ndata["y"] = face_uv, labels
            if "gdv" in self.features:
                graph.ndata["gdv"] = graphlet_degree_vectors(graph)
            if "pe" in self.features:
                graph.ndata["pe"] = laplacian_positional_encoding(graph, self.pe_dim)
            self.cache[index] = {"id": sample["id"], "graph": graph}
        return self.cache[index]


def collate_samples(samples):
    return {"ids": [sample["id"] for sample in samples],
            "graph": dgl.batch([sample["graph"] for sample in samples])}
