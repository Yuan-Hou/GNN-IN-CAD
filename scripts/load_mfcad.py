"""Utilities for loading the preprocessed MFCAD DGL dataset."""

from __future__ import annotations

import json
from pathlib import Path

from dgl.data.utils import load_graphs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "MFCADDataset"


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    """Load the official train, validation, and test sample IDs."""
    split_path = data_dir / "split.json"
    with split_path.open() as file:
        return json.load(file)


def load_sample(sample_id: str, data_dir: Path = DATA_DIR) -> dict:
    """Load one DGL graph and its aligned per-face annotations."""
    graph_path = data_dir / "graph" / f"{sample_id}.bin"
    label_path = data_dir / "labels" / f"{sample_id}_ids.json"
    if not graph_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing graph or label file for {sample_id}")

    graphs, graph_metadata = load_graphs(str(graph_path))
    if len(graphs) != 1:
        raise ValueError(f"Expected one graph for {sample_id}, found {len(graphs)}")

    with label_path.open() as file:
        label_data = json.load(file)

    graph = graphs[0]
    faces = label_data["body"]["faces"]
    if graph.num_nodes() != len(faces):
        raise ValueError(
            f"{sample_id}: {graph.num_nodes()} graph nodes but "
            f"{len(faces)} face labels"
        )

    return {
        "id": sample_id,
        "graph": graph,
        "faces": faces,
        "graph_metadata": graph_metadata,
    }


def load_subset(
    split: str = "train",
    subset_size: int = 5,
    data_dir: Path = DATA_DIR,
) -> tuple[dict[str, list[str]], list[dict]]:
    """Load up to ``subset_size`` samples; a negative size loads the full split."""
    splits = load_splits(data_dir)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; choose from {tuple(splits)}")

    sample_ids = splits[split] if subset_size < 0 else splits[split][:subset_size]
    samples = [load_sample(sample_id, data_dir) for sample_id in sample_ids]
    return splits, samples
