from __future__ import annotations


"""Utilities for loading the preprocessed MFCAD DGL dataset."""


import json
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "MFCADDataset"


def _load_splits(data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    """Load the official train, validation, and test sample IDs."""
    split_path = data_dir / "split.json"
    with split_path.open() as file:
        return json.load(file)


def _load_sample(sample_id: str, data_dir: Path = DATA_DIR) -> dict:
    """Load one DGL graph and its aligned per-face annotations."""
    graph_path = data_dir / "graph" / f"{sample_id}.bin"
    label_path = data_dir / "labels" / f"{sample_id}_ids.json"
    if not graph_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing graph or label file for {sample_id}")

    from dgl.data.utils import load_graphs
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


def load_mfcad(
    split: str = "train",
    head: int | None = None,
    data_dir: Path = DATA_DIR,
) -> list[dict]:
    """Load an MFCAD split, optionally limited to its first ``head`` samples."""
    splits = _load_splits(data_dir)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}; choose from {tuple(splits)}")
    if head is not None and head < 0:
        raise ValueError("head must be non-negative or None")

    sample_ids = splits[split] if head is None else splits[split][:head]
    return [_load_sample(sample_id, data_dir) for sample_id in sample_ids]


"""Reviewed semantic names for supplied label names; raw IDs/files remain unchanged."""
from pathlib import Path
import json

MAPPING_PATH = Path(__file__).resolve().parents[1] / 'data' / 'label_mapping.json'
LABEL_MAPPING = json.loads(MAPPING_PATH.read_text())
if len(LABEL_MAPPING) != 16 or set(LABEL_MAPPING) != set(LABEL_MAPPING.values()):
    raise ValueError('Expected a one-to-one mapping of all 16 classes, including stock.')

# Presentation order established by the EDA: feature family, then profile shape.
# Reuse for class tables, axes, heatmaps and legends in subsequent analysis.
CLASS_ORDER = json.loads((PROJECT_ROOT / "data/class_order.json").read_text())
if len(CLASS_ORDER) != 16 or set(CLASS_ORDER) != set(LABEL_MAPPING.values()):
    raise ValueError("Class order must contain each reviewed class exactly once.")


def ordered_class_indices(class_names, *, include_stock=True):
    """Map the shared semantic display order to supplied numeric class IDs."""
    by_name = {name: int(index) for index, name in class_names.items()}
    return [by_name[name] for name in CLASS_ORDER
            if name in by_name and (include_stock or name != "stock")]


def corrected_label(supplied_name):
    return LABEL_MAPPING[supplied_name]


"""Unpack the dataset bundled with the report. Requires 7zz or 7z."""
from pathlib import Path
import shutil
import subprocess
import zipfile


def unpack_data():
    report_dir = Path(__file__).resolve().parents[1]
    archive_path = report_dir / "data" / "MFCADDataset.zip"
    data_dir = report_dir / "data" / "MFCADDataset"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Place the submitted dataset ZIP at {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        # The published ZIP has no wrapper; also accept a MFCADDataset/ wrapper.
        names = [name for name in archive.namelist() if not name.startswith("__MACOSX/")]
        wrapped = any(name.startswith("MFCADDataset/") for name in names)
        destination = data_dir.parent if wrapped else data_dir
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)

    for folder in ("graph", "labels"):
        nested_archive = data_dir / f"{folder}.7z"
        if nested_archive.is_file():
            extractor = shutil.which("7zz") or shutil.which("7z")
            if extractor is None:
                raise RuntimeError("Install 7-Zip (7zz or 7z), then rerun this cell.")
            subprocess.run(
                [extractor, "x", "-y", f"-o{data_dir}", str(nested_archive)],
                check=True, capture_output=True, text=True,
            )

    for required in (data_dir / "split.json", data_dir / "graph", data_dir / "labels"):
        if not required.exists():
            raise FileNotFoundError(f"Dataset extraction is incomplete: missing {required}")

    print(f"Dataset ready: {data_dir.relative_to(report_dir)}")
    print(f"Graphs: {len(list((data_dir / 'graph').glob('*.bin'))):,}")
    print(f"Label files: {len(list((data_dir / 'labels').glob('*.json'))):,}")
