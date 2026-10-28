"""Map flat prediction rows to solids and zero-based graph face indices."""

import json
from pathlib import Path

import torch


def attach_prediction_index(results, sample_ids, split, data_dir=None):
    """Validate label order and attach a compact, explicit row index.

    Call only for evaluation in the given sample order (never shuffled).
    ``sample_ids[sample_indices[row]]`` identifies the solid; ``face_indices``
    identifies its graph node/annotation position, not an OCC STEP face ID.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data/MFCADDataset"
    data_dir = Path(data_dir)
    sample_ids = list(sample_ids)
    if split not in ("train", "validation", "test"):
        raise ValueError("Provide the dataset split for indexed predictions.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique within the evaluation split.")
    labels, counts = [], []
    for sample_id in sample_ids:
        if Path(sample_id).name != sample_id:
            raise ValueError("Sample IDs must be simple filenames.")
        path = data_dir / "labels" / f"{sample_id}_ids.json"
        faces = json.loads(path.read_text())["body"]["faces"]
        counts.append(len(faces))
        labels.extend(face["segment"]["index"] for face in faces)
    expected = torch.tensor(labels, dtype=torch.long)
    if not torch.equal(results["labels"].cpu().long(), expected):
        raise ValueError("Prediction label order does not match the supplied solid/face order.")
    if results["probabilities"].shape[0] != len(labels) or results["predictions"].shape != expected.shape:
        raise ValueError("Prediction rows do not match the supplied index.")
    counts = torch.tensor(counts, dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    sample_indices = torch.repeat_interleave(torch.arange(len(sample_ids)), counts)
    face_indices = torch.arange(len(labels)) - offsets[sample_indices]
    return {**results, "index_format_version": 1, "split": split,
            "sample_ids": sample_ids, "sample_indices": sample_indices,
            "face_indices": face_indices, "sample_offsets": offsets}
