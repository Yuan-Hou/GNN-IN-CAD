"""Audit every published graph and annotation before training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .data import DATA_DIR, NUM_CLASSES, _load_sample, _load_splits, normalize_uv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    splits = _load_splits(args.data_dir)
    seen, mapping, result = set(), {}, {}
    for split, ids in splits.items():
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError(f"Duplicate or overlapping sample IDs in {split}")
        seen.update(ids)
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        face_counts, edge_counts = [], []
        missing_reverse = loops = duplicates = zero_edge_graphs = 0
        xyz_min, xyz_max = float("inf"), -float("inf")
        mask_counts = Counter()
        for position, sample_id in enumerate(ids):
            sample = _load_sample(sample_id, args.data_dir)
            graph = sample["graph"]
            faces, edges = graph.ndata["x"], graph.edata["x"]
            if (faces.shape != (graph.num_nodes(), 10, 10, 7)
                    or edges.shape != (graph.num_edges(), 10, 6)):
                raise ValueError(f"Invalid shape: {sample_id}")
            normalize_uv(faces, edges)
            for face in sample["faces"]:
                index, name = face["segment"]["index"], face["segment"]["name"]
                if not 0 <= index < NUM_CLASSES or (index in mapping and mapping[index] != name):
                    raise ValueError(f"Invalid label mapping: {sample_id}")
                mapping[index] = name
                counts[index] += 1
            src, dst = graph.edges()
            pairs = set(zip(src.tolist(), dst.tolist()))
            duplicates += graph.num_edges() - len(pairs)
            loops += sum(left == right for left, right in pairs)
            missing_reverse += sum((right, left) not in pairs for left, right in pairs)
            zero_edge_graphs += graph.num_edges() == 0
            face_counts.append(graph.num_nodes())
            edge_counts.append(graph.num_edges())
            valid_xyz = faces[..., :3][faces[..., 6] == 1]
            xyz_min = min(xyz_min, valid_xyz.min().item())
            xyz_max = max(xyz_max, valid_xyz.max().item())
            values, frequencies = torch.unique(faces[..., 6], return_counts=True)
            mask_counts.update({str(value): count for value, count in
                                zip(values.tolist(), frequencies.tolist())})
            if (position + 1) % 1000 == 0:
                print(f"{split}: audited {position + 1}/{len(ids)}", flush=True)
        result[split] = {"solids": len(ids), "faces": sum(face_counts),
                         "raw_graph_edges": sum(edge_counts),
                         "min_faces": min(face_counts), "max_faces": max(face_counts),
                         "class_counts": counts.tolist(),
                         "missing_reverse_edges": missing_reverse,
                         "self_loops": loops, "parallel_edges": duplicates,
                         "zero_edge_graphs": zero_edge_graphs,
                         "valid_xyz_range": [xyz_min, xyz_max],
                         "mask_values": dict(mask_counts)}
    result["class_names"] = mapping
    result["alignment"] = ("face JSON order follows official loader; counts checked; "
                           "not independently proven")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
