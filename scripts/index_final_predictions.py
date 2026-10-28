"""Backfill explicit solid/face indices in final caches without inference.

Run from the project: python scripts/index_final_predictions.py
Weights/checkpoints are untouched: other runs refer to baseline file hashes.
"""
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch

from uvnet_project.prediction_index import attach_prediction_index


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/MFCADDataset"
    splits = json.loads((data_dir / "split.json").read_text())
    manifest_path = root / "results/final_analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        folders = [root / run for run in manifest["runs"]]
    else:
        folders = []
        for run in sorted((root / "results").iterdir()):
            if not run.is_dir():
                continue
            completed = [p for p in run.glob("epoch_*") if p.name[6:].isdigit()
                         and (p / "checkpoint.pt").is_file()
                         and all((p / f"{split}_predictions.pt").is_file()
                                 for split in ("train", "validation"))]
            if completed:
                folders.append(max(completed, key=lambda p: int(p.name[6:])))
    backup = root / "notebook_backups" / ("before_prediction_index_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    changed = 0
    for folder in folders:
        checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        metadata = {**checkpoint.get("extra", {}), **checkpoint["config"]}
        del checkpoint
        for split in ("train", "validation"):
            ids = metadata.get(f"{split}_sample_ids")
            provenance = "checkpoint" if ids is not None else "dataset_split_label_validated"
            if ids is None:
                ids = splits[split]
            path = folder / f"{split}_predictions.pt"
            results = torch.load(path, map_location="cpu", weights_only=True)
            indexed = attach_prediction_index(results, ids, split, data_dir)
            indexed["index_source"] = provenance
            if results.get("index_format_version") == 1:
                assert results["sample_ids"] == indexed["sample_ids"]
                for key in ("sample_indices", "face_indices", "sample_offsets"):
                    assert torch.equal(results[key], indexed[key]), (path, key)
                continue
            previous = backup / path.relative_to(root)
            previous.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, previous)
            temporary = path.with_suffix(".pt.tmp")
            torch.save(indexed, temporary)
            os.replace(temporary, path)
            # Confirm only metadata was added, with numerical results unchanged.
            saved = torch.load(path, map_location="cpu", weights_only=True)
            for key, value in results.items():
                assert torch.equal(saved[key], value) if isinstance(value, torch.Tensor) else saved[key] == value
            changed += 1
            print(path.relative_to(root), "indexed", len(saved["sample_ids"]), "solids", flush=True)
        del metadata
    if manifest:
        # Keep the local inventory accurate after the metadata-only migration.
        for entry in manifest["files"]:
            path = root / entry["path"]
            if path.name not in ("train_predictions.pt", "validation_predictions.pt"):
                continue
            original = entry["sha256"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != original:
                entry.setdefault("source_sha256_before_index", original)
            entry["sha256"] = digest
            entry["bytes"] = path.stat().st_size
        manifest["prediction_index_format_version"] = 1
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Updated", changed, "prediction files across", len(folders), "runs. Checkpoints unchanged.")
    if changed:
        print("Backups:", backup)


if __name__ == "__main__":
    main()
