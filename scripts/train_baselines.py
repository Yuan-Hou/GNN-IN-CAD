"""Reproducible CPU/CUDA training and evaluation for the three DGL baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import dgl
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from load_mfcad import DATA_DIR, _load_splits
from mfcad_dataset import MFCADDataset, NUM_CLASSES, collate_samples
from models import FaceModel, MODEL_NAMES


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def metrics_from_confusion(confusion):
    true_count, pred_count = confusion.sum(1), confusion.sum(0)
    intersection = confusion.diag()
    union = true_count + pred_count - intersection
    valid = union > 0
    iou = intersection.float() / union.clamp_min(1)
    return {"accuracy": (intersection.sum() / confusion.sum().clamp_min(1)).item(),
            "macro_iou": iou[valid].mean().item() if valid.any() else 0.0,
            "per_class_iou": [float(iou[i]) if valid[i] else None for i in range(NUM_CLASSES)],
            "class_support": true_count.tolist(), "confusion_matrix": confusion.tolist()}


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    loss_sum, faces = 0.0, 0
    start = time.perf_counter()
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index, batch in enumerate(loader, 1):
            graph = batch["graph"].to(device)
            labels = graph.ndata["y"]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(graph)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = labels.numel()
            loss_sum += loss.detach().item() * count
            faces += count
            indices = (labels * NUM_CLASSES + logits.argmax(1)).detach().cpu()
            confusion += torch.bincount(indices, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
            if batch_index % 100 == 0:
                phase = "train" if optimizer is not None else "eval"
                print(f"  {phase} batch={batch_index}/{len(loader)} loss={loss_sum / faces:.4f}", flush=True)
    return {"loss": loss_sum / faces, "faces": faces,
            "seconds": time.perf_counter() - start, **metrics_from_confusion(confusion)}


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def train_one(args, model_name, datasets, run_dir):
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    device = torch.device(args.device)
    model = FaceModel(model_name, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                 generator=generator if split == "train" else None,
                                 collate_fn=collate_samples, num_workers=0, drop_last=False)
               for split, dataset in datasets.items()}
    config = {**vars(args), "data_dir": str(args.data_dir), "output": str(args.output),
              "model": model_name, "parameters": sum(p.numel() for p in model.parameters()),
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "dgl": dgl.__version__, "numpy": np.__version__},
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("train_baselines.py", "models.py", "mfcad_dataset.py", "load_mfcad.py")},
              "split_ids": {split: dataset.ids for split, dataset in datasets.items()},
              "normalization": "whole-solid masked bbox, longest extent 2; GroupNorm/LayerNorm",
              "iou_policy": "dataset confusion; exclude classes with zero union"}
    write_json(run_dir / "config.json", config)
    best_loss, best_epoch = float("inf"), None
    start = time.perf_counter()
    with (run_dir / "history.jsonl").open("w") as history:
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, loaders["train"], device, optimizer)
            validation = run_epoch(model, loaders["validation"], device)
            record = {"epoch": epoch, "train": train, "validation": validation}
            history.write(json.dumps(record, allow_nan=False) + "\n")
            history.flush()
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "config": config, "validation": validation}, run_dir / "best.pt")
            print(f"{model_name} epoch={epoch:03d} train_loss={train['loss']:.4f} "
                  f"val_loss={validation['loss']:.4f} val_acc={validation['accuracy']:.4f} "
                  f"val_mIoU={validation['macro_iou']:.4f} "
                  f"seconds={train['seconds'] + validation['seconds']:.1f}", flush=True)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    result = {"model": model_name, "parameters": config["parameters"], "best_epoch": best_epoch,
              "validation": checkpoint["validation"], "elapsed_seconds": time.perf_counter() - start}
    if "test" in loaders:
        result["test"] = run_epoch(model, loaders["test"], device)
    write_json(run_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--test", action="store_true", help="Evaluate test once, after validation selects checkpoint")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) <= 0:
        parser.error("epochs, batch-size, and threads must be positive")
    torch.set_num_threads(args.threads)
    splits = _load_splits(args.data_dir)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(splits[left]) & set(splits[right]):
            raise ValueError(f"Overlapping split IDs: {left}, {right}")
    datasets = {"train": MFCADDataset("train", args.data_dir, args.train_limit),
                "validation": MFCADDataset("validation", args.data_dir, args.val_limit)}
    if args.test:
        datasets["test"] = MFCADDataset("test", args.data_dir, args.test_limit)
    args.output.mkdir(parents=True, exist_ok=False)
    models = MODEL_NAMES if args.model == "all" else (args.model,)
    results = []
    for model_name in models:
        results.append(train_one(args, model_name, datasets, args.output / model_name))
        write_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
