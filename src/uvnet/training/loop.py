"""Seeded training/evaluation loops shared by every model."""

from __future__ import annotations

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

from ..data.graphs import NUM_CLASSES, collate_samples
from ..models import FaceModel
from .metrics import metrics_from_confusion

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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
            confusion += torch.bincount(indices, minlength=NUM_CLASSES ** 2).reshape(
                NUM_CLASSES, NUM_CLASSES)
            if batch_index % 100 == 0:
                phase = "train" if optimizer is not None else "eval"
                print(f"  {phase} batch={batch_index}/{len(loader)} "
                      f"loss={loss_sum / faces:.4f}", flush=True)
    return {"loss": loss_sum / faces, "faces": faces,
            "seconds": time.perf_counter() - start, **metrics_from_confusion(confusion)}


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def source_hashes():
    """SHA-256 of every package source file, so a run records exactly what it ran."""
    return {str(path.relative_to(PACKAGE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(PACKAGE_ROOT.rglob("*.py"))}


def train_one(args, model_name, datasets, run_dir):
    run_dir.mkdir(parents=True, exist_ok=False)
    seed_all(args.seed)
    device = torch.device(args.device)
    model = FaceModel(model_name, dropout=args.dropout, pe_dim=args.pe_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {split: DataLoader(dataset, batch_size=args.batch_size, shuffle=split == "train",
                                 generator=generator if split == "train" else None,
                                 collate_fn=collate_samples, num_workers=0, drop_last=False)
               for split, dataset in datasets.items()}
    config = {**vars(args), "data_dir": str(args.data_dir), "output": str(args.output),
              "model": model_name,
              "parameters": sum(parameter.numel() for parameter in model.parameters()),
              "dataset_features": list(datasets["train"].features),
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "dgl": dgl.__version__, "numpy": np.__version__},
              "source_sha256": source_hashes(),
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
            history.write(json.dumps({"epoch": epoch, "train": train,
                                      "validation": validation}, allow_nan=False) + "\n")
            history.flush()
            if validation["loss"] < best_loss:
                best_loss, best_epoch = validation["loss"], epoch
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "epoch": epoch, "config": config, "validation": validation},
                           run_dir / "best.pt")
            print(f"{model_name} epoch={epoch:03d} train_loss={train['loss']:.4f} "
                  f"val_loss={validation['loss']:.4f} val_acc={validation['accuracy']:.4f} "
                  f"val_mIoU={validation['macro_iou']:.4f} "
                  f"seconds={train['seconds'] + validation['seconds']:.1f}", flush=True)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    result = {"model": model_name, "parameters": config["parameters"], "best_epoch": best_epoch,
              "validation": checkpoint["validation"],
              "elapsed_seconds": time.perf_counter() - start}
    if "test" in loaders:
        result["test"] = run_epoch(model, loaders["test"], device)
    write_json(run_dir / "result.json", result)
    return result


__all__ = ["seed_all", "run_epoch", "write_json", "source_hashes", "train_one"]
