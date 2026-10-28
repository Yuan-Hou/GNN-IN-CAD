"""Command-line entry point: train one model (or all of them) on MFCAD."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import DATA_DIR, MFCADDataset, _load_splits
from .models import MODEL_FEATURES, MODEL_NAMES
from .training import train_one, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pe-dim", type=int, default=8,
                        help="non-trivial Laplacian eigenvectors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--test", action="store_true",
                        help="Evaluate test once, after validation selects the checkpoint")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads) <= 0:
        raise SystemExit("epochs, batch-size and threads must be positive")
    torch.set_num_threads(args.threads)
    splits = _load_splits(args.data_dir)
    for left, right in (("train", "validation"), ("train", "test"),
                        ("validation", "test")):
        if set(splits[left]) & set(splits[right]):
            raise ValueError(f"Overlapping split IDs: {left}, {right}")

    models = MODEL_NAMES if args.model == "all" else (args.model,)
    needed = tuple(sorted({feature for name in models
                           for feature in MODEL_FEATURES.get(name, ())}))
    datasets = {"train": MFCADDataset("train", args.data_dir, args.train_limit,
                                      features=needed, pe_dim=args.pe_dim),
                "validation": MFCADDataset("validation", args.data_dir, args.val_limit,
                                           features=needed, pe_dim=args.pe_dim)}
    if args.test:
        datasets["test"] = MFCADDataset("test", args.data_dir, args.test_limit,
                                        features=needed, pe_dim=args.pe_dim)

    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for model_name in models:
        results.append(train_one(args, model_name, datasets, args.output / model_name))
        write_json(args.output / "summary.json", results)
    return results


if __name__ == "__main__":
    main()
