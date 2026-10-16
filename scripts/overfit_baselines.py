"""Verify each baseline can fit one real solid; not a validation benchmark."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from mfcad_dataset import MFCADDataset
from models import FaceModel, MODEL_NAMES
from train_baselines import seed_all


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    sample = MFCADDataset(limit=1)[0]
    graph = sample["graph"]
    labels = graph.ndata["y"]
    results = []
    for name in MODEL_NAMES:
        seed_all(42)
        model = FaceModel(name, dropout=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial_loss = F.cross_entropy(model(graph), labels).item()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(graph), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(graph)
            final_loss = F.cross_entropy(logits, labels).item()
            accuracy = (logits.argmax(1) == labels).float().mean().item()
        passed = final_loss < initial_loss * .2 and accuracy >= .95
        record = {"model": name, "sample_id": sample["id"], "faces": graph.num_nodes(),
                  "steps": args.steps, "initial_loss": initial_loss, "final_loss": final_loss,
                  "accuracy": accuracy, "passed": passed}
        results.append(record)
        print(json.dumps(record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not all(record["passed"] for record in results):
        raise RuntimeError("At least one model failed the single-solid overfit check")


if __name__ == "__main__":
    main()
