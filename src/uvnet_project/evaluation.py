"""Multiclass evaluation and prediction storage."""

from pathlib import Path

import torch
from torch import nn


def evaluate(model, evaluation_set, output_path=None, device=None):
    """Evaluate a model, optionally saving results with ``torch.save``.

    ``evaluation_set`` yields (inputs, labels) batches. Inputs must support
    ``.to(device)`` (e.g. tensors or DGL graphs), and ``model(inputs)`` must
    return (N, C) raw logits. Labels are a 1D long tensor of class indices.
    For segmentation, N is the number of faces, not the number of solids.

    Returns a dictionary containing mean cross-entropy loss,
    accuracy in [0, 1], confusion matrix (rows=true, columns=predicted),
    softmax probabilities, labels, and predictions. All tensors are on CPU
    and retain evaluation iteration order, including order within batches.
    Use a non-shuffled loader when matching rows back to source faces.
    Pass ``output_path`` to save the dictionary; None performs no file writes.
    The model is already on ``device``; inputs and labels are moved there.
    """
    if device is None:
        device = next(model.parameters(), torch.empty(0)).device

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    probabilities = []
    labels_all = []
    num_classes = None
    was_training = model.training
    model.eval()

    try:
        with torch.no_grad():
            for inputs, labels in evaluation_set:
                inputs = inputs.to(device)
                labels = labels.to(device)
                logits = model(inputs)
                if logits.ndim != 2 or labels.ndim != 1:
                    raise ValueError("Expected logits (N, C) and labels (N,).")
                if logits.shape[0] != labels.shape[0]:
                    raise ValueError("Logits and labels must have the same number of rows.")
                if num_classes is None:
                    num_classes = logits.shape[1]
                elif logits.shape[1] != num_classes:
                    raise ValueError("Number of classes changed between batches.")
                if labels.numel() == 0:
                    continue
                # Negative labels (including the loss's default ignore index)
                # are not valid classes in this evaluator.
                if labels.min().item() < 0 or labels.max().item() >= num_classes:
                    raise ValueError("Labels must be in [0, num_classes).")

                total_loss += loss_fn(logits, labels).item()
                probabilities.append(torch.softmax(logits, dim=1).cpu())
                labels_all.append(labels.cpu())
    finally:
        model.train(was_training)

    if not labels_all:
        raise ValueError("The evaluation set contains no labelled examples.")

    probabilities = torch.cat(probabilities)
    labels = torch.cat(labels_all)
    predictions = probabilities.argmax(dim=1)
    confusion_matrix = torch.bincount(
        labels * num_classes + predictions, minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    results = {
        "loss": total_loss / labels.numel(),
        "accuracy": (predictions == labels).float().mean().item(),
        "confusion_matrix": confusion_matrix,
        "probabilities": probabilities,
        "labels": labels,
        "predictions": predictions,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, output_path)
    return results
