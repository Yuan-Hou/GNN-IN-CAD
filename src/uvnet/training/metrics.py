"""Face-level metrics computed from a dataset-wide confusion matrix."""

from __future__ import annotations

import numpy as np
import torch

from ..data.graphs import NUM_CLASSES


def metrics_from_confusion(confusion):
    """Accuracy, dataset-level macro IoU and per-class IoU (classes with empty union skip)."""
    true_count, pred_count = confusion.sum(1), confusion.sum(0)
    intersection = confusion.diag()
    union = true_count + pred_count - intersection
    valid = union > 0
    iou = intersection.float() / union.clamp_min(1)
    return {"accuracy": (intersection.sum() / confusion.sum().clamp_min(1)).item(),
            "macro_iou": iou[valid].mean().item() if valid.any() else 0.0,
            "per_class_iou": [float(iou[i]) if valid[i] else None for i in range(NUM_CLASSES)],
            "class_support": true_count.tolist(), "confusion_matrix": confusion.tolist()}


def class_metrics(confusion):
    """Precision, recall, F1 and IoU per class from a confusion matrix (array or list)."""
    confusion = np.asarray(confusion, dtype=np.float64)
    true_count, pred_count = confusion.sum(1), confusion.sum(0)
    true_positive = np.diag(confusion)
    precision = np.divide(true_positive, pred_count, out=np.zeros_like(true_positive),
                          where=pred_count > 0)
    recall = np.divide(true_positive, true_count, out=np.zeros_like(true_positive),
                       where=true_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(true_positive), where=(precision + recall) > 0)
    union = true_count + pred_count - true_positive
    valid = union > 0
    iou = np.divide(true_positive, union, out=np.zeros_like(true_positive), where=valid)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "support": true_count.astype(int), "valid": valid,
            "macro_f1": float(f1[valid].mean()), "balanced_accuracy": float(recall[valid].mean())}


__all__ = ["metrics_from_confusion", "class_metrics"]
