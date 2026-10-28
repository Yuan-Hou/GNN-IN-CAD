"""Render the test confusion matrix of every model as PNG + CSV files.

Outputs
-------
``results/confusion/<model>_seed42.png`` and ``<model>_seeds42-1-2.png``
    Row-normalized matrices (colour = recall) with raw counts annotated.
``results/confusion/<model>_{seed42,seeds42-1-2}_{counts,row_normalized}.csv``
    The same matrices as CSV (class names as headers).
``results/figures/confusion_matrices_all.png``
    Overview grid with all seven models (seed 42).
``results/confusion_matrices.zip``
    Bundle of every file above.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUT = RESULTS / "confusion"
SEEDS = (42, 1, 2)
RUN_DIRS = {
    "cnn": "full-seed{seed}/cnn",
    "complete_gcn": "full-seed{seed}/complete_gcn",
    "adjacency_gcn": "full-seed{seed}/adjacency_gcn",
    "topo_gdv": "topo-gdv-seed{seed}/topo_gdv",
    "adjacency_gcn_lape": "lape-seed{seed}/adjacency_gcn_lape",
    "uvnet": "uvnet-seed{seed}/uvnet",
    "uvnet_lape": "uvnet-lape-seed{seed}/uvnet_lape",
}
LABELS = {
    "cnn": "CNN (geometry only)",
    "complete_gcn": "Complete GCN (global)",
    "adjacency_gcn": "Adjacency GCN (B-rep)",
    "topo_gdv": "Topo-GDV (topology only)",
    "adjacency_gcn_lape": "Adjacency GCN + LapPE",
    "uvnet": "UV-Net (surface + curve + ECC)",
    "uvnet_lape": "UV-Net + LapPE",
}

AUDIT = json.loads((RESULTS / "data_audit.json").read_text())
CLASS_NAMES = [AUDIT["class_names"][str(index)] for index in range(len(AUDIT["class_names"]))]
LABELS_SHORT = [name.replace("_", "\n", 1) if len(name) > 18 else name for name in CLASS_NAMES]


def load_matrix(model, seed=42):
    path = RESULTS / RUN_DIRS[model].format(seed=seed) / "result.json"
    return np.asarray(json.loads(path.read_text())["test"]["confusion_matrix"], dtype=np.int64)


def metrics(matrix):
    true_count, pred_count = matrix.sum(1), matrix.sum(0)
    diagonal = np.diag(matrix)
    union = true_count + pred_count - diagonal
    valid = union > 0
    iou = np.divide(diagonal, union, out=np.zeros(len(matrix)), where=valid)
    return diagonal.sum() / matrix.sum(), iou[valid].mean()


def write_csv(path, matrix, normalized=False):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred"] + CLASS_NAMES)
        for name, row in zip(CLASS_NAMES, matrix):
            if normalized:
                total = row.sum()
                writer.writerow([name] + [round(float(value) / total, 4) if total else 0.0
                                          for value in row])
            else:
                writer.writerow([name] + [int(value) for value in row])


def plot(matrix, title, path, figsize=(9.5, 8.5)):
    row_sums = matrix.sum(1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float),
                           where=row_sums > 0)
    accuracy, macro_iou = metrics(matrix)

    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if value:
                axis.text(column, row, f"{value}", ha="center", va="center", fontsize=6.5,
                          color="white" if normalized[row, column] > .55 else "#1a1a1a")
    figure.colorbar(image, ax=axis, label="row-normalized count (recall)")
    axis.set_title(f"{title}\ntest faces {int(matrix.sum())}, accuracy {100 * accuracy:.2f}%, "
                   f"macro IoU {100 * macro_iou:.2f}%", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def grid(matrices, path):
    figure, axes = plt.subplots(4, 2, figsize=(17, 20))
    for axis, model in zip(axes.ravel(), RUN_DIRS):
        matrix = matrices[model]
        row_sums = matrix.sum(1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float),
                               where=row_sums > 0)
        accuracy, macro_iou = metrics(matrix)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(range(len(CLASS_NAMES)),
                        [name if index % 2 == 0 else "" for index, name in enumerate(CLASS_NAMES)],
                        rotation=45, ha="right", fontsize=5)
        axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=5)
        axis.set_title(f"{LABELS[model]}\naccuracy {100 * accuracy:.2f}%, "
                       f"macro IoU {100 * macro_iou:.2f}%", fontsize=9)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                if matrix[row, column] and matrix[row, column] > 20:
                    axis.text(column, row, f"{matrix[row, column]}", ha="center", va="center",
                              fontsize=4, color="white" if normalized[row, column] > .55
                              else "#222222")
    axes.ravel()[-1].axis("off")
    figure.suptitle("Test confusion matrices, seed 42 (colour = recall, annotated counts > 20)",
                    fontsize=13, y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)

    seed42 = {}
    for model in RUN_DIRS:
        matrix = load_matrix(model, 42)
        seed42[model] = matrix
        plot(matrix, f"{LABELS[model]} - seed 42", OUT / f"{model}_seed42.png")
        write_csv(OUT / f"{model}_seed42_counts.csv", matrix)
        write_csv(OUT / f"{model}_seed42_row_normalized.csv", matrix, normalized=True)

        total = sum(load_matrix(model, seed) for seed in SEEDS)
        plot(total, f"{LABELS[model]} - seeds 42/1/2 (counts summed)",
             OUT / f"{model}_seeds42-1-2.png")
        write_csv(OUT / f"{model}_seeds42-1-2_counts.csv", total)
        write_csv(OUT / f"{model}_seeds42-1-2_row_normalized.csv", total, normalized=True)

    grid(seed42, RESULTS / "figures" / "confusion_matrices_all.png")

    (OUT / "README.md").write_text(
        "# Test confusion matrices\n\n"
        "One PNG and two CSVs per model: `<model>_seed42.*` is the seed-42 run used in the\n"
        "report figures; `<model>_seeds42-1-2.*` sums the test counts of seeds 42, 1 and 2\n"
        "(same test split, so the matrix is directly comparable). PNGs are coloured by\n"
        "row-normalized counts (recall) and annotate raw counts; `*_counts.csv` holds raw\n"
        "counts and `*_row_normalized.csv` the row-normalized matrix. The overview grid is\n"
        "`figures/confusion_matrices_all.png`. Regenerate with:\n\n"
        "```sh\npython analysis/make_confusion_matrices.py\n```\n")

    with zipfile.ZipFile(RESULTS / "confusion_matrices.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.glob("*")):
            bundle.write(path, path.relative_to(RESULTS))
        bundle.write(RESULTS / "figures" / "confusion_matrices_all.png",
                     "figures/confusion_matrices_all.png")

    print(f"wrote {len(list(OUT.glob('*')))} files to {OUT}")
    for model, matrix in seed42.items():
        accuracy, macro_iou = metrics(matrix)
        print(f"{model:20s} accuracy {100 * accuracy:6.2f}%  macro IoU {100 * macro_iou:6.2f}%")


if __name__ == "__main__":
    main()
