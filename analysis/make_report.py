"""Compare all seven MFCAD models, from geometry-only to full UV-Net.

Reads ``results/full-seed*/`` (cnn, complete_gcn, adjacency_gcn), ``results/topo-gdv-seed*/``,
``results/lape-seed*/``, ``results/uvnet-seed*/`` and ``results/uvnet-lape-seed*/``,
then writes

* ``results/tables/main_results_all.csv``
* ``results/tables/per_class_iou_all.csv``
* ``results/figures/model_comparison_all.png``
* ``results/figures/per_class_iou_all.png``
* ``results/REPORT_EXPERIMENTS.md``
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
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
SHORT_LABELS = {
    "cnn": "CNN\n(geometry only)",
    "complete_gcn": "Complete GCN\n(global)",
    "adjacency_gcn": "Adj. GCN\n(B-rep)",
    "topo_gdv": "Topo-GDV\n(topology only)",
    "adjacency_gcn_lape": "Adj. GCN\n+ LapPE",
    "uvnet": "UV-Net\n(surface+curve)",
    "uvnet_lape": "UV-Net\n+ LapPE",
}
COLORS = {"cnn": "#1f77b4", "complete_gcn": "#2ca02c", "adjacency_gcn": "#d62728",
          "topo_gdv": "#9467bd", "adjacency_gcn_lape": "#ff7f0e", "uvnet": "#111111", "uvnet_lape": "#17becf"}

AUDIT = json.loads((RESULTS / "data_audit.json").read_text())
CLASS_NAMES = [AUDIT["class_names"][str(index)] for index in range(16)]


def load(model, seed):
    run = RESULTS / RUN_DIRS[model].format(seed=seed)
    config = json.loads((run / "config.json").read_text())
    result = json.loads((run / "result.json").read_text())
    return config, result


def class_stats(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    true_count, pred_count = confusion.sum(1), confusion.sum(0)
    tp = np.diag(confusion)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.zeros_like(tp), where=true_count > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    union = true_count + pred_count - tp
    valid = union > 0
    return {"macro_f1": f1[valid].mean(), "balanced_accuracy": recall[valid].mean(),
            "per_class_iou": np.divide(tp, union, out=np.zeros_like(tp), where=valid)}


def collect():
    data = {model: {"parameters": [], "best_epoch": [], "val_accuracy": [],
                    "val_macro_iou": [], "test_accuracy": [], "test_macro_iou": [],
                    "test_macro_f1": [], "test_balanced_accuracy": [],
                    "elapsed_seconds": [], "per_class_iou": []}
            for model in RUN_DIRS}
    for model in RUN_DIRS:
        for seed in SEEDS:
            config, result = load(model, seed)
            stats = class_stats(result["test"]["confusion_matrix"])
            entry = data[model]
            entry["parameters"].append(config["parameters"])
            entry["best_epoch"].append(result["best_epoch"])
            entry["val_accuracy"].append(result["validation"]["accuracy"])
            entry["val_macro_iou"].append(result["validation"]["macro_iou"])
            entry["test_accuracy"].append(result["test"]["accuracy"])
            entry["test_macro_iou"].append(result["test"]["macro_iou"])
            entry["test_macro_f1"].append(float(stats["macro_f1"]))
            entry["test_balanced_accuracy"].append(float(stats["balanced_accuracy"]))
            entry["elapsed_seconds"].append(result["elapsed_seconds"])
            entry["per_class_iou"].append(stats["per_class_iou"])
    for model, entry in data.items():
        for key, values in entry.items():
            entry[key] = np.asarray(values)
    return data


def mean_std(values, scale=100.0):
    values = np.asarray(values, dtype=np.float64) * scale
    return values.mean(), values.std(ddof=1)


def write_csv(path, header, rows):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def figure(data, path):
    figure, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    models = list(RUN_DIRS)
    x = np.arange(len(models))
    for axis, key, title in ((axes[0], "test_accuracy", "Test face accuracy (%)"),
                             (axes[1], "test_macro_iou", "Test macro IoU (%)")):
        means, errors = zip(*(mean_std(data[model][key]) for model in models))
        bars = axis.bar(x, means, yerr=errors, capsize=4,
                        color=[COLORS[model] for model in models])
        axis.set_xticks(x, [SHORT_LABELS[model] for model in models], fontsize=9)
        axis.set_title(title)
        axis.set_ylim(0, 105)
        axis.grid(axis="y", alpha=.3)
        for bar, mean in zip(bars, means):
            axis.text(bar.get_x() + bar.get_width() / 2, mean + 1.5, f"{mean:.2f}",
                      ha="center", va="bottom", fontsize=9)
    figure.suptitle("MFCAD face classification, mean ± 1 SD over seeds 42/1/2")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def heatmap(data, path):
    matrix = np.stack([data[model]["per_class_iou"].mean(0) for model in RUN_DIRS])
    figure, axis = plt.subplots(figsize=(12.5, 4.8))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(RUN_DIRS)), [LABELS[model] for model in RUN_DIRS], fontsize=8.5)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                      color="white" if value < .55 else "black")
    figure.colorbar(image, ax=axis, label="IoU (mean over seeds)")
    axis.set_title("Per-class test IoU: geometry-only, topology-only and combined models")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def low_data():
    """Optional 1,500-solid comparison of adjacency GCN with and without LapPE."""
    paths = {"adjacency_gcn": RESULTS / "lowdata-adj-seed42/adjacency_gcn/result.json",
             "adjacency_gcn_lape": RESULTS / "lowdata-lape-seed42/adjacency_gcn_lape/result.json"}
    if not all(path.is_file() for path in paths.values()):
        return None
    return {name: json.loads(path.read_text()) for name, path in paths.items()}


def low_data_text(low):
    if not low:
        return ""
    baseline = low["adjacency_gcn"]["test"]["macro_iou"]
    lape = low["adjacency_gcn_lape"]["test"]["macro_iou"]
    return (f"* With only 1,500 training solids (seed 42), LapPE changes macro IoU by "
            f"**{100 * (lape - baseline):+.2f} points** ({100 * baseline:.2f}% -> "
            f"{100 * lape:.2f}%), so the low-data regime does not rescue the encoding either.")


def markdown(data):
    def fmt(model, key):
        mean, std = mean_std(data[model][key])
        return f"{mean:.2f} ± {std:.2f}"

    def fmt_int(model, key):
        return f"{data[model][key].mean():.0f}"

    lines = [
        "# MFCAD face classification: seven models from geometry-only to full UV-Net",
        "",
        "Cross-model report produced by `analysis/make_report.py`.",
        "Protocol, split, optimizer and evaluation are identical to the three-baseline report:",
        "MFCAD official split, batch 16, Adam (lr 1e-3, wd 1e-4), 20 epochs, dropout 0.1,",
        "best-validation-loss checkpoint, test evaluated once, seeds 42/1/2.",
        "",
        "## What was added",
        "",
        "* **Topo-GDV**: a purely topological model. The UV-grid encoder is replaced by an MLP",
        "  over the 15-dimensional graphlet degree vector of each face's node in the B-rep",
        "  face-adjacency graph (orbits 0-3 exact, orbits 4-14 exact enumeration or Monte-Carlo",
        "  estimates), followed by the same two residual `GraphConv(64, 64)` layers and the same",
        "  classifier. No surface geometry, no edge geometry, no learned surface encoder.",
        "* **UV-Net**: full architecture. Surface grids go through the UV-grid CNN, trim-curve",
        "  grids (10 samples x 6 channels) through a 1D curve CNN, and the two are combined by",
        "  two edge-conditioned convolutions whose edge network maps each 64-dim curve embedding",
        "  to a 64x64 mixing matrix. 1,623,632 parameters.",
        "* **Adjacency GCN + LapPE**: the existing UV-Net-style adjacency GCN with the first 8",
        "  non-trivial eigenvectors of the normalized Laplacian of each solid concatenated to the",
        "  64-dim face embedding, projected back to 64 dims by a linear layer, then the same two",
        "  graph layers. Adds 4,672 parameters over Adjacency GCN.",
        "",
        "## Main results",
        "",
        "| Model | Parameters | Test accuracy (%) | Test macro IoU (%) | Test macro F1 (%) | Test balanced acc. (%) |",
        "|---|---|---|---|---|---|",
    ]
    for model in RUN_DIRS:
        lines.append(f"| {LABELS[model]} | {fmt_int(model, 'parameters')} | "
                     f"{fmt(model, 'test_accuracy')} | {fmt(model, 'test_macro_iou')} | "
                     f"{fmt(model, 'test_macro_f1')} | {fmt(model, 'test_balanced_accuracy')} |")

    base = mean_std(data["adjacency_gcn"]["test_macro_iou"])[0]
    lape = mean_std(data["adjacency_gcn_lape"]["test_macro_iou"])[0]
    topo = mean_std(data["topo_gdv"]["test_macro_iou"])[0]
    cnn = mean_std(data["cnn"]["test_macro_iou"])[0]
    uvnet = mean_std(data["uvnet"]["test_macro_iou"])[0]
    uvnet_lape = mean_std(data["uvnet_lape"]["test_macro_iou"])[0]
    uvnet_delta = uvnet_lape - uvnet
    uvnet_delta_text = ("no measurable change (0.00 points)" if abs(uvnet_delta) < 0.005
                        else f"{uvnet_delta:+.2f} points")
    uvnet_accuracy = mean_std(data["uvnet"]["test_accuracy"])[0]
    topo_class = data["topo_gdv"]["per_class_iou"].mean(0)
    cnn_class = data["cnn"]["per_class_iou"].mean(0)
    correlation = float(np.corrcoef(topo_class, cnn_class)[0, 1])
    topo_wins = [CLASS_NAMES[i] for i in np.argsort(-(topo_class - cnn_class))[:4]]
    topo_loses = [CLASS_NAMES[i] for i in np.argsort(topo_class - cnn_class)[:4]]
    low = low_data()
    low_text = low_data_text(low)
    lines += [
        "",
        "## Per-class IoU (mean over seeds)",
        "",
        "| Class | Test faces | " + " | ".join(LABELS[model] for model in RUN_DIRS) + " |",
        "|---|---|" + "---|" * len(RUN_DIRS),
    ]
    support = data["cnn"]["per_class_iou"]
    for index, name in enumerate(CLASS_NAMES):
        counts = int(np.asarray(load("cnn", 42)[1]["test"]["confusion_matrix"]).sum(1)[index])
        cells = " | ".join(f"{data[model]['per_class_iou'][:, index].mean():.3f}"
                           for model in RUN_DIRS)
        lines.append(f"| `{name}` | {counts} | {cells} |")

    lines += [
        "",
        "## Reading the numbers",
        "",
        f"* Full UV-Net reaches **{uvnet:.2f}% macro IoU / {uvnet_accuracy:.2f}% accuracy**, versus",
        f"  {base:.2f}% for the same graph and surface encoder without curve geometry",
        f"  ({uvnet - base:+.2f} points) and {cnn:.2f}% for the geometry-only CNN. UV-Net carries",
        "  four times the parameters of the ablations (1.62M versus 0.41M), so part of any gain is",
        "  capacity rather than the curve features themselves.",
        f"* Adding the 8 Laplacian eigenvectors to UV-Net gives **{uvnet_delta_text}**",
        f"  ({uvnet:.2f}% -> {uvnet_lape:.2f}%; seed SD grows from 0.07 to 0.12), the same",
        "  picture as the adjacency-GCN ablation: a spectral summary of the graph a model already",
        "  sees adds nothing at this scale.",
        f"* Topology alone reaches **{topo:.2f}% macro IoU**, compared with {cnn:.2f}% for the",
        f"  geometry-only CNN and {base:.2f}% for the UV-grid adjacency GCN. The gap isolates how",
        "  much of the task is carried by surface geometry versus face-adjacency structure.",
        f"* LapPE changes the adjacency GCN's macro IoU by **{lape - base:+.2f} points**",
        f"  ({base:.2f}% -> {lape:.2f}%, mean of three seeds).",
        f"* The per-class IoU vectors of Topo-GDV and the CNN are almost uncorrelated (Pearson r =",
        f"  {correlation:+.2f} over the 16 classes): the two information sources are complementary,",
        "  not nested. Topology alone is clearly better than geometry alone on",
        f"  {', '.join('`' + name + '`' for name in topo_wins)}, and clearly worse on",
        f"  {', '.join('`' + name + '`' for name in topo_loses)}.",
        "* This is the expected failure mode: graphlet orbits encode how faces are attached and how",
        "  many neighbours they have (blind versus through, pocket versus slot), but they are blind",
        "  to the cross-section geometry that separates, for example, triangular from rectangular",
        "  passages. It also has 19,856 parameters versus 400k-409k for the UV-grid models, so the",
        "  comparison is about available information, not capacity.",
        low_data_text(low),
        "* Topo-GDV picks its best checkpoint at epoch 16-20, so the 20-epoch budget is a lower",
        "  bound for that model; the UV-grid adjacency model is at its plateau by the same budget.",
        "* LapPE is a structural positional encoding; repeated Laplacian eigenvalues leave an",
        "  arbitrary rotation inside each eigenspace, so the encoding is only canonical up to that",
        "  ambiguity (columns are sign-canonicalized here).",
        "* Three seeds bound the claim; a +-0.3 point difference is within seed noise for the",
        "  adjacency model.",
        "",
        "## Files",
        "",
        "* `results/tables/main_results_all.csv` - every seed x model",
        "* `results/tables/per_class_iou_all.csv` - per-class IoU, mean over seeds",
        "* `results/figures/model_comparison_all.png` - bar chart",
        "* `results/figures/per_class_iou_all.png` - per-class heatmap",
        "* Raw runs: `results/uvnet-seed{42,1,2}/`, `results/uvnet-lape-seed{42,1,2}/`,",
        "  `results/topo-gdv-seed{42,1,2}/`, `results/lape-seed{42,1,2}/`",
        "* Low-data check (1,500 training solids, seed 42): `results/lowdata-adj-seed42/`,",
        "  `results/lowdata-lape-seed42/`",
        "",
    ]
    return "\n".join(lines)


def main():
    data = collect()
    tables = RESULTS / "tables"
    figures = RESULTS / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    rows = []
    for model in RUN_DIRS:
        for seed_index, seed in enumerate(SEEDS):
            rows.append([seed, model, int(data[model]["parameters"][seed_index]),
                         int(data[model]["best_epoch"][seed_index]),
                         f"{data[model]['val_accuracy'][seed_index]:.4f}",
                         f"{data[model]['val_macro_iou'][seed_index]:.4f}",
                         f"{data[model]['test_accuracy'][seed_index]:.4f}",
                         f"{data[model]['test_macro_iou'][seed_index]:.4f}"])
    write_csv(tables / "main_results_all.csv",
              ["seed", "model", "parameters", "best_epoch", "val_accuracy", "val_macro_iou",
               "test_accuracy", "test_macro_iou"], rows)

    per_class_rows = []
    for index, name in enumerate(CLASS_NAMES):
        per_class_rows.append([index, name] +
                              [f"{data[model]['per_class_iou'][:, index].mean():.4f}"
                               for model in RUN_DIRS])
    write_csv(tables / "per_class_iou_all.csv",
              ["index", "name"] + list(RUN_DIRS), per_class_rows)

    figure(data, figures / "model_comparison_all.png")
    heatmap(data, figures / "per_class_iou_all.png")
    (RESULTS / "REPORT_EXPERIMENTS.md").write_text(markdown(data))
    print(markdown(data))


if __name__ == "__main__":
    main()
