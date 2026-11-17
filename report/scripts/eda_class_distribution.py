from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Reproduce class-distribution statistics and plots from submitted MFCAD labels."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import corrected_label, CLASS_ORDER

REPORT_DIR = Path(__file__).resolve().parents[1]

def load_class_statistics():
    data_dir = REPORT_DIR / "data" / "MFCADDataset"
    splits = json.loads((data_dir / "split.json").read_text())
    face_rows, model_rows = [], []
    for split, ids in splits.items():
        for sample_id in ids:
            annotations = json.loads((data_dir / "labels" / f"{sample_id}_ids.json").read_text())
            labels = [corrected_label(f["segment"]["name"]) for f in annotations["body"]["faces"]]
            counts = Counter(labels)
            model_rows.append(dict(model=sample_id, split=split, faces=len(labels),
                classes=len(set(labels) - {"stock"}), stock_faces=counts["stock"],
                stock_fraction=counts["stock"] / len(labels)))
            face_rows.extend(dict(model=sample_id, split=split, label=label) for label in labels)
    return splits, pd.DataFrame(face_rows), pd.DataFrame(model_rows)


def initialize_eda():
    global splits, faces, models, class_order
    splits, faces, models = load_class_statistics()
    class_order = list(CLASS_ORDER)

def histogram(values, title, xlabel, bins=30):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pd.Series(values).dropna(), bins=bins, edgecolor="white")
    ax.set(title=title, xlabel=xlabel, ylabel="Count")
    fig.tight_layout()
    plt.show()

def global_distribution():
    global class_counts, class_table, attributes, family_totals, shape_order, shape_colours, plot_order, positions, group_centres, plot_class_groups
    import textwrap
    from matplotlib.patches import Patch

    class_counts = faces["label"].value_counts().reindex(class_order)
    class_table = class_counts.to_frame("Faces")
    class_table["Percent"] = 100 * class_counts / class_counts.sum()

    # Separate feature type, profile shape, and through/blind extent.
    def class_attributes(label):
        words = label.split("_")
        feature_type = words[-1].capitalize()
        if label in {"slanted_through_step", "2sides_through_step"}:
            shape = "Triangular"
        elif label == "chamfer":
            shape = "Chamfer"
        elif words[0] == "rectangular":
            shape = "Rectangular"
        elif words[0] == "triangular":
            shape = "Triangular"
        elif words[0] == "6sides":
            shape = "Six-sided"
        else:
            shape = "Other / not specified"
        # Infer the extent implicit in these MFCAD feature names.
        if feature_type == "Pocket":
            extent = "Blind"
        elif feature_type == "Passage" or label == "chamfer":
            extent = "Through"
        else:
            extent = "Through" if "through" in words else "Blind" if "blind" in words else ""
        if feature_type == "Step":
            feature_type = f"{extent} step"
        # Keep the entire original class name; wrap only for readability.
        tick = label.replace("_", " ")
        tick = "\n".join(textwrap.wrap(tick, width=20))
        return feature_type, shape, extent, tick

    attributes = pd.DataFrame(
        [class_attributes(label) for label in class_counts.index],
        index=class_counts.index, columns=["Feature type", "Shape", "Extent", "Tick"]
    )
    class_table = attributes[["Feature type", "Shape", "Extent"]].join(class_table)
    display(class_table)

    shape_order = ["Rectangular", "Triangular", "Six-sided", "Chamfer", "Other / not specified"]

    family_order = list(dict.fromkeys(attributes.loc[label, "Feature type"] for label in CLASS_ORDER))
    family_totals = class_table.groupby("Feature type", sort=False)["Faces"].sum().reindex(family_order)
    plot_order, positions, group_centres = [], [], []
    position = 0
    for family in family_totals.index:
        labels = [label for label in CLASS_ORDER if attributes.loc[label, "Feature type"] == family]
        start = position
        for label in labels:
            plot_order.append(label)
            positions.append(position)
            position += 1
        group_centres.append((family, (start + position - 1) / 2))
        position += 0.5

    shape_colours = {"Rectangular": "#4c78a8", "Triangular": "#f58518",
                     "Six-sided": "#54a24b", "Chamfer": "#e45756",
                     "Other / not specified": "#9c9c9c"}
    bar_colours = [shape_colours[attributes.loc[label, "Shape"]] for label in plot_order]

    def plot_class_groups(ax, values, title, ylabel):
        ax.bar(positions, values.reindex(plot_order), width=0.9, color=bar_colours)
        ax.set_xticks(positions, [attributes.loc[label, "Tick"] for label in plot_order], rotation=60, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.margins(x=0.015)
        for family, centre in group_centres:
            ax.text(centre, -0.48, family, transform=ax.get_xaxis_transform(), ha="center", va="top", fontweight="bold")
        for j in range(1, len(group_centres)):
            previous = family_totals.index[j - 1]
            left = max(pos for label, pos in zip(plot_order, positions) if attributes.loc[label, "Feature type"] == previous)
            ax.axvline(left + 0.75, color="#dddddd", linewidth=0.8)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    plot_class_groups(ax, class_counts, "Global face-class distribution by feature type and shape", "Number of faces")
    ax.legend(handles=[Patch(facecolor=colour, label=shape) for shape, colour in shape_colours.items()], title="Shape", fontsize=9)
    fig.subplots_adjust(bottom=0.40, top=0.9, left=0.07, right=0.99)
    plt.show()
    # Show totals across both grouping dimensions as stacked bars.
    feature_shape_counts = class_table.pivot_table(
        index="Feature type", columns="Shape", values="Faces",
        aggfunc="sum", fill_value=0
    ).reindex(index=family_totals.index, columns=shape_order, fill_value=0)
    assert feature_shape_counts.to_numpy().sum() == class_counts.sum()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    feature_shape_counts.plot.bar(
        stacked=True, width=0.85, ax=axes[0],
        color=[shape_colours[shape] for shape in shape_order],
        title="Face totals by feature type", ylabel="Number of faces", rot=35
    )
    axes[0].legend(title="Shape", fontsize=8)

    feature_colours = plt.get_cmap("tab10").colors[:len(family_totals)]
    # Show stock as a separate bar, rather than as an unspecified shape.
    shape_feature_counts = feature_shape_counts.drop(index="Stock").T
    shape_feature_counts["Stock"] = 0
    shape_feature_counts.loc["Stock"] = 0
    shape_feature_counts.loc["Stock", "Stock"] = feature_shape_counts.loc["Stock"].sum()
    feature_palette = dict(zip(family_totals.index, feature_colours))
    shape_feature_counts.plot.bar(
        stacked=True, width=0.85, ax=axes[1],
        color=[feature_palette[family] for family in shape_feature_counts.columns],
        title="Face totals by shape, with stock separate", ylabel="Number of faces", rot=25
    )
    axes[1].legend(title="Feature type", fontsize=8)
    for ax, totals in zip(axes, [feature_shape_counts.sum(axis=1), shape_feature_counts.sum(axis=1)]):
        for x, total in enumerate(totals):
            ax.annotate(f"{total:,}", (x, total), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_ylim(0, totals.max() * 1.16)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    plt.show()


def classes_per_solid():
    display(models["classes"].describe().to_frame("Distinct machining-feature classes per solid").T)
    histogram(models["classes"], "Distinct machining-feature classes per solid", "Number of machining-feature classes", bins=np.arange(models["classes"].max() + 2) - 0.5)

def stock_distribution():
    stock_total = models["stock_faces"].sum()
    display(pd.DataFrame({"Faces": [stock_total, len(faces) - stock_total]}, index=["Stock", "Other classes"]).assign(Percent=lambda x: 100 * x["Faces"] / len(faces)))
    display(models["stock_fraction"].describe().to_frame("Stock fraction per model").T)
    histogram(models["stock_fraction"], "Stock proportion per model", "Fraction of faces labelled stock", bins=np.linspace(0, 1, 31))

def split_distribution():
    split_counts = pd.crosstab(faces["label"], faces["split"]).reindex(index=class_order, columns=list(splits), fill_value=0)
    split_percent = split_counts.div(split_counts.sum(axis=0), axis=1) * 100
    # Raw counts in the table; percentages in the chart.
    display(split_counts)

    # One facet per split, sharing the class axis and percentage scale.
    fig, axes = plt.subplots(len(splits), 1, figsize=(14, 10),
                             sharex=True, sharey=True, squeeze=False)
    for row, (ax, split) in enumerate(zip(axes[:, 0], splits)):
        plot_class_groups(ax, split_percent[split], "", "Faces within split (%)")
        ax.text(1.01, 0.5, split.capitalize(), transform=ax.transAxes,
                rotation=270, ha="left", va="center", fontweight="bold")
        if row < len(splits) - 1:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
            for text in list(ax.texts):
                if text.get_text() in family_totals.index:
                    text.remove()
        else:
            for text in ax.texts:
                if text.get_text() in family_totals.index:
                    text.set_position((text.get_position()[0], -0.82))
    fig.suptitle("Class proportions across splits", y=0.99)
    fig.legend(handles=[Patch(facecolor=colour, label=shape) for shape, colour in shape_colours.items()],
               loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=len(shape_colours), title="Shape")
    fig.subplots_adjust(bottom=0.27, top=0.89, hspace=0.12, left=0.07, right=0.95)
    plt.show()
