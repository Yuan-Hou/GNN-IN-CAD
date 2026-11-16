from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Measure simple geometric cues from the submitted surface grids."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import DATA_DIR, _load_sample, _load_splits
from data import corrected_label


def geometry_statistics():
    splits = _load_splits()
    rows = []
    for split, ids in splits.items():
        for sample_id in ids:
            sample = _load_sample(sample_id)
            for index, grid in enumerate(sample['graph'].ndata['x'].numpy()):
                valid = grid[..., 6] > 0.5
                points = grid[..., :3][valid].astype(float)
                normals = grid[..., 3:6][valid].astype(float)
                lengths = np.linalg.norm(normals, axis=1)
                normals = normals[np.isfinite(normals).all(axis=1) & (lengths > 1e-8)]
                if len(normals):
                    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                    mean = normals.mean(axis=0)
                    norm = np.linalg.norm(mean)
                    min_dot = float(np.min(normals @ (mean / norm))) if norm > 1e-8 else np.nan
                    flat = bool(np.isclose(min_dot, 1.0, atol=1e-6, rtol=0))
                    components = np.sort(np.abs(normals[0]))
                    aligned = bool(flat and np.all(components[:2] <= 1e-6) and np.isclose(components[2], 1, atol=1e-6, rtol=0))
                else:
                    min_dot = np.nan
                    flat, aligned = None, None
                spans = np.ptp(points, axis=0) if len(points) else np.full(3, np.nan)
                positive = spans[spans > 1e-6]
                rows.append(dict(model=sample_id, face_index=index, split=split,
                    label=corrected_label(sample['faces'][index]['segment']['name']),
                    shortest_span=positive.min() if len(positive) else np.nan,
                    longest_span=spans.max(), min_normal_dot=min_dot, flat=flat, axis_aligned=aligned))
    frame = pd.DataFrame(rows)
    return frame


def geometry_summary(frame=None):
    if frame is None:
        frame = geometry_statistics()
    defined = frame['min_normal_dot'].notna()
    flat = np.isclose(frame.loc[defined, 'min_normal_dot'], 1.0, atol=1e-6, rtol=0)
    print(f"Flat faces: {int(flat.sum()):,} / {len(frame):,}")
    print(f"Non-flat faces: {int((~flat).sum()):,}")
    print(f"Undefined normals: {int((~defined).sum()):,}")


def distribution_class_order(frame):
    """Use the shared EDA order even when a frame contains only a subset."""
    from data import CLASS_ORDER
    present = set(frame['label'])
    return [label for label in CLASS_ORDER if label in present]


def face_span_distributions(frame=None):
    if frame is None:
        frame = geometry_statistics()
    order = distribution_class_order(frame)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    for ax, column, title in zip(axes, ['shortest_span', 'longest_span'],
                                ['Shortest non-zero XYZ span', 'Longest XYZ span']):
        values = [frame.loc[frame.label == label, column].dropna().to_numpy() for label in order]
        ax.boxplot(values, vert=False, showfliers=False, widths=0.65,
                   medianprops={'color': '#c44e52'}, whis=(5, 95))
        ax.set_yticks(range(1, len(order) + 1), [label.replace('_', ' ') for label in order])
        ax.set(title=title, xlabel='Span (dataset coordinate units)', xlim=(0, 2.05))
        ax.grid(axis='x', alpha=0.2)
    axes[0].invert_yaxis()
    fig.suptitle('Face-span distributions by class')
    fig.tight_layout()
    plt.show()


def face_axis_alignment(frame=None):
    if frame is None:
        frame = geometry_statistics()
    fractions = frame.groupby('label')['axis_aligned'].mean().reindex(distribution_class_order(frame))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([label.replace('_', ' ') for label in fractions.index], fractions * 100,
            color='#4c78a8')
    for i, value in enumerate(fractions * 100):
        ax.text(value + 1, i, f'{value:.1f}%', va='center', fontsize=8)
    ax.set(xlim=(0, 114), xlabel='Axis-aligned faces (%)', title='Face-normal alignment with the X, Y, or Z axis')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2)
    fig.tight_layout()
    plt.show()
