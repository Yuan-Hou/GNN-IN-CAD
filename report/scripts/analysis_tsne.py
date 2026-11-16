"""Read and plot cached embedding views; computation lives in analysis_tsne.ipynb."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from model_evaluation import REPORT_DIR, load_cached_run, load_validation_predictions
from data import ordered_class_indices

CACHE_DIR = REPORT_DIR / 'data' / 'tsne'
VIEWS = {
    'ladder': [('Geometry', 'baseline'), ('GCN joint', 'cheb2_joint'),
               ('Edge updated', 'edge_updated')],
    'shallow_input': [('Laplacian', 'laplacian_frozen'), ('WL (T=3)', 'wl3_frozen'),
                      ('Graphlets (≤4)', 'graphlet4_frozen')],
    'shallow_hidden': [('Laplacian', 'laplacian_frozen'), ('WL (T=3)', 'wl3_frozen'),
                       ('Graphlets (≤4)', 'graphlet4_frozen')],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_inputs():
    """Verify all six source caches and return one shared validation index."""
    runs = dict(pair for pairs in VIEWS.values() for pair in pairs)
    results = {}
    for label, name in runs.items():
        run = load_cached_run(name)
        final = run['root'] / f"epoch_{run['epoch']:03d}"
        results[label] = load_validation_predictions(final)
    reference = results['Geometry']
    for label, result in results.items():
        for key in ['labels', 'sample_offsets']:
            if not torch.equal(result[key], reference[key]):
                raise ValueError(f'Validation row order differs: {label}, {key}')
        if result['sample_ids'] != reference['sample_ids']:
            raise ValueError(f'Validation solid order differs: {label}')
    return results


def dependencies():
    """Bind coordinates to exact current model artifacts, face index and mapping."""
    paths = [REPORT_DIR / 'data/label_mapping.json', REPORT_DIR / 'data/class_order.json']
    paths.extend(sorted((REPORT_DIR / 'data/validation_index').iterdir()))
    for name in sorted({name for pairs in VIEWS.values() for _, name in pairs}):
        paths.append(REPORT_DIR / 'data/model_runs' / name / 'result.json')
    paths.extend(REPORT_DIR / 'scripts' / name for name in ['model_components.py', 'model_utils.py'])
    return {path.relative_to(REPORT_DIR).as_posix(): sha256(path) for path in paths if path.is_file()}


def load_tsne_cache():
    """Read coordinates only; stale/missing inputs never trigger expensive work."""
    manifest_path = CACHE_DIR / 'metadata.json'
    if not manifest_path.exists():
        raise FileNotFoundError('t-SNE cache missing. Use working/analysis_tsne.ipynb to rebuild explicitly.')
    metadata = json.loads(manifest_path.read_text())
    if metadata['dependencies'] != dependencies():
        raise ValueError('t-SNE source artifacts changed. Recompute explicitly in working/analysis_tsne.ipynb.')
    if sha256(CACHE_DIR / 'coordinates.npz') != metadata['coordinate_sha256']:
        raise ValueError('t-SNE coordinate checksum mismatch.')
    results = validation_inputs()
    with np.load(CACHE_DIR / 'coordinates.npz', allow_pickle=False) as arrays:
        cache = {key: arrays[key].copy() for key in arrays.files}
    rows = cache['source_rows']
    reference = results['Geometry']['labels'].numpy()
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError('Invalid t-SNE source rows.')
    if len(rows) != metadata['selected_faces'] or np.any(np.diff(rows) <= 0):
        raise ValueError('t-SNE source rows must be unique and ordered.')
    if len(rows) == 0 or rows[0] < 0 or rows[-1] >= len(reference):
        raise ValueError('t-SNE source row outside validation split.')
    if not np.array_equal(cache['labels'], reference[rows]):
        raise ValueError('t-SNE labels differ from validation index.')
    union = np.logical_or.reduce([
        result['probabilities'].argmax(1).numpy() != reference
        for label, result in results.items() if label in dict(VIEWS['ladder'])])
    if not np.array_equal(cache['union_error'], union[rows]) or int(union[rows].sum()) != int(union.sum()):
        raise ValueError('t-SNE sample does not include all main-ladder errors.')
    largest = int(np.bincount(reference[union], minlength=16).max())
    for view, pairs in VIEWS.items():
        for i, (label, _) in enumerate(pairs):
            coordinates = cache[f'{view}_{i}']
            if coordinates.shape != (len(rows), 2) or not np.isfinite(coordinates).all():
                raise ValueError(f'Invalid t-SNE coordinates: {view}, {label}')
    return dict(arrays=cache, metadata=metadata, results=results, largest_class_errors=largest)


def plot_tsne(view="pre_classifier", cache=None):
    """Show all six models in fixed positions, changing only the shallow inputs."""
    if view not in {"pre_classifier", "first_layer"}:
        raise ValueError("Choose pre_classifier or first_layer.")
    cache = load_tsne_cache() if cache is None else cache
    arrays = cache['arrays']
    names = json.loads((REPORT_DIR / 'data/model_runs/baseline/result.json').read_text())['class_names']
    shallow = 'shallow_input' if view == 'pre_classifier' else 'shallow_hidden'
    panels = [('Geometry', 'ladder_0'), ('Laplacian', f'{shallow}_0'),
              ('WL (T=3)', f'{shallow}_1'), ('Graphlets (≤4)', f'{shallow}_2'),
              ('GCN joint', 'ladder_1'), ('Edge updated', 'ladder_2')]
    title = ('Pre-classifier representations' if view == 'pre_classifier'
             else 'Shallow first-layer representations in the same comparison')
    fig, axes = plt.subplots(3, 2, figsize=(13, 16))
    cmap = plt.get_cmap('tab20')
    for ax, (label, key) in zip(axes.flat, panels):
        coordinates = arrays[key]
        for index in ordered_class_indices(names):
            mask = arrays['labels'] == index
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=5, alpha=.58,
                       linewidths=0, rasterized=True, color=cmap(index),
                       label=names[str(index)].replace('_', ' '))
        result = cache['results'][label]
        errors = int((result['probabilities'].argmax(1).numpy()[arrays['source_rows']]
                      != arrays['labels']).sum())
        ax.set_title(f'{label} ({errors:,} errors shown)', fontsize=11)
        ax.set(xticks=[], yticks=[], xlabel='t-SNE 1', ylabel='t-SNE 2')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    # Legend fills columns; arrange entries to read in EDA order across rows.
    permutation = np.arange(len(labels)).reshape(-1, 4).T.ravel()
    handles = [handles[i] for i in permutation]
    labels = [labels[i] for i in permutation]
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               markerscale=2, fontsize=9, bbox_to_anchor=(.5, .01))
    fig.suptitle(title, fontsize=15, y=.99)
    fig.tight_layout(rect=(0, .105, 1, .975), h_pad=2, w_pad=1)
    plt.show()
    return fig
