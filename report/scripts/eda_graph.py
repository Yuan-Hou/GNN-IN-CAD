from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eda_geometry import distribution_class_order

"""Graph-size and non-stock node-degree summaries for the report."""
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from data import DATA_DIR, _load_sample, _load_splits
from data import corrected_label


def graph_statistics():
    splits = _load_splits()
    models, faces = [], []
    for split, ids in splits.items():
        for sid in ids:
            sample = _load_sample(sid)
            graph = sample['graph']
            src, dst = (a.numpy() for a in graph.edges())
            directed = Counter(zip(src.tolist(), dst.tolist()))
            if not all(u != v and n == directed[(v, u)] for (u, v), n in directed.items()):
                raise ValueError(f'{sid}: expected reciprocal edges without self-loops')
            boundaries = int(np.sum(src < dst))
            degrees = graph.out_degrees().numpy()
            assert int(degrees.sum()) == 2 * boundaries
            models.append(dict(model=sid, split=split, faces=graph.num_nodes(), edges=boundaries))
            for i, face in enumerate(sample['faces']):
                faces.append(dict(model=sid, split=split, face_index=i,
                                  label=corrected_label(face['segment']['name']), degree=int(degrees[i])))
    return pd.DataFrame(models), pd.DataFrame(faces)


def graph_sizes(statistics=None):
    models, _ = graph_statistics() if statistics is None else statistics
    display(models[['faces', 'edges']].agg(['min', 'median', 'mean', 'max']).T)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column in zip(axes, ['faces', 'edges']):
        values = models[column]
        ax.hist(values, bins=np.arange(values.min(), values.max() + 2) - 0.5,
                color='#4c78a8', edgecolor='white')
        ax.set(title=f'{column.capitalize()} per solid', xlabel=f'Number of {column}', ylabel='Number of solids')
    fig.tight_layout()
    plt.show()


def nonstock_node_degrees(statistics=None):
    _, faces = graph_statistics() if statistics is None else statistics
    nonstock = faces.loc[faces.label != 'stock', 'degree']
    display(nonstock.agg(['min', 'median', 'mean', 'max']).to_frame('Non-stock face degree').T)
    counts = nonstock.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color='#4c78a8')
    ax.set(title='Node degree of machining-feature faces (stock excluded)',
           xlabel='Incident B-rep boundary edges', ylabel='Number of faces')
    ax.set_xticks(counts.index)
    fig.tight_layout()
    plt.show()
    for name, selected in [('Non-stock', faces.label != 'stock'), ('Stock', faces.label == 'stock')]:
        values = faces.loc[selected, 'degree']
        print(f'{name}: {int((values == 4).sum()):,} / {len(values):,} faces have degree 4 ({100 * (values == 4).mean():.1f}%).')


def topology_statistics():
    """Compute multiplicity, diameter, and per-face hop coverage from the dataset."""
    import networkx as nx
    diameters, parallel, multiplicities, coverages = [], [], [], []
    for ids in _load_splits().values():
        for sid in ids:
            graph = _load_sample(sid)['graph']
            src, dst = (a.numpy() for a in graph.edges())
            pairs = Counter((int(u), int(v)) for u, v in zip(src, dst) if u < v)
            multiplicities.extend(pairs.values())
            parallel.append(any(n >= 2 for n in pairs.values()))
            topology = nx.MultiGraph()
            topology.add_nodes_from(range(graph.num_nodes()))
            topology.add_edges_from((int(u), int(v)) for u, v in zip(src, dst) if u < v)
            if not nx.is_connected(topology):
                raise ValueError(f'{sid}: disconnected graph')
            rows = []
            diameter = 0
            for _, distances in nx.all_pairs_shortest_path_length(topology):
                distances = np.array(list(distances.values()))
                diameter = max(diameter, int(distances.max()))
                counts = np.bincount(distances, minlength=graph.num_nodes())
                counts[0] = 0
                if graph.num_nodes() > 1:
                    rows.append(100 * np.cumsum(counts) / (graph.num_nodes() - 1))
            diameters.append(diameter)
            coverages.append(np.asarray(rows, dtype=np.float32)[:, :diameter + 1])
    max_hops = max(diameters)
    reachability = np.concatenate([np.pad(a, ((0, 0), (0, max_hops + 1 - a.shape[1])),
                                          constant_values=100) for a in coverages], axis=0)[:, 1:]
    result = dict(diameters=np.asarray(diameters), parallel=np.asarray(parallel),
                  multiplicities=np.asarray(multiplicities), reachability=reachability)
    return result


def node_degree_by_class(statistics=None):
    _, faces = graph_statistics() if statistics is None else statistics
    order = [label for label in distribution_class_order(faces) if label != 'stock']
    subset = faces.loc[faces.label != 'stock']
    counts = pd.crosstab(subset.label, subset.degree).reindex(index=order, fill_value=0)
    percent = counts.reindex(columns=[3, 4, 5, 6, 8], fill_value=0).div(counts.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(percent, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(5), ['3', '4', '5', '6', '8'])
    ax.set_yticks(range(len(order)), [label.replace('_', ' ') for label in order])
    ax.set(xlabel='Incident B-rep boundary edges', title='Node degree by class (stock excluded)')
    for i in range(len(order)):
        for j in range(5):
            value = percent.iloc[i, j]
            ax.text(j, i, f'{value:.1f}', ha='center', va='center', color='white' if value > 60 else 'black', fontsize=8)
    fig.colorbar(im, ax=ax, label='Faces within class (%)')
    fig.tight_layout()
    plt.show()


def maximum_node_degrees(statistics=None):
    _, faces = graph_statistics() if statistics is None else statistics
    values = faces.loc[faces.label != 'stock'].groupby('model').degree.max()
    display(values.describe().to_frame('Maximum non-stock degree per solid').T)
    _integer_histogram(values, 'Maximum non-stock node degree per solid', 'Incident B-rep boundary edges')


def _integer_histogram(values, title, xlabel):
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(values, bins=np.arange(values.min(), values.max() + 2) - 0.5,
            color='#4c78a8', edgecolor='white')
    ax.set(title=title, xlabel=xlabel, ylabel='Number of solids')
    fig.tight_layout()
    plt.show()


def edge_multiplicity(stats=None):
    if stats is None:
        stats = topology_statistics()
    display(pd.Series(stats['multiplicities']).value_counts().sort_index().to_frame('Face pairs').rename_axis('Boundaries per face pair'))
    print(f"Solids with parallel boundaries: {int(stats['parallel'].sum()):,} / {len(stats['parallel']):,} ({100 * stats['parallel'].mean():.1f}%).")


def graph_diameters(stats=None):
    values = (topology_statistics() if stats is None else stats)['diameters']
    display(pd.Series(values).describe().to_frame('Shortest-path diameter').T)
    _integer_histogram(values, 'Shortest-path diameter per solid', 'Maximum shortest-path distance (hops)')


def face_reachability(stats=None):
    values = (topology_statistics() if stats is None else stats)['reachability']
    assert np.allclose(values[:, -1], 100)
    display(pd.DataFrame(values, columns=np.arange(1, values.shape[1] + 1)).describe().T.rename_axis('Hops'))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([values[:, i] for i in range(values.shape[1])],
               labels=np.arange(1, values.shape[1] + 1), showfliers=False,
               patch_artist=True, boxprops={'facecolor': '#9ecae1'},
               medianprops={'color': '#b22222'})
    ax.set(xlabel='Number of hops', ylabel='Other faces reachable (%)',
           title='Face reachability within N hops', ylim=(0, 105))
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    plt.show()


def edge_multiplicity_by_stock():
    """Compare boundary multiplicity by endpoint label types."""
    rows = []
    for ids in _load_splits().values():
        for sid in ids:
            sample = _load_sample(sid)
            stock = [f['segment']['name'] == 'stock' for f in sample['faces']]
            src, dst = (a.numpy() for a in sample['graph'].edges())
            pairs = Counter((int(u), int(v)) for u, v in zip(src, dst) if u < v)
            for (u, v), multiplicity in pairs.items():
                category = 'Stock–stock' if stock[u] and stock[v] else 'Stock–feature' if stock[u] or stock[v] else 'Feature–feature'
                rows.append(dict(model=sid, category=category, multiplicity=multiplicity))
    pairs = pd.DataFrame(rows)
    order = ['Stock–stock', 'Stock–feature', 'Feature–feature']
    counts = pd.crosstab(pairs.category, pairs.multiplicity).reindex(order, fill_value=0)
    counts.columns = [f'{m} boundaries' for m in counts.columns]
    counts['Total face pairs'] = counts.sum(axis=1)
    counts['Multiple boundaries (%)'] = 100 * (counts['Total face pairs'] - counts['1 boundaries']) / counts['Total face pairs']
    display(counts)
    totals = pairs.groupby(['model', 'category']).multiplicity.max().unstack().reindex(columns=order)
    summary = pd.DataFrame({'Solids with this pair type': totals.notna().sum(),
                            'Solids with multiple boundaries': (totals >= 2).sum()})
    summary['Multiple boundaries (%)'] = 100 * summary.iloc[:, 1] / summary.iloc[:, 0]
    display(summary)
