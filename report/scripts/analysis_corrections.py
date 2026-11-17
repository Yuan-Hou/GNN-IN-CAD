"""Face properties and plots for fixed wrong-to-correct model transitions."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from data import _load_sample
from model_evaluation import load_cached_run, load_validation_predictions
from analysis_results import MODEL_SPECS

TRANSITIONS = [
    ('Geometry', 'Laplacian', 'Laplacian'),
    ('Geometry', 'WL (T=3)', 'WL (T=3)'),
    ('Geometry', 'Graphlets (≤4)', 'Graphlets'),
    ('Graphlets (≤4)', 'GCN frozen', 'GCN frozen'),
    ('Graphlets (≤4)', 'GCN joint', 'GCN joint'),
    ('GCN joint', 'Edge static', 'Edge static'),
    ('GCN joint', 'Edge updated', 'Edge updated'),
]


def correction_properties():
    results = {}
    for _, model, name in MODEL_SPECS:
        run = load_cached_run(name)
        results[model] = load_validation_predictions(run['root'] / f"epoch_{run['epoch']:03d}")
    reference = results['Geometry']
    labels = reference['labels'].numpy()
    for result in results.values():
        if result['sample_ids'] != reference['sample_ids'] or not np.array_equal(result['labels'].numpy(), labels):
            raise ValueError('Correction analysis requires aligned validation face rows.')
    rows = []
    for sample_index, sample_id in enumerate(reference['sample_ids']):
        sample = _load_sample(sample_id)
        graph = sample['graph']
        grids = graph.ndata['x'].numpy()
        offset, end = reference['sample_offsets'][sample_index:sample_index+2].tolist()
        if len(grids) != end-offset:
            raise ValueError('Graph faces differ from the shared validation index.')
        src,dst = (values.numpy() for values in graph.edges())
        neighbours = [set() for _ in range(len(grids))]
        for u,v in zip(src,dst):
            if u != v:neighbours[int(u)].add(int(v))
        degrees = graph.out_degrees().numpy()
        for face_index,grid in enumerate(grids):
            label = int(sample['faces'][face_index]['segment']['index'])
            if label != int(labels[offset+face_index]):
                raise ValueError('Face labels differ from the cached predictions.')
            if label == 15:continue
            valid = grid[...,6] > .5
            points = grid[...,:3][valid].astype(float)
            spans = np.ptp(points,axis=0) if len(points) else np.full(3,np.nan)
            positive = spans[spans > 1e-6]
            shortest = float(positive.min()) if len(positive) else np.nan
            longest = float(spans.max())
            normals = grid[...,3:6][valid].astype(float)
            lengths = np.linalg.norm(normals,axis=1)
            normals = normals[np.isfinite(normals).all(axis=1) & (lengths > 1e-8)]
            if len(normals):
                normals /= np.linalg.norm(normals,axis=1,keepdims=True)
                mean = normals.mean(0);mean /= max(np.linalg.norm(mean),1e-12)
                components = np.sort(np.abs(mean))
                aligned = bool(np.all(components[:2] <= 1e-6) and np.isclose(components[2],1,atol=1e-6,rtol=0))
            else:raise ValueError('Face has no valid normal.')
            reached = {face_index};frontier = {face_index};coverage = []
            for _ in range(3):
                frontier = set().union(*(neighbours[node] for node in frontier)) - reached if frontier else set()
                reached.update(frontier);coverage.append(len(reached)-1)
            rows.append(dict(row=offset+face_index, class_index=label, sample_id=sample_id,
                face_index=face_index, shortest_span=shortest, span_product=longest*shortest,
                span_ratio=longest/shortest if shortest>0 else np.nan,
                axis_aligned=aligned, non_axis_aligned=not aligned,
                degree=int(degrees[face_index]), non_four_degree=bool(degrees[face_index]!=4),
                two_hop=coverage[1],three_hop=coverage[2]))
    frame = pd.DataFrame(rows)
    indices = frame.row.to_numpy()
    correct = {model:(saved['probabilities'].argmax(1).numpy()[indices] == labels[indices])
               for model,saved in results.items()}
    transitions = []
    for source,target,label in TRANSITIONS:
        pool = ~correct[source]
        fixed = pool & correct[target]
        transitions.append(dict(source=source,target=target,label=label,pool=pool,fixed=fixed))
    return dict(faces=frame,correct=correct,transitions=transitions)


def correction_summary(analysis):
    rows=[]
    for item in analysis['transitions']:
        rows.append({'Source':item['source'],'Target':item['target'],
                     'Source errors':int(item['pool'].sum()),'Corrected':int(item['fixed'].sum()),
                     'Correction rate':float(item['fixed'].sum()/item['pool'].sum())})
    return pd.DataFrame(rows)


def _boxplot(ax,groups,labels,colours):
    boxes=ax.boxplot(groups,labels=labels,patch_artist=True,showfliers=False,
                     medianprops=dict(color='#222222',linewidth=1.4))
    for patch,colour in zip(boxes['boxes'],colours):patch.set_facecolor(colour)
    medians = [float(np.median(values)) if len(values) else np.nan for values in groups]
    ax.plot(range(1, len(groups) + 1), medians, color='#8b3038',
            marker='o', markersize=3.5, linewidth=1.5, zorder=4)
    for i,values in enumerate(groups,1):
        if len(values)<=50:
            ax.scatter(i+np.linspace(-.08,.08,len(values)),values,s=12,color='#555555',alpha=.7,zorder=3)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)


def plot_correction_distributions(analysis,metrics):
    frame=analysis['faces'];transitions=analysis['transitions'];correct=analysis['correct']['Geometry']
    fig,axes=plt.subplots(len(metrics),2,figsize=(16,4.4*len(metrics)),squeeze=False,
                          gridspec_kw={'width_ratios':[1,2.6]})
    colours=['#d8c09e']*3+['#9fc5e8']*2+['#e7a1a1']*2
    for (left,right),(metric,title,ylabel,logarithmic) in zip(axes,metrics):
        values=frame[metric].to_numpy();finite=np.isfinite(values)
        groups=[values[correct&finite],values[(~correct)&finite]]
        _boxplot(left,groups,[f'Correct\n(n={len(groups[0]):,})',f'Wrong\n(n={len(groups[1]):,})'],['#8ac49b','#e7a1a1'])
        fixed=[values[item['fixed']&finite] for item in transitions]
        labels=[f"{item['label']}\n(n={len(group):,})" for item,group in zip(transitions,fixed)]
        _boxplot(right,fixed,labels,colours)
        left.set_title(f'{title}: geometry-only predictions')
        right.set_title(f'{title}: faces corrected at each transition')
        for ax in [left,right]:
            ax.set_ylabel(ylabel)
            if logarithmic:ax.set_yscale('log')
        # Keep the same scale for the reference and corrected-face distributions.
        low=min(left.get_ylim()[0],right.get_ylim()[0]);high=max(left.get_ylim()[1],right.get_ylim()[1])
        left.set_ylim(low,high);right.set_ylim(low,high)
        plt.setp(right.get_xticklabels(),rotation=18,ha='right',fontsize=8)
    fig.tight_layout();plt.show()
    return fig


def plot_binary_correction_property(analysis,column,title):
    frame=analysis['faces'];transitions=analysis['transitions'];values=frame[column].to_numpy(dtype=bool)
    correct=analysis['correct']['Geometry']
    fig,axes=plt.subplots(1,2,figsize=(14,4.5),gridspec_kw={'width_ratios':[1,2.6]},sharey=True)
    axes[0].bar([0,1],[values[correct].mean(),values[~correct].mean()],color=['#8ac49b','#e7a1a1'])
    axes[0].set_xticks([0,1],['Geometry correct','Geometry wrong'],rotation=15,ha='right')
    x=np.arange(len(transitions))
    fixed=np.array([values[item['fixed']].mean() for item in transitions])
    axes[1].bar(x,fixed,width=.65,color=['#d8c09e']*3+['#9fc5e8']*2+['#e7a1a1']*2)
    axes[1].plot(x,fixed,color='#8b3038',marker='o',markersize=4,linewidth=1.5)
    axes[1].set_xticks(x,[item['label'] for item in transitions],rotation=18,ha='right',fontsize=8)
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylim(0,.5 if column == 'non_four_degree' else 1.04);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Faces in subset (%)');fig.suptitle(title);fig.tight_layout();plt.show()
    return fig


def describe_corrections(analysis):
    frame=analysis['faces'];rows=[]
    masks={'Geometry correct':analysis['correct']['Geometry'],'Geometry wrong':~analysis['correct']['Geometry']}
    masks.update({item['label']:item['fixed'] for item in analysis['transitions']})
    for name,mask in masks.items():
        subset=frame.loc[mask]
        rows.append(dict(group=name,count=len(subset),
            **{metric+'_median':float(subset[metric].median()) for metric in
               ['shortest_span','span_product','span_ratio','two_hop','three_hop']},
            non_axis_aligned_fraction=float(subset.non_axis_aligned.mean()),
            non_four_degree_fraction=float(subset.non_four_degree.mean())))
    return pd.DataFrame(rows).set_index('group')
