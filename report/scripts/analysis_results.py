"""Small data-loading and presentation helpers for the result section."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from IPython.display import display
from model_evaluation import validation_class_counts

MODEL_SPECS = [
    ('1', 'Geometry', 'baseline'),
    ('1.5', 'Laplacian', 'laplacian_frozen'),
    ('1.5', 'WL (T=3)', 'wl3_frozen'),
    ('1.5', 'Graphlets (≤4)', 'graphlet4_frozen'),
    ('2a', 'GCN frozen', 'cheb2_frozen'),
    ('2b', 'GCN joint', 'cheb2_joint'),
    ('3a', 'Edge static', 'edge_static'),
    ('3b', 'Edge updated', 'edge_updated'),
]


def load_analysis_metrics():
    return pd.concat([validation_class_counts(run).assign(stage=stage, model=model)
                      for stage, model, run in MODEL_SPECS], ignore_index=True)


def plot_nonstock_iou(class_metrics):
    model_order = [model for _, model, _ in MODEL_SPECS]
    nonstock = class_metrics.loc[class_metrics.class_name != 'stock']
    groups = nonstock.groupby('model', sort=False)
    iou = groups.iou.mean().reindex(model_order)
    accuracy = (groups.true_positive.sum() / groups.support.sum()).reindex(model_order)
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = range(len(model_order))
    ax.axvspan(.5, 3.5, color='#718096', alpha=.08, zorder=0)
    for values, label, colour, marker, offset in [
        (accuracy, 'Overall accuracy', '#4279a8', 'o', 10),
        (iou, 'Mean IoU', '#d87529', 's', -16),
    ]:
        ax.plot(positions, values, marker=marker, linewidth=2, color=colour, label=label)
        for position, value in enumerate(values):
            ax.annotate(f'{value:.2%}', (position, value), xytext=(0, offset),
                        textcoords='offset points', ha='center', fontsize=8, color=colour)
    ax.set_xticks(list(positions), [f'{stage}\n{model}' for stage, model, _ in MODEL_SPECS],
                  rotation=15, ha='right')
    ax.set(ylabel='Validation score (stock excluded)', ylim=(.70, 1.04),
           title='Validation performance across the model ladder')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    plt.show()
    return fig


def show_stock_table(class_metrics):
    order = [model for _, model, _ in MODEL_SPECS]
    stock = class_metrics.loc[class_metrics.class_name == 'stock'].set_index('model').reindex(order)
    table = stock[['iou', 'false_positive', 'false_negative']].rename(columns={
        'iou': 'IoU', 'false_positive': 'False positives', 'false_negative': 'False negatives'})
    table.index.name = 'Model'
    display(table.style.format({'IoU': '{:.4%}', 'False positives': '{:,}', 'False negatives': '{:,}'}))
    return table


def show_class_metrics(class_metrics):
    from data import CLASS_ORDER
    from matplotlib.colors import LinearSegmentedColormap
    models = [model for _, model, _ in MODEL_SPECS]
    classes = [name for name in CLASS_ORDER if name != 'stock']
    values = class_metrics.copy()
    values['recall'] = values.true_positive / values.support.replace(0, float('nan'))
    tables = {}
    colours = LinearSegmentedColormap.from_list('class_scores', ['#fff1de', '#266b8e'])
    for metric, title in [('iou', 'IoU'), ('recall', 'Recall (per-class accuracy)')]:
        table = values.pivot(index='class_name', columns='model', values=metric).reindex(
            index=classes, columns=models)
        table.index = table.index.str.replace('_', ' ')
        table.index.name = 'Class'
        table.columns.name = 'Model'
        print(title)
        display(table.style.format('{:.2%}').background_gradient(
            cmap=colours, axis=None, vmin=table.min().min(), vmax=table.max().max()))
        tables[metric] = table
    return tables


def plot_group_iou(class_metrics):
    from data import CLASS_ORDER
    def groups(name):
        words = name.split('_')
        kind = words[-1].capitalize()
        if kind == 'Step':
            kind = 'Through step' if 'through' in words else 'Blind step'
        if name in {'slanted_through_step', '2sides_through_step'}:
            shape = 'Triangular'
        elif name == 'chamfer':
            shape = 'Chamfer'
        else:
            shape = {'rectangular': 'Rectangular', 'triangular': 'Triangular',
                     '6sides': 'Six-sided'}[words[0]]
        return kind, shape
    values = class_metrics.loc[class_metrics.class_name != 'stock'].copy()
    values[['Feature type', 'Profile shape']] = pd.DataFrame(
        [groups(name) for name in values.class_name], index=values.index)
    type_order = list(dict.fromkeys(groups(name)[0] for name in CLASS_ORDER if name != 'stock'))
    orders = {'Feature type': type_order,
              'Profile shape': ['Rectangular', 'Triangular', 'Six-sided', 'Chamfer']}
    models = [model for _, model, _ in MODEL_SPECS]
    labels = [f'{stage}\n{model}' for stage, model, _ in MODEL_SPECS]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    tables = {}
    for ax, (dimension, order) in zip(axes, orders.items()):
        table = values.groupby([dimension, 'model'], sort=False).iou.mean().unstack('model').reindex(
            index=order, columns=models)
        tables[dimension] = table
        for index, group in enumerate(order):
            ax.plot(range(len(models)), table.loc[group], marker='o', linewidth=1.8,
                    label=group, color=plt.get_cmap('tab10')(index))
        ax.axvspan(.5, 3.5, color='#718096', alpha=.08, zorder=0)
        ax.set_xticks(range(len(models)), labels, rotation=30, ha='right', fontsize=8)
        ax.set(title=f'Mean IoU by {dimension.lower()}', ylim=(.45, 1.025))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(title=dimension, fontsize=8, loc='lower right')
    axes[0].set_ylabel('Validation mean class IoU (stock excluded)')
    fig.tight_layout()
    plt.show()
    return tables


def plot_hardest_five_iou(class_metrics):
    from data import CLASS_ORDER
    baseline = class_metrics.loc[(class_metrics.model == 'Geometry') &
                                 (class_metrics.class_name != 'stock')]
    selected = baseline.sort_values(['iou', 'class_index']).head(5)
    classes = [name for name in CLASS_ORDER if name in set(selected.class_name)]
    models = [model for _, model, _ in MODEL_SPECS]
    table = class_metrics.pivot(index='class_name', columns='model', values='iou').reindex(
        index=classes, columns=models)
    ids = baseline.set_index('class_name').class_index
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.axvspan(.5, 3.5, color='#718096', alpha=.08, zorder=0)
    for name in classes:
        ax.plot(range(len(models)), table.loc[name], marker='o', linewidth=1.8,
                color=plt.get_cmap('tab20')(int(ids[name])), label=name.replace('_', ' '))
    ax.set_xticks(range(len(models)), [f'{stage}\n{model}' for stage, model, _ in MODEL_SPECS],
                  rotation=15, ha='right')
    ax.set(ylabel='Validation class IoU', ylim=(.45, 1.025),
           title='Five lowest-IoU classes in the geometry-only baseline')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    plt.show()
    return table


def show_uncertain_edge_faces():
    """Rank final edge-model faces by predictive entropy, independent of truth."""
    import json
    import torch
    from model_evaluation import load_cached_run, load_validation_predictions
    run = load_cached_run('edge_updated')
    saved = load_validation_predictions(run['root'] / f"epoch_{run['epoch']:03d}")
    names = json.loads((run['root'] / 'result.json').read_text())['class_names']
    probabilities = saved['probabilities'].double()
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(1)
    selected = torch.argsort(entropy, descending=True, stable=True)[:10]
    offsets = saved['sample_offsets']
    rows = []
    for index in selected.tolist():
        sample_index = int(torch.searchsorted(offsets[1:], torch.tensor(index), right=True))
        true = int(saved['labels'][index])
        prediction = int(probabilities[index].argmax())
        rows.append({'Solid': saved['sample_ids'][sample_index],
                     'Face': index - int(offsets[sample_index]),
                     'True class': names[str(true)].replace('_', ' '),
                     'Predicted class': names[str(prediction)].replace('_', ' '),
                     'Correct': prediction == true,
                     'Confidence': float(probabilities[index, prediction]),
                     'Entropy (nats)': float(entropy[index])})
    table = pd.DataFrame(rows, index=range(1, len(rows) + 1))
    table.index.name = 'Rank'
    display(table.style.format({'Confidence': '{:.1%}', 'Entropy (nats)': '{:.4f}'})
            .applymap(lambda value: 'background-color: #d8efdd; color: #14532d' if value
                      else 'background-color: #f7dddd; color: #7f1d1d', subset=['Correct']))
    return table


def plot_uncertain_edge_face(face):
    """Inspect one ranked face in its full solid, without a standalone export."""
    import json
    import numpy as np
    import plotly.graph_objects as go
    from data import _load_sample
    from eda_inspect_feature_labels import triangulate_uv_face, mesh_trace
    from model_evaluation import load_cached_run, load_validation_predictions
    run = load_cached_run('edge_updated')
    saved = load_validation_predictions(run['root'] / f"epoch_{run['epoch']:03d}")
    names = json.loads((run['root'] / 'result.json').read_text())['class_names']
    sample_id, face_index = str(face['Solid']), int(face['Face'])
    sample_index = saved['sample_ids'].index(sample_id)
    row = int(saved['sample_offsets'][sample_index]) + face_index
    true = int(saved['labels'][row])
    probabilities = saved['probabilities'][row].double()
    predicted = int(probabilities.argmax())
    sample = _load_sample(sample_id)
    if sample['faces'][face_index]['segment']['index'] != true:
        raise ValueError('Preview face does not match the cached validation label.')
    grids = sample['graph'].ndata['x'].numpy()
    figure = go.Figure()
    context = []
    for index, grid in enumerate(grids):
        triangles = triangulate_uv_face(grid)
        if not len(triangles):
            continue
        selected = index == face_index
        label = names[str(sample['faces'][index]['segment']['index'])].replace('_', ' ')
        hover = f"{'TARGET ' if selected else ''}Face {index}: {label}"
        if selected:
            hover += (f"<br>Predicted: {names[str(predicted)].replace('_', ' ')}"
                      f"<br>P(true): {float(probabilities[true]):.1%}"
                      f"<br>P(predicted): {float(probabilities[predicted]):.1%}")
        else:
            context.append(len(figure.data))
        figure.add_trace(mesh_trace(triangles, hover,
            '#ce353b' if selected else '#a5afb9', 1.0 if selected else .28))
    valid = grids[face_index,...,6] > .5
    normal = grids[face_index,...,3:6][valid].mean(axis=0)
    normal /= max(np.linalg.norm(normal),1e-12)
    tangent = np.cross(normal,np.eye(3)[np.argmin(np.abs(normal))])
    tangent /= max(np.linalg.norm(tangent),1e-12)
    eye = 2 * (normal + .5 * tangent)
    figure.update_layout(
        title=dict(text=f"Highest-entropy face in red — face {face_index}<br>"
              f"True: {names[str(true)].replace('_',' ')} ({float(probabilities[true]):.1%}) | "
              f"Predicted: {names[str(predicted)].replace('_',' ')} ({float(probabilities[predicted]):.1%})",
              font=dict(size=13)),
        height=600, margin=dict(l=0,r=0,b=0,t=85), showlegend=False,
        scene=dict(aspectmode='data',dragmode='orbit',xaxis=dict(visible=False),
                   yaxis=dict(visible=False),zaxis=dict(visible=False),
                   camera=dict(eye=dict(zip(['x','y','z'],eye)))),
        updatemenus=[dict(type='buttons',direction='right',x=0,y=1.04,
            buttons=[dict(label='Transparent context',method='restyle',args=[{'opacity':.28},context]),
                     dict(label='Solid context',method='restyle',args=[{'opacity':1.0},context])])])
    figure.show(renderer='plotly_mimetype',config=dict(scrollZoom=True,displaylogo=False))
    return figure


def style_transition_confusion(table, source_model, target_model, minimum_fixed):
    """Colour only rows with enough corrected faces to interpret percentages."""
    from html import escape
    count_columns = table.columns[:3].tolist()
    percent_columns = table.columns[3:].tolist()
    eligible = table.index[table['Fixed'] > minimum_fixed].tolist()
    sparse = table.index[table['Fixed'] <= minimum_fixed].tolist()
    style = (table.style
        .set_caption(f'{source_model} → {target_model}: previous predictions among corrected faces')
        .format_index(lambda label: (
            '<div class="rotated-header"><span>' + escape(label.replace('_', ' ')) + '</span></div>'
            if label in percent_columns else escape(label).replace(' ', '<br>')), axis='columns')
        .format_index(lambda label: escape(label.replace('_', ' ')), axis='index')
        .format({column: '{:,.0f}' for column in count_columns} |
                {column: '{:.1f}%' for column in percent_columns})
        .set_properties(subset=count_columns, **{'background-color':'#e8edf3',
                        'color':'#1f2937', 'font-weight':'bold'})
        .set_properties(**{'text-align':'center'})
        .set_table_styles([
            {'selector':'caption','props':[('caption-side','top'),('font-size','1.1em'),
                                          ('font-weight','bold'),('text-align','left')]},
            {'selector':'th','props':[('text-align','center')]},
            {'selector':'th.row_heading','props':[('text-align','left')]},
            {'selector':'th.col_heading','props':[('vertical-align','bottom')]},
            {'selector':'.rotated-header','props':[('height','180px'),('width','38px'),('position','relative')]},
            {'selector':'.rotated-header span','props':[('position','absolute'),('bottom','8px'),
                ('left','8px'),('white-space','nowrap'),('transform','rotate(-60deg)'),
                ('transform-origin','left bottom')]},
        ]))
    if eligible:
        style = style.background_gradient(cmap='YlOrRd',subset=pd.IndexSlice[eligible,percent_columns],
                                          axis=None,vmin=0,vmax=100)
    if sparse:
        style = style.set_properties(subset=pd.IndexSlice[sparse,percent_columns],
                                    **{'background-color':'#f4f4f4','color':'#777777'})
    # Matching true/predicted classes are outside these wrong-to-correct subsets.
    for label in table.index:
        if label in percent_columns:
            style = style.set_properties(subset=pd.IndexSlice[[label], [label]],
                                         **{'background-color': '#f4f4f4', 'color': '#777777'})
    return style


def transition_confusion_table(source_model, target_model):
    """Tabulate source mistakes corrected by the target; percentages sum by row."""
    import json
    import torch
    from data import ordered_class_indices
    from model_evaluation import load_cached_run, load_validation_predictions
    run_names = {label:name for _,label,name in MODEL_SPECS}
    saved = {}
    for model in dict.fromkeys(['Geometry',source_model,target_model]):
        run = load_cached_run(run_names[model])
        saved[model] = load_validation_predictions(run['root']/f"epoch_{run['epoch']:03d}")
        names = json.loads((run['root']/'result.json').read_text())['class_names']
    labels = saved['Geometry']['labels'].long()
    for result in saved.values():
        if not torch.equal(result['labels'],labels):
            raise ValueError('Confusion comparison requires aligned validation labels.')
    stock = next(int(index) for index,name in names.items() if name=='stock')
    predictions = {model:result['probabilities'].argmax(1) for model,result in saved.items()}
    baseline_errors = (labels!=stock) & (predictions['Geometry']!=labels)
    source_errors = (labels!=stock) & (predictions[source_model]!=labels)
    corrected = source_errors & (predictions[target_model]==labels)
    classes = len(names)
    fixed_confusion = torch.bincount(labels[corrected]*classes+predictions[source_model][corrected],
                                    minlength=classes**2).reshape(classes,classes)
    fixed = fixed_confusion.sum(1)
    rows = list(ordered_class_indices(names,include_stock=False))
    columns = list(ordered_class_indices(names))
    proportions = 100*fixed_confusion[rows][:,columns].double()/fixed[rows,None].clamp_min(1)
    table = pd.DataFrame(proportions.numpy(),index=[names[str(i)] for i in rows],
                         columns=[names[str(i)] for i in columns])
    table.insert(0,'Fixed',fixed[rows].numpy())
    table.insert(0,f'{source_model} errors',torch.bincount(labels[source_errors],minlength=classes)[rows].numpy())
    table.insert(0,'Geometry baseline errors',torch.bincount(labels[baseline_errors],minlength=classes)[rows].numpy())
    table.index.name='Correct class'
    return table


def show_transition_confusion(source_model, target_model, minimum_fixed):
    table=transition_confusion_table(source_model,target_model)
    display(style_transition_confusion(table,source_model,target_model,minimum_fixed))
    return table


def show_transition_solid_examples():
    """Display CE-selected whole-solid examples from aligned validation caches."""
    import json
    import numpy as np
    import torch
    from data import _load_sample
    from eda_inspect_feature_labels import triangulate_uv_face, mesh_trace
    from model_evaluation import load_cached_run, load_validation_predictions
    cached = {}
    class_names = None
    for _, model, run_name in MODEL_SPECS:
        run = load_cached_run(run_name)
        saved = load_validation_predictions(run['root'] / f"epoch_{run['epoch']:03d}")
        cached[model] = {**saved, 'predictions': saved['probabilities'].argmax(1)}
        names = {int(k):v for k,v in json.loads((run['root']/'result.json').read_text())['class_names'].items()}
        if class_names is not None and names != class_names:
            raise ValueError('Class names differ across caches.')
        class_names = names
    reference = cached['Geometry']
    split_ids = reference['sample_ids']
    reference_offsets = reference['sample_offsets']
    for saved in cached.values():
        if saved['sample_ids'] != split_ids or not torch.equal(saved['labels'], reference['labels']) or not torch.equal(saved['sample_offsets'], reference_offsets):
            raise ValueError('Examples require aligned validation caches.')
    stock_index = next(i for i,name in class_names.items() if name == 'stock')
    import math
    from plotly.subplots import make_subplots

    # High-source-mean-CE solids that become completely correct at each transition.
    solid_example_transitions = [
        dict(stage="Baseline correct", source_model="Geometry", target_model="Geometry",
             require_source_error=False),
        dict(stage="Graphlets correction", source_model="Geometry", target_model="Graphlets (≤4)",
             require_source_error=True),
        dict(stage="GCN joint correction", source_model="Graphlets (≤4)", target_model="GCN joint",
             require_source_error=True),
        dict(stage="Edge updated correction", source_model="GCN joint", target_model="Edge updated",
             require_source_error=True),
    ]

    def solid_transition_candidate(spec, sample_index, sample_id):
        source_result = cached[spec["source_model"]]
        target_result = cached[spec["target_model"]]
        start = int(reference_offsets[sample_index])
        stop = int(reference_offsets[sample_index + 1])
        labels = target_result["labels"][start:stop].long()
        source_predictions = source_result["predictions"][start:stop]
        target_predictions = target_result["predictions"][start:stop]
        source_wrong = source_predictions != labels
        target_wrong = target_predictions != labels
        if bool(target_wrong.any()):
            return None
        if spec["require_source_error"] and not bool(source_wrong.any()):
            return None
        if not spec["require_source_error"] and bool(source_wrong.any()):
            return None
        local_rows = torch.arange(start, stop)
        source_true_probability = source_result["probabilities"].double()[local_rows, labels]
        target_true_probability = target_result["probabilities"].double()[local_rows, labels]
        source_face_ce = -source_true_probability.clamp_min(1e-300).log()
        target_face_ce = -target_true_probability.clamp_min(1e-300).log()
        return dict(
            stage=spec["stage"], sample_index=sample_index, sample_id=sample_id,
            face_count=stop - start, source_model=spec["source_model"],
            source_error_count=int(source_wrong.sum()),
            source_mean_CE=float(source_face_ce.mean()),
            source_geometric_mean_true_probability_percent=(
                100 * math.exp(-float(source_face_ce.mean()))),
            target_model=spec["target_model"], target_error_count=int(target_wrong.sum()),
            target_mean_CE=float(target_face_ce.mean()),
            target_geometric_mean_true_probability_percent=(
                100 * math.exp(-float(target_face_ce.mean()))),
            mean_CE_drop=float(source_face_ce.mean() - target_face_ce.mean()),
        )

    solid_example_records = []
    for spec in solid_example_transitions:
        candidates = []
        for sample_index, sample_id in enumerate(split_ids):
            candidate = solid_transition_candidate(spec, sample_index, sample_id)
            if candidate is not None:
                candidates.append(candidate)
        assert candidates, spec["stage"]
        selected = max(
            candidates,
            key=lambda record: (record["mean_CE_drop"],
                                record["source_mean_CE"], -record["sample_index"]),
        )
        solid_example_records.append(selected)

    transition_solid_examples = pd.DataFrame(solid_example_records).set_index("stage")
    # Per-face probability details for every source-model misclassification.
    transition_error_face_records = []
    for record in solid_example_records:
        if record["source_error_count"] == 0:
            continue
        sample_index = int(record["sample_index"])
        start = int(reference_offsets[sample_index])
        stop = int(reference_offsets[sample_index + 1])
        source_result = cached[record["source_model"]]
        target_result = cached[record["target_model"]]
        labels = source_result["labels"][start:stop].long()
        source_predictions = source_result["predictions"][start:stop]
        wrong_local_faces = torch.nonzero(
            source_predictions != labels, as_tuple=False).flatten().tolist()
        for face_index in wrong_local_faces:
            row = start + face_index
            true_index = int(labels[face_index])
            before_prediction_index = int(source_predictions[face_index])
            after_prediction_index = int(target_result["predictions"][row])
            transition_error_face_records.append(dict(
                stage=record["stage"], sample_id=record["sample_id"], face_index=face_index,
                before_model=record["source_model"], after_model=record["target_model"],
                Pred=class_names[before_prediction_index],
                GT=class_names[true_index],
                before_pred_probability_percent=(
                    100 * float(source_result["probabilities"][row, before_prediction_index])),
                before_GT_probability_percent=(
                    100 * float(source_result["probabilities"][row, true_index])),
                after_pred_GT_probability_percent=(
                    100 * float(target_result["probabilities"][row, after_prediction_index])),
            ))

    transition_error_faces = pd.DataFrame(transition_error_face_records)
    transition_fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            (f"{record['stage']}<br>{record['source_model']} mean CE "
             f"{record['source_mean_CE']:.3f} "
             f"({record['source_geometric_mean_true_probability_percent']:.1f}%)"
             if record["source_model"] == record["target_model"] else
             f"{record['stage']}<br>{record['source_model']} mean CE "
             f"{record['source_mean_CE']:.3f} "
             f"({record['source_geometric_mean_true_probability_percent']:.1f}%) → "
             f"{record['target_model']} {record['target_mean_CE']:.3f} "
             f"({record['target_geometric_mean_true_probability_percent']:.1f}%)")
            for record in solid_example_records
        ],
        horizontal_spacing=.04, vertical_spacing=.08,
    )

    for panel_index, record in enumerate(solid_example_records):
        panel_row, panel_col = divmod(panel_index, 2)
        panel_row += 1
        panel_col += 1
        sample = _load_sample(record["sample_id"])
        grids = sample["graph"].ndata["x"].numpy()
        sample_index = int(record["sample_index"])
        start = int(reference_offsets[sample_index])
        source_result = cached[record["source_model"]]
        target_result = cached[record["target_model"]]
        for graph_face_index, grid in enumerate(grids):
            triangles = triangulate_uv_face(grid)
            if not len(triangles):
                continue
            row = start + graph_face_index
            true_index = int(target_result["labels"][row])
            source_prediction = int(source_result["predictions"][row])
            target_prediction = int(target_result["predictions"][row])
            source_wrong = source_prediction != true_index
            if source_wrong:
                face_color, face_opacity = "#e45756", 1.0
            elif true_index == stock_index:
                face_color, face_opacity = "#a5afb9", .16
            else:
                face_color, face_opacity = "#4c78a8", .62
            source_true_probability = float(source_result["probabilities"][row, true_index])
            source_predicted_probability = float(
                source_result["probabilities"][row, source_prediction])
            target_predicted_probability = float(
                target_result["probabilities"][row, target_prediction])
            if source_wrong:
                hover_name = (
                    f"Face {graph_face_index}<br>"
                    f"Pred: {class_names[source_prediction]}<br>"
                    f"GT: {class_names[true_index]}<br>"
                    f"before pred: {100 * source_predicted_probability:.1f}%<br>"
                    f"before GT: {100 * source_true_probability:.1f}%<br>"
                    f"after pred/GT: {100 * target_predicted_probability:.1f}%"
                )
            else:
                hover_name = (
                    f"Face {graph_face_index}: {class_names[true_index]}<br>"
                    f"{record['source_model']} → {class_names[source_prediction]}<br>"
                    f"{record['target_model']} → {class_names[target_prediction]}"
                )
            trace = mesh_trace(
                triangles, hover_name, face_color, face_opacity,
            )
            trace.showlegend = False
            transition_fig.add_trace(trace, row=panel_row, col=panel_col)

    scene_settings = dict(
        aspectmode="data",
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        camera=dict(eye=dict(x=1.45, y=1.45, z=1.1)),
    )
    transition_fig.update_layout(
        title="Whole solids corrected across the model ladder",
        height=1000, width=1400, margin=dict(l=10, r=10, t=105, b=10),
        scene=scene_settings, scene2=scene_settings,
        scene3=scene_settings, scene4=scene_settings,
    )
    transition_fig.show()

    display(transition_solid_examples.drop(columns="sample_index").style.format({
        "face_count": "{:,.0f}", "source_error_count": "{:,.0f}",
        "target_error_count": "{:,.0f}", "source_mean_CE": "{:.4f}",
        "target_mean_CE": "{:.4f}", "mean_CE_drop": "{:.4f}",
        "source_geometric_mean_true_probability_percent": "{:.1f}%",
        "target_geometric_mean_true_probability_percent": "{:.1f}%",
    }).set_caption("Largest whole-solid mean-CE improvement at each transition"))

    display(transition_error_faces.style.format({
        "before_pred_probability_percent": "{:.1f}%",
        "before_GT_probability_percent": "{:.1f}%",
        "after_pred_GT_probability_percent": "{:.1f}%",
    }).set_caption("Source-model misclassifications in the selected corrected solids"))

