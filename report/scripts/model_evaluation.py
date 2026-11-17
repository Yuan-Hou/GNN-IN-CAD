from __future__ import annotations

"""Map flat prediction rows to solids and zero-based graph face indices."""

import json
from pathlib import Path

import torch


def attach_prediction_index(results, sample_ids, split, data_dir=None):
    """Validate label order and attach a compact, explicit row index.

    Call only for evaluation in the given sample order (never shuffled).
    ``sample_ids[sample_indices[row]]`` identifies the solid; ``face_indices``
    identifies its graph node/annotation position, not an OCC STEP face ID.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[1] / "data/MFCADDataset"
    data_dir = Path(data_dir)
    sample_ids = list(sample_ids)
    if split not in ("train", "validation", "test"):
        raise ValueError("Provide the dataset split for indexed predictions.")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Sample IDs must be unique within the evaluation split.")
    labels, counts = [], []
    for sample_id in sample_ids:
        if Path(sample_id).name != sample_id:
            raise ValueError("Sample IDs must be simple filenames.")
        path = data_dir / "labels" / f"{sample_id}_ids.json"
        faces = json.loads(path.read_text())["body"]["faces"]
        counts.append(len(faces))
        labels.extend(face["segment"]["index"] for face in faces)
    expected = torch.tensor(labels, dtype=torch.long)
    if not torch.equal(results["labels"].cpu().long(), expected):
        raise ValueError("Prediction label order does not match the supplied solid/face order.")
    if results["probabilities"].shape[0] != len(labels) or results["predictions"].shape != expected.shape:
        raise ValueError("Prediction rows do not match the supplied index.")
    counts = torch.tensor(counts, dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    sample_indices = torch.repeat_interleave(torch.arange(len(sample_ids)), counts)
    face_indices = torch.arange(len(labels)) - offsets[sample_indices]
    return {**results, "index_format_version": 1, "split": split,
            "sample_ids": sample_ids, "sample_indices": sample_indices,
            "face_indices": face_indices, "sample_offsets": offsets}


"""Multiclass evaluation and prediction storage."""

from pathlib import Path

import torch
from torch import nn


def evaluate(model, evaluation_set, output_path=None, device=None, *,
             sample_ids=None, split=None, data_dir=None):
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
    Provide ordered ``sample_ids`` and ``split`` to attach an explicit
    solid/face index, validated against dataset annotation labels. This
    requires a non-shuffled evaluation loader in the supplied sample order.
    """
    if device is None:
        device = next(model.parameters(), torch.empty(0)).device
    if sample_ids is not None:
        from torch.utils.data import DataLoader, SequentialSampler
        if isinstance(evaluation_set, DataLoader) and not isinstance(evaluation_set.sampler, SequentialSampler):
            raise ValueError("Indexed prediction export requires a sequential evaluation loader.")

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
    if sample_ids is not None:
        results = attach_prediction_index(results, sample_ids, split, data_dir)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, output_path)
    return results


"""Read and display cached model runs; never import or execute training code."""
from pathlib import Path
import hashlib
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

REPORT_DIR = Path(__file__).resolve().parents[1]


def load_cached_run(model_name):
    if Path(model_name).name != model_name:
        raise ValueError('Use a simple model directory name.')
    root = REPORT_DIR / 'data' / 'model_runs' / model_name
    manifest_path = root / 'result.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Cached results missing: {root}. No training will be started.')
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest['files'].items():
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Cached artifact missing or changed: {path}')
    final = root / f"epoch_{manifest['final_epoch']:03d}"
    checkpoint = (torch.load(final / 'checkpoint.pt', map_location='cpu', weights_only=True)
                  if (final / 'checkpoint.pt').exists()
                  else json.loads((final / 'embedding_metadata.json').read_text()))
    history = pd.DataFrame(checkpoint['history'])
    summary, _ = read_cached_metrics(root, manifest['final_epoch'], manifest['class_names'])
    return dict(name=model_name, root=root, epoch=manifest['final_epoch'],
                config=checkpoint.get('config', {}), history=history,
                summary=summary)


def show_training_setup(run):
    config = run['config']
    settings = {'Epochs': run['epoch']}
    for name, title in [('batch_size', 'Batch size (solids)'), ('learning_rate', 'Learning rate'),
                        ('seed', 'Seed'), ('embedding_dim', 'Face embedding dimension'), ('hidden_dim', 'Classifier hidden dimension')]:
        if name in config:
            settings[title] = config[name]
    display(pd.DataFrame({'Setting': list(settings), 'Value': [str(value) for value in settings.values()]}).set_index('Setting'))


def show_learning_curves(run):
    history = run['history']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ('loss', 'accuracy')):
        for split in ('train', 'validation'):
            ax.plot(history.epoch, history[f'{split}_{metric}'], label=split.capitalize())
        ax.set(xlabel='Epoch', ylabel='Cross-entropy loss' if metric == 'loss' else 'Face accuracy',
               title='Loss' if metric == 'loss' else 'Accuracy')
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    plt.show()


def show_performance(run):
    display(run['summary'].style.format({'faces': '{:,}', 'accuracy': '{:.2%}', 'mean_iou': '{:.2%}', 'loss': '{:.4f}'}))


def compare_cached_runs(model_names, *, show_curves=True):
    runs = {label: load_cached_run(name) for label, name in model_names.items()}
    rows = []
    for label, run in runs.items():
        for split, row in run['summary'].iterrows():
            rows.append({'Model':label, 'Split':split, **row.to_dict()})
    summary = pd.DataFrame(rows).set_index(['Model','Split'])
    metrics = ['loss', 'accuracy', 'mean_iou']
    columns = pd.MultiIndex.from_product([metrics, ['train', 'validation']],
                                        names=['Metric', 'Split'])
    comparison = summary[metrics].unstack('Split').reindex(
        index=list(model_names), columns=columns)
    comparison.columns = pd.MultiIndex.from_tuples(
        [(dict(loss='Loss', accuracy='Accuracy', mean_iou='Mean IoU')[metric],
          split.capitalize()) for metric, split in comparison.columns],
        names=['Metric', 'Split'])
    formats = {column: ('{:.4f}' if column[0] == 'Loss' else '{:.2%}')
               for column in comparison.columns}
    display(comparison.style.format(formats))
    if show_curves:
        fig, axes = plt.subplots(1,2,figsize=(12,4))
        for label, run in runs.items():
            for ax, split in zip(axes,('train','validation')):
                ax.plot(run['history'].epoch,run['history'][f'{split}_loss'],label=label)
                ax.set(xlabel='Epoch',ylabel='Cross-entropy loss',title=f'{split.capitalize()} loss')
        for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.tight_layout();plt.show()
    return summary


def save_training_summary(result, path):
    """Retain training loss and confusion counts instead of per-face predictions."""
    path = Path(path)
    path.write_text(json.dumps({'loss': float(result['loss']),
                               'confusion_matrix': result['confusion_matrix'].long().tolist()}, indent=2) + '\n')


def read_cached_metrics(root, final_epoch, class_names):
    root = Path(root)
    final = root / f'epoch_{final_epoch:03d}'
    rows, class_rows = [], []
    for split in ('train', 'validation'):
        if split == 'train':
            saved = json.loads((final / 'train_metrics.json').read_text())
            confusion = torch.tensor(saved['confusion_matrix'], dtype=torch.float64)
        else:
            saved = load_validation_predictions(final)
            labels = saved['labels'].long()
            predictions = saved['probabilities'].argmax(1)
            classes = saved['probabilities'].shape[1]
            confusion = torch.bincount(labels * classes + predictions, minlength=classes**2).reshape(classes, classes).double()
        correct = confusion.diag()
        support, predicted = confusion.sum(1), confusion.sum(0)
        union = support + predicted - correct
        iou = correct / union.masked_fill(union == 0, float('nan'))
        recall = correct / support.masked_fill(support == 0, float('nan'))
        precision = correct / predicted.masked_fill(predicted == 0, float('nan'))
        rows.append(dict(split=split, faces=int(support.sum()), loss=saved['loss'],
                         accuracy=(correct.sum() / support.sum()).item(), mean_iou=iou.nanmean().item()))
        for index in range(len(correct)):
            class_rows.append(dict(split=split, class_index=index, class_name=class_names[str(index)],
                support=int(support[index]), recall=recall[index].item(), precision=precision[index].item(), iou=iou[index].item()))
    return pd.DataFrame(rows).set_index('split'), pd.DataFrame(class_rows)


def load_validation_predictions(final):
    """Read compact probabilities with a shared solid/face index, or a fresh export."""
    import numpy as np
    final = Path(final)
    if (final / 'validation_predictions.pt').exists():
        return torch.load(final / 'validation_predictions.pt', map_location='cpu', weights_only=True)
    metadata = json.loads((final / 'validation_metrics.json').read_text())
    shared = REPORT_DIR / 'data' / 'validation_index'
    index = json.loads((shared / 'sample_ids.json').read_text())
    return {**metadata, 'probabilities': torch.from_numpy(np.load(final / 'validation_probabilities.npy')),
            'labels': torch.from_numpy(np.load(shared / 'labels.npy').astype('int64')),
            'sample_ids': index,
            'sample_offsets': torch.from_numpy(np.load(shared / 'sample_offsets.npy').astype('int64'))}


def compact_validation_cache(run_dir, final_epoch):
    """Store probabilities once per model, sharing the verified validation index."""
    import numpy as np
    final = Path(run_dir) / f'epoch_{final_epoch:03d}'
    source = final / 'validation_predictions.pt'
    if not source.exists():
        return
    saved = torch.load(source, map_location='cpu', weights_only=True)
    shared = REPORT_DIR / 'data' / 'validation_index'
    shared.mkdir(exist_ok=True)
    arrays = {'labels.npy': saved['labels'].numpy().astype('uint8'),
              'sample_offsets.npy': saved['sample_offsets'].numpy().astype('int32')}
    for name, array in arrays.items():
        path = shared / name
        if path.exists():
            if not np.array_equal(np.load(path), array):
                raise ValueError('Validation face order differs from the shared index.')
        else:
            np.save(path, array)
    index = shared / 'sample_ids.json'
    if index.exists() and json.loads(index.read_text()) != saved['sample_ids']:
        raise ValueError('Validation solid order differs from the shared index.')
    index.write_text(json.dumps(saved['sample_ids']) + '\n')
    np.save(final / 'validation_probabilities.npy', saved['probabilities'].numpy())
    (final / 'validation_metrics.json').write_text(json.dumps({'loss': float(saved['loss']),
        'split': 'validation', 'index': 'data/validation_index', 'faces': len(saved['labels'])}, indent=2) + '\n')
    source.unlink()


def cache_wl_embeddings(model, validation_loader, final, config, history, vocabulary):
    """Keep complete validation inputs and first-layer activations instead of WL weights."""
    import numpy as np
    final = Path(final)
    geometry, colors, activations, probabilities = [], [], [], []
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        for inputs, _ in validation_loader:
            inputs = inputs.to(device)
            hidden = torch.relu(model.geometry_projection(inputs.geometry) +
                (model.color_projection(inputs.colors) *
                 (inputs.colors != model.unknown_indices).unsqueeze(-1)).sum(1))
            geometry.append(inputs.geometry.cpu().numpy())
            colors.append(inputs.colors.cpu().numpy().astype('int32'))
            activations.append(hidden.cpu().numpy())
            probabilities.append(torch.softmax(model.linear2(hidden), 1).cpu())
    saved = load_validation_predictions(final)
    actual = torch.cat(probabilities)
    if not torch.allclose(actual, saved['probabilities'], atol=1e-4, rtol=1e-4):
        raise ValueError('WL embeddings do not reproduce validation probabilities.')
    for name, values in [('geometry.npy', geometry), ('wl_tokens.npy', colors),
                         ('first_layer_relu.npy', activations)]:
        np.save(final / name, np.concatenate(values))
    metadata = {'config': config, 'history': history, 'wl_encoding': {
        'size': vocabulary['size'], 'iterations': vocabulary['iterations'],
        'unknown_indices': vocabulary['unknown_indices'],
        'encoding': 'Disjoint-round sparse one-hot column IDs; unknowns contribute zero'}}
    (final / 'embedding_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (final / 'checkpoint.pt').unlink()


def validation_class_counts(model_name):
    """Recompute ordered per-class counts and IoU from a verified final cache."""
    from data import ordered_class_indices
    run = load_cached_run(model_name)
    names = json.loads((run['root'] / 'result.json').read_text())['class_names']
    saved = load_validation_predictions(run['root'] / f"epoch_{run['epoch']:03d}")
    labels = saved['labels'].long()
    probabilities = saved['probabilities']
    classes = probabilities.shape[1]
    confusion = torch.bincount(labels * classes + probabilities.argmax(1),
                              minlength=classes**2).reshape(classes, classes)
    true_positive = confusion.diag()
    support, predicted = confusion.sum(1), confusion.sum(0)
    false_negative, false_positive = support - true_positive, predicted - true_positive
    union = support + predicted - true_positive
    iou = true_positive.double() / union.double().masked_fill(union == 0, float('nan'))
    order = ordered_class_indices(names)
    return pd.DataFrame([dict(class_index=i, class_name=names[str(i)],
        support=int(support[i]), true_positive=int(true_positive[i]),
        false_positive=int(false_positive[i]), false_negative=int(false_negative[i]),
        iou=float(iou[i])) for i in order])
