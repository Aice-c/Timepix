"""Validation-only angle outputs with source-frame provenance."""
import csv
import json
from pathlib import Path

import numpy as np

from .logger import write_json
from .metrics import classification_metrics


def chance_references(train_labels, val_labels, angles):
    train_labels = np.asarray(train_labels, dtype=int)
    val_labels = np.asarray(val_labels, dtype=int)
    counts = np.bincount(train_labels, minlength=len(angles))
    majority = int(counts.argmax())
    # Lower empirical median is an observed class and minimizes training L1 error.
    median = int(np.searchsorted(counts.cumsum(), (len(train_labels) + 1) // 2))
    output = {'uniform_random_expected_accuracy': 1 / len(angles)}
    for name, index in [('train_majority', majority), ('train_median_angle', median)]:
        logits = np.full((len(val_labels), len(angles)), -100., dtype=float)
        logits[:, index] = 0
        output[name] = dict(predicted_angle=float(angles[index]),
                            validation=classification_metrics(logits, val_labels, list(angles)))
    return output


def export_validation(out, payload, loaders, data_info, cfg):
    if data_info.get('label_type', 'angle_folder') != 'angle_folder':
        raise ValueError('Frame-qualified validation export currently supports angle tasks only')
    out = Path(out)
    records = loaders['val'].dataset.records
    labels = np.asarray(payload['labels'], dtype=int)
    if len(labels) != len(records) or not np.array_equal(labels, [r.label for r in records]):
        raise ValueError('Validation prediction order does not match dataset records')
    angles = np.asarray(data_info['angle_values'], dtype=float)
    logits = np.asarray(payload['logits'])
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted); probs /= probs.sum(axis=1, keepdims=True)
    manifest = json.loads(Path(data_info['split_path']).read_text(encoding='utf-8'))
    source_rows = manifest.get('samples', {})
    if not all(r.key in source_rows for r in records):
        raise ValueError('Validation provenance missing in split manifest')
    evaluation = cfg.get('evaluation', {})
    probability_fields = [f'prob_{a:g}' for a in angles]
    fields = ['task_id', 'experiment_id', 'seed', 'sample_key', 'frame_group_key',
              'raw_frame_member', 'true_angle', 'pred_angle', 'abs_error'] + probability_fields
    with (out / 'validation_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for record, label, prob in zip(records, labels, probs):
            source = source_rows[record.key]
            pred_angle = float(angles[prob.argmax()])
            row = dict(task_id=evaluation.get('task_id', ''), experiment_id=evaluation.get('experiment_id', ''),
                       seed=cfg.get('training', {}).get('seed', 42), sample_key=record.key,
                       frame_group_key=source['frame_group_key'], raw_frame_member=source.get('raw_frame_member', ''),
                       true_angle=float(angles[label]), pred_angle=pred_angle,
                       abs_error=abs(pred_angle - angles[label]))
            row.update(zip(probability_fields, map(float, prob))); writer.writerow(row)
    metrics = classification_metrics(logits, labels, angles.tolist())
    write_json(out / 'validation_metrics.json', metrics)
    cm = np.array(metrics['confusion_matrix'])
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for name, values in [('validation_confusion_counts', cm), ('validation_confusion_row_normalized', norm)]:
        with (out / f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['true_angle'] + angles.tolist())
            writer.writerows([[float(a)] + row.tolist() for a, row in zip(angles, values)])
    write_json(out / 'validation_adjacent_confusions.json', {
        f'{a:g}<->{b:g}': int(cm[i, i+1] + cm[i+1, i])
        for i, (a, b) in enumerate(zip(angles[:-1], angles[1:]))})
    write_json(out / 'chance_references.json', chance_references(
        [r.label for r in loaders['train'].dataset.records], labels, angles))
