#!/usr/bin/env python
"""Validation-only, per-run then per-seed summaries for the frozen T7 queue."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from timepix.training.metrics import classification_metrics
from scripts.run_carbon_difference_queue import BASELINE, GROUP, SPLIT_SHA256, atomic_json

ANGLES = [10, 20, 30, 45, 50, 60, 70]
METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'mae_argmax', 'p90_error']


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else None)


def aggregate_rows(rows, metrics=METRICS):
    aggregate, pairs = [], []
    for metric in metrics:
        values = {m: {int(r['seed']): float(r[metric]) for r in rows if r['method'] == m and metric in r}
                  for m in ['R', 'A', 'B']}
        for method, by_seed in values.items():
            if by_seed:
                mean, std = mean_std(list(by_seed.values()))
                aggregate.append(dict(method=method, metric=metric, n=len(by_seed), mean=mean, std=std,
                                      **{f'seed{s}': v for s, v in sorted(by_seed.items())}))
        seeds = sorted(values['A'].keys() & values['B'].keys())
        differences = [values['B'][s] - values['A'][s] for s in seeds]
        lower = metric in {'mae_argmax', 'p90_error'}
        for seed, value in zip(seeds, differences):
            pairs.append(dict(comparison_kind='paired_B_minus_A_seed', seed=seed, metric=metric,
                              difference=value, n_pairs=len(seeds), better='lower' if lower else 'higher'))
        if differences:
            mean, std = mean_std(differences)
            pairs.append(dict(comparison_kind='paired_B_minus_A_summary', metric=metric,
                              difference=mean, std=std, n_pairs=len(seeds),
                              improved_seeds=sum(v < 0 if lower else v > 0 for v in differences)))
        if 42 in values['R']:
            for method in ['A', 'B']:
                if 42 in values[method]:
                    pairs.append(dict(comparison_kind=f'{method}42_minus_R42', seed=42, metric=metric,
                                      difference=values[method][42] - values['R'][42], n_pairs=1))
                if values[method]:
                    pairs.append(dict(comparison_kind=f'{method}_mean_minus_historical_single_R42_reference',
                                      metric=metric, difference=float(np.mean(list(values[method].values()))) - values['R'][42],
                                      candidate_n=len(values[method]), baseline_n=1, n_pairs=0))
    return aggregate, pairs


def local_run(path):
    path = Path(path)
    if path.is_dir():
        return path
    # A remote absolute path is resolved only to the same exact local group/run.
    relative = Path('outputs/experiments') / path.parent.name / path.name
    candidate = ROOT / relative
    if not candidate.is_dir():
        raise FileNotFoundError(f'Exact run not found: {path}; {candidate}')
    return candidate


def read_run(item, baseline_keys=None):
    path = local_run(item['run_dir'])
    metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['metrics'].get('test_evaluated', True) or metadata['split_manifest_hash'] != SPLIT_SHA256:
        raise ValueError(f'Wrong split or test evaluation in {path}')
    with (path / 'validation_predictions.csv').open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 10204:
        raise ValueError(f'Expected 10204 original validation events: {path}')
    keys = [(r['sample_key'], r['frame_group_key'], float(r['true_angle'])) for r in records]
    if len(set(keys)) != len(keys) or (baseline_keys is not None and keys != baseline_keys):
        raise ValueError(f'Validation membership/order/provenance differs: {path}')
    probabilities = np.asarray([[float(r[f'prob_{a}']) for a in ANGLES] for r in records])
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-6)):
        raise ValueError(f'Invalid probabilities: {path}')
    y = np.asarray([ANGLES.index(float(r['true_angle'])) for r in records])
    metrics = classification_metrics(np.log(np.maximum(probabilities, np.finfo(float).tiny)), y, ANGLES)
    predicted = np.asarray(ANGLES)[probabilities.argmax(1)]
    if not np.array_equal(predicted, [float(r['pred_angle']) for r in records]):
        raise ValueError(f'Argmax labels disagree: {path}')
    errors = np.abs(predicted - np.asarray(ANGLES)[y])
    if not np.allclose(errors, [float(r['abs_error']) for r in records], atol=1e-8):
        raise ValueError(f'Angle errors disagree: {path}')
    for metric in METRICS:
        if not np.isclose(metrics[metric], metadata['metrics']['validation'][metric], atol=1e-7, rtol=0):
            raise ValueError(f'Prediction/recorded metric differs: {path} {metric}')
    m = metadata['metrics']
    row = dict(id=item['id'], method=item['method'], seed=item['seed'], run_id=path.name,
               run_dir=str(path), status=item['status'], stop_reason=item.get('stop_reason', 'historical reuse'),
               best_epoch=m['best_epoch'], stopped_epoch=m['stopped_epoch'], fit_seconds=m['fit_seconds'],
               param_count=metadata['param_count'], **{k: metrics[k] for k in METRICS})
    high = np.asarray(ANGLES)[y] >= 45
    row.update(high_true_event_count=int(high.sum()), high_true_accuracy=float(np.mean(errors[high] == 0)),
               high_true_mae=float(np.mean(errors[high])), high_angle_macro_f1=metrics['high_angle_macro_f1'])
    return row, metrics, metadata, keys


def package(output, manifest):
    output = Path(output)
    destination = output / 'carbon_difference_review.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file.suffix.lower() in {'.json', '.csv', '.md', '.yaml', '.log', '.png', '.txt'}:
                archive.write(file, f'summary/{file.relative_to(output).as_posix()}')
        for item in [manifest['baseline']] + manifest['runs']:
            if not item.get('run_dir'):
                continue
            path = local_run(item['run_dir'])
            for file in sorted(path.iterdir()):
                if file.is_file() and file.suffix.lower() in {'.json', '.csv', '.yaml'}:
                    archive.write(file, f"runs/{item['id']}/{file.name}")
        for pattern in ['configs/experiments/carbon*layer1*.yaml', 'configs/experiments/carbon_difference*.yaml',
                        'scripts/*carbon_difference*.py', 'tests/test_difference*.py',
                        'agent/CARBON_DIFFERENCE*.md', 'timepix/models/difference.py']:
            for file in ROOT.glob(pattern):
                archive.write(file, f'code/{file.relative_to(ROOT).as_posix()}')
    return destination


def summarize(output):
    output = Path(output)
    manifest = json.loads((output / 'queue_manifest.json').read_text(encoding='utf-8'))
    rows, per_class, cms, metadata_by_id = [], [], {}, {}
    baseline_keys = None
    for item in [manifest['baseline']] + manifest['runs']:
        if item['status'] not in {'completed', 'early_stopped', 'reused'}:
            rows.append(dict(id=item['id'], method=item['method'], seed=item['seed'], status=item['status'],
                             run_dir=item.get('run_dir'), stop_reason=item.get('stop_reason')))
            continue
        row, metrics, metadata, keys = read_run(item, baseline_keys)
        if baseline_keys is None:
            baseline_keys = keys
        rows.append(row)
        metadata_by_id[item['id']] = metadata
        for cls in metrics['per_class']:
            per_class.append(dict(id=item['id'], method=item['method'], seed=item['seed'],
                                  angle=cls['class_name'], recall=cls['recall'], f1=cls['f1'], support=cls['support']))
        cm = np.asarray(metrics['confusion_matrix'])
        normalized = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        cms.setdefault(item['method'], []).append(normalized)
        for suffix, matrix in [('counts', cm), ('row_normalized', normalized)]:
            write_csv(output / 'confusions' / f"{item['id']}_{suffix}.csv",
                      [dict(true_angle=a, **{f'pred_{b}': float(v) for b, v in zip(ANGLES, values)})
                       for a, values in zip(ANGLES, matrix)])
    paired_initializations = []
    for seed in [42, 43, 44]:
        a, b = metadata_by_id.get(f'A{seed}'), metadata_by_id.get(f'B{seed}')
        if a and b:
            for key in ['initial_raw_state_sha256', 'cpu_rng_after_initialization_sha256']:
                if a['reproducibility'][key] != b['reproducibility'][key]:
                    raise ValueError(f'Paired initialization/RNG mismatch seed{seed}: {key}')
            paired_initializations.append(seed)
    aggregate, pairs = aggregate_rows(rows)
    write_csv(output / 'validation_summary.csv', rows)
    write_csv(output / 'seed_aggregate_summary.csv', aggregate)
    write_csv(output / 'paired_differences.csv', pairs)
    write_csv(output / 'per_class_by_seed.csv', per_class)
    class_summary = []
    for method in ['R', 'A', 'B']:
        for angle in map(str, ANGLES):
            selected = [r for r in per_class if r['method'] == method and r['angle'] == angle]
            if selected:
                for metric in ['recall', 'f1']:
                    mean, std = mean_std([r[metric] for r in selected])
                    class_summary.append(dict(method=method, angle=angle, metric=metric, n=len(selected), mean=mean, std=std))
    write_csv(output / 'per_class_mean_std.csv', class_summary)
    for method, matrices in cms.items():
        write_csv(output / 'confusions' / f'{method}_mean_row_normalized.csv',
                  [dict(true_angle=a, n_runs=len(matrices), **{f'pred_{b}': float(v) for b, v in zip(ANGLES, values)})
                   for a, values in zip(ANGLES, np.mean(matrices, axis=0))])
    atomic_json(output / 'summary_verification.json', dict(test_evaluated=False, prediction_metrics_match=True,
        validation_events_per_run=10204, paired_initialization_verified_seeds=paired_initializations,
        aggregation='per-run first, sample std ddof=1; R n=1 std blank',
        average_confusion='mean of per-run true-class-normalized seven-class matrices'))
    lines = ['# Carbon T7 CDC / Mixed APDC Experiment Report', '',
             'Fixed split seed42; A/B training seeds42/43/44. R42 is a historical single-seed reference.',
             'Validation only. No test evaluation, new data selection, extra seeds or V6 transfer.', '',
             '| Cell | Status | Val Acc | Val Macro-F1 | Val MAE / deg | P90 | Best/Stop | Fit / s |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |']
    for row in rows:
        if 'accuracy' not in row:
            lines.append(f"| {row['id']} | {row['status']} | | | | | | |")
        else:
            lines.append(f"| {row['id']} | {row['status']} | {row['accuracy']:.6f} | {row['macro_f1']:.6f} | "
                         f"{row['mae_argmax']:.6f} | {row['p90_error']:.3f} | {row['best_epoch']}/{row['stopped_epoch']} | {row['fit_seconds']:.2f} |")
    lines += ['', 'See seed_aggregate_summary.csv for per-seed values, mean and sample std; paired_differences.csv for B minus A.',
              'A/B means minus R42 are historical single-seed references, not three-seed paired improvements.', '',
              'Stopping is only max25/patience8. A run ending at25 may still be improving; no convergence claim is made.',
              'Recorded fit time is descriptive, not a stopping limit or independent performance benchmark.', '',
              'Both operators change kernel parameterization, effective initialization and optimization/regularization.',
              'They neither add measurement information nor prove that physical within-cluster gradients caused any gain.',
              'No statistical significance, causal gate/gradient interpretation, or V6 separability is claimed.',
              'Controller review after result return supplies the comparative conclusion and any observed exceptions.']
    (output / 'experiment_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    package(output, manifest)
    print(f'Summary and lightweight package: {output}', flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs' / GROUP)
    args = parser.parse_args()
    summarize(args.output)
