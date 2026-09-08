#!/usr/bin/env python
"""Summarize only explicitly selected carbon validation runs, never select by test."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_carbon_controls import csv_text, immutable_text, json_text
from scripts.check_carbon_controls import pair_protocol

METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'mae_argmax', 'p90_error']


def summarize(paths):
    rows = []; seen = set(); protocols = {}
    for path in paths:
        path = Path(path)
        metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        import yaml
        cfg = yaml.safe_load((path / 'config.yaml').read_text(encoding='utf-8'))
        experiment = cfg['evaluation']['experiment_id']
        if experiment in seen:
            raise ValueError(f'Duplicate experiment; specify one approved run per cell: {experiment}')
        seen.add(experiment)
        protocols[experiment] = pair_protocol(cfg)
        m = metadata['metrics']
        if m.get('test_evaluated', True):
            raise ValueError(f'Run not validation-only: {path}')
        if not (path / 'validation_predictions.csv').is_file():
            raise ValueError(f'Missing validation predictions: {path}')
        row = dict(experiment_id=experiment, task_id=cfg['evaluation']['task_id'], run_dir=str(path),
                   seed=cfg['training']['seed'], split_sha256=metadata['split_manifest_hash'],
                   best_epoch=m['best_epoch'], stopped_epoch=m['stopped_epoch'],
                   training_batches_completed=m.get('training_batches_completed'), fit_seconds=m['fit_seconds'])
        row.update({k: m['validation'][k] for k in METRICS}); rows.append(row)
    pairs = []
    by_id = {r['experiment_id']: r for r in rows}
    for a, b in [('T7-ToT', 'T7-Mask'), ('V6-HiRes', 'V6-Base')]:
        if a in by_id and b in by_id:
            left, right = by_id[a], by_id[b]
            if protocols[a] != protocols[b]:
                raise ValueError(f'Paired training protocol differs: {a}, {b}')
            if left['split_sha256'] != right['split_sha256'] or left['seed'] != right['seed']:
                raise ValueError(f'Incompatible paired split/seed: {a}, {b}')
            for metric in METRICS:
                pairs.append(dict(comparison=f'{a} minus {b}', metric=metric,
                                  difference=left[metric]-right[metric],
                                  better='lower' if metric in ['mae_argmax', 'p90_error'] else 'higher'))
    return rows, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True, help='Exact completed run directory; repeat for pairs')
    parser.add_argument('--output', required=True, type=Path, help='New summary directory; existing different files refused')
    args = parser.parse_args()
    rows, pairs = summarize(args.run)
    immutable_text(args.output / 'validation_summary.csv', csv_text(rows))
    immutable_text(args.output / 'paired_differences.csv', csv_text(pairs, ['comparison', 'metric', 'difference', 'better']))
    immutable_text(args.output / 'summary_status.json', json_text(dict(
        completed_cells=[r['experiment_id'] for r in rows], single_seed=True, test_included=False,
        statistical_significance_not_established=True)))
    print(csv_text(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
