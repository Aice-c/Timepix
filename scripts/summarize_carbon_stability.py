#!/usr/bin/env python
"""Summarize the single stability run, preserving the selection-rule change."""
import csv
import json
import math
from pathlib import Path
import sys
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from scripts.diagnose_carbon_stability import OUTPUT, sha256, CONFIG, GROUP
from timepix.config import load_experiment_config
from scripts.run_carbon_difference_queue import BASELINE, atomic_json, clean_config
from scripts.summarize_carbon_difference import read_run, write_csv, local_run


def verify_historical_sources(checks, baseline_dir=BASELINE):
    expected = {'best_model.pth', 'last_checkpoint.pth', 'metadata.json', 'validation_predictions.csv'}
    names = {str(path).replace('\\', '/').rsplit('/', 1)[-1] for path in checks}
    if names != expected or len(checks) != len(expected):
        raise ValueError('Incomplete historical source hash manifest')
    for original, digest in checks.items():
        name = str(original).replace('\\', '/').rsplit('/', 1)[-1]
        if sha256(Path(baseline_dir) / name) != digest:
            raise ValueError(f'Historical source changed: {name}')
    return True


def verify_run_identity(saved, metadata, approved):
    if clean_config(saved) != clean_config(approved):
        raise ValueError('Run config identity differs from approved single run')
    for key in ['dataset', 'data', 'model', 'loss', 'training', 'experiment_name', 'experiment_group']:
        if metadata[key] != approved[key]:
            raise ValueError(f'Run metadata identity mismatch: {key}')
    if metadata['primary_metric'] != approved['task']['primary_metric']:
        raise ValueError('Run selection identity mismatch')
    return True


def verify_step_inventory(observed, epochs, batches_per_epoch, total_batches):
    if epochs != list(range(1, len(epochs) + 1)) or set(observed) != set(epochs):
        raise ValueError('Unexpected epoch/batch inventory')
    for epoch in epochs:
        if observed[epoch] != list(range(1, batches_per_epoch + 1)):
            raise ValueError(f'Missing, duplicated or out-of-order batch at epoch {epoch}')
    if sum(map(len, observed.values())) != total_batches:
        raise ValueError('Total batch inventory differs from completed training')
    return True


def summarize():
    status = json.loads((OUTPUT / 'training_status.json').read_text(encoding='utf-8'))
    if status['status'] != 'complete':
        raise ValueError('Training is not complete; no final summary')
    diagnostic = json.loads((OUTPUT / 'diagnostic/verification.json').read_text(encoding='utf-8'))
    old_unchanged = verify_historical_sources(diagnostic['source_sha256'])
    baseline, _, _, keys = read_run(dict(id='R42', method='R', seed=42, status='historical', run_dir=str(BASELINE)))
    current, metrics, meta, _ = read_run(dict(id='S42', method='S', seed=42, status='complete', run_dir=status['run_dir']), keys)
    run = local_run(status['run_dir'])
    approved = load_experiment_config(CONFIG)
    approved['dataset']['root'] = meta['dataset']['root']
    saved = yaml.safe_load((run / 'config.yaml').read_text(encoding='utf-8'))
    verify_run_identity(saved, meta, approved)
    matches = list((ROOT / 'outputs/experiments' / GROUP).iterdir())
    if matches != [run]:
        raise ValueError('Expected exactly the single approved S42 run')
    baseline['selection_rule'] = 'Val MAE > Macro-F1 > earlier epoch'
    current['selection_rule'] = 'Val Acc > lower Val MAE > Macro-F1 > earlier epoch'
    write_csv(OUTPUT / 'validation_comparison.csv', [baseline, current])
    with (run / 'training_log.csv').open(encoding='utf-8', newline='') as f:
        logs = list(csv.DictReader(f))
    selected = max(logs, key=lambda r: (float(r['val_accuracy']), -float(r['val_mae_argmax']),
                                        float(r['val_macro_f1']), -int(r['epoch'])))
    if int(selected['epoch']) != int(current['best_epoch']):
        raise ValueError('Acc-first checkpoint selection mismatch')
    step_summary, observed_batches = {}, {}
    with (run / 'stability/steps.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            epoch = int(r['epoch'])
            observed_batches.setdefault(epoch, []).append(int(r['batch']))
            s = step_summary.setdefault(epoch, dict(epoch=epoch, batches=0, optimizer_steps=0,
                skipped_steps=0, nonfinite_gradient_batches=0, max_gradient_l2=0., scale_min=r['scale_before']))
            s['batches'] += 1
            s['optimizer_steps'] += int(r['optimizer_step_executed'])
            s['skipped_steps'] += int(not r['optimizer_step_executed'])
            s['nonfinite_gradient_batches'] += int(not r['gradient_finite'])
            if r['gradient_l2_unscaled'] is not None:
                s['max_gradient_l2'] = max(s['max_gradient_l2'], r['gradient_l2_unscaled'])
            s['scale_min'] = min(s['scale_min'], r['scale_after'])
    epochs = [int(row['epoch']) for row in logs]
    views = 4 if saved['augmentation'].get('rotation_90', False) else 1
    expected_batches = math.ceil(meta['data_info']['split_counts']['train'] * views / saved['training']['batch_size'])
    verify_step_inventory(observed_batches, epochs, expected_batches, meta['metrics']['training_batches_completed'])
    if len(epochs) != int(meta['metrics']['stopped_epoch']):
        raise ValueError('Epoch log differs from stopped epoch')
    for row in logs:
        epoch = int(row['epoch'])
        if epoch not in step_summary or not (run / f'stability/epoch_{epoch:03d}.pth').is_file():
            raise ValueError(f'Missing epoch diagnostic: {epoch}')
        step_summary[epoch].update({k: float(row[k]) for k in ['lr', 'train_loss', 'val_loss', 'train_accuracy',
            'val_accuracy', 'val_mae_argmax', 'val_macro_f1']})
    write_csv(OUTPUT / 'epoch_diagnostics.csv', list(step_summary.values()))
    write_csv(OUTPUT / 'per_class.csv', metrics['per_class'])
    atomic_json(OUTPUT / 'summary_verification.json', dict(prediction_metrics_recomputed=True,
        acc_first_selection_verified=True, test_evaluated=False, n_new_runs=1, seed=42,
        run_identity_verified=True, complete_batch_inventory_verified=True,
        same_time_lr3e4_repeat=False, old_outputs_unchanged=old_unchanged,
        total_batches=sum(r['batches'] for r in step_summary.values()),
        total_optimizer_steps=sum(r['optimizer_steps'] for r in step_summary.values())))
    lines = ['# Carbon T7 stability follow-up', '',
        '**Protocol change for Astra:** 2026-09-16 user restored Val Accuracy first; ties: lower actual-angle MAE, higher Macro-F1, earlier epoch. Early stopping follows the same ordering.',
        'R42 and previous CDC/APDC remain historical MAE-selected results; they are NOT relabeled as Acc-selected.',
        'Only one new lr1e-4 seed42 run. Same-time lr3e-4 reproduction was explicitly cancelled.',
        'LR and selection/early-stopping changed together; do not attribute all historical differences to LR.',
        'No new data split, model, input transform, test evaluation or seed selection.', '',
        '| Run | LR | Selection | Val Acc % | Val MAE | Val Macro-F1 | Best/Stop |',
        '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for r, lr in [(baseline, '3e-4'), (current, '1e-4')]:
        lines.append(f"| {r['id']} | {lr} | {r['selection_rule']} | {100*r['accuracy']:.4f} | {r['mae_argmax']:.6f} | {r['macro_f1']:.6f} | {r['best_epoch']}/{r['stopped_epoch']} |")
    lines += ['', 'See diagnostic/ for the no-training precision and copied-model BN probe. Those scores are not formal replacements.',
              'This is n=1; no mean/std or stable-across-seeds claim. Inspect full epoch_diagnostics.csv before describing stability.']
    (OUTPUT / 'astra_handoff.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    with zipfile.ZipFile(OUTPUT / 'carbon_stability_review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and p.suffix in {'.json', '.csv', '.md', '.yaml', '.log', '.npz'}:
                z.write(p, 'summary/' + p.relative_to(OUTPUT).as_posix())
        for p in run.rglob('*'):
            if p.is_file() and p.suffix in {'.json', '.csv', '.yaml', '.jsonl'}:
                z.write(p, 'run/' + p.relative_to(run).as_posix())
        for pattern in ['scripts/*carbon_stability.py', 'tests/test_stability*.py',
                        'configs/experiments/carbon_t7_stability*.yaml', 'agent/CARBON_STABILITY*.md',
                        'timepix/training/stability.py']:
            for p in ROOT.glob(pattern):
                z.write(p, 'code/' + p.relative_to(ROOT).as_posix())
    print(f'Summary and upload package: {OUTPUT}')


if __name__ == '__main__':
    summarize()
