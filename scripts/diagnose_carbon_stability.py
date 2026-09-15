#!/usr/bin/env python
"""Read-only R42 inference and copied-model, train-only BatchNorm sensitivity."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from timepix.config import load_experiment_config, resolve_project_path
from timepix.data import build_dataloaders
from timepix.data.normalization import load_frozen_normalizer
from timepix.losses import build_loss
from timepix.models import build_model
from timepix.training.runner import _cuda_autocast_factory
from timepix.training.trainer import evaluate
from timepix.training.metrics import classification_metrics
from timepix.training.stability import stratified_indices, reestimate_batchnorm, batchnorm_snapshot
from timepix.utils.seed import set_seed
from scripts.run_carbon_difference_queue import BASELINE, SPLIT_SHA256, METADATA_SHA256, atomic_json
from scripts.summarize_carbon_difference import write_csv, METRICS

GROUP = 'carbon_stability_20260916'
OUTPUT = ROOT / 'outputs' / GROUP
CONFIG = ROOT / 'configs/experiments/carbon_t7_stability_lr1e4_seed42.yaml'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen_config(data_root):
    cfg = load_experiment_config(CONFIG)
    cfg['dataset']['root'] = str(data_root)
    _, provenance = load_frozen_normalizer(
        resolve_project_path(cfg['data']['normalizer_metadata']), ['ToT'], cfg['normalization'],
        resolve_project_path(cfg['split']['path']), cfg['dataset'], cfg['data'])
    if provenance['split_sha256'] != SPLIT_SHA256 or provenance['metadata_sha256'] != METADATA_SHA256:
        raise ValueError('Frozen R42 metadata or original frame split identity changed')
    return cfg, provenance


def manifest_rows(dataset, indices, split, purpose, frame_map):
    return [dict(purpose=purpose, split=split, index=i, sample_key=dataset.records[i].key,
                 angle=dataset.records[i].angle, file=str(dataset.records[i].modalities['ToT']),
                 **{key: frame_map[dataset.records[i].key][key]
                    for key in ['frame_group_key', 'raw_frame_member']}) for i in indices]


def claim_output(output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)


def run(data_root, output):
    output = Path(output)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU inference required; do not heat the local laptop with a CPU fallback')
    cfg, normalizer = frozen_config(data_root)
    claim_output(output)
    sources = {str(BASELINE / f): sha256(BASELINE / f)
               for f in ['best_model.pth', 'last_checkpoint.pth', 'metadata.json', 'validation_predictions.csv']}
    atomic_json(output / 'started.json', dict(source_sha256=sources, normalizer=normalizer,
        protocol_change='New training: ValAcc > lower ValMAE > ValMacroF1 > earlier epoch; old R42 unchanged',
        training_started=False, test_evaluated=False))
    set_seed(42)
    loaders, info = build_dataloaders(cfg, data_root_override=str(data_root), eval_mode=True)
    # Do not iterate the constructed test loader, including for statistics.
    train_ds = loaders['train'].dataset
    probe_indices = stratified_indices(train_ds.records, 256, 20260916)
    calibration_indices = stratified_indices(train_ds.records, 512, 20260917)
    def subset_loader(indices):
        return DataLoader(Subset(train_ds, indices), batch_size=128, shuffle=False,
                          num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(42))
    probe_loader, calibration_loader = subset_loader(probe_indices), subset_loader(calibration_indices)
    manifest = json.loads(resolve_project_path(cfg['split']['path']).read_text(encoding='utf-8'))
    frame_map = manifest['samples']
    provenance = manifest_rows(train_ds, probe_indices, 'train', 'train_eval', frame_map)
    provenance += manifest_rows(train_ds, calibration_indices, 'train', 'BN_calibration', frame_map)
    provenance += manifest_rows(loaders['val'].dataset, range(len(loaders['val'].dataset)), 'val', 'val_eval', frame_map)
    write_csv(output / 'sample_manifest.csv', provenance)
    device = torch.device('cuda')
    criterion = build_loss(cfg, info['num_classes'], info['label_map']).to(device)
    rows, comparisons, bn_checks = [], [], []
    all_metrics = {}
    for checkpoint_kind in ['best', 'last']:
        path = BASELINE / ('best_model.pth' if checkpoint_kind == 'best' else 'last_checkpoint.pth')
        stored = torch.load(path, map_location='cpu', weights_only=True)
        state = stored if checkpoint_kind == 'best' else stored['model_state']
        model = build_model(cfg, 1, info['num_classes'], 'classification', 0).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        del stored, state
        for bn_mode in ['original', 'train_reestimated']:
            variant = model if bn_mode == 'original' else copy.deepcopy(model)
            case = f'{checkpoint_kind}_{bn_mode}'
            if bn_mode == 'train_reestimated':
                before = batchnorm_snapshot(variant)
                details = reestimate_batchnorm(variant, calibration_loader, device)
                bn_checks.append(dict(case=case, before=before, after=batchnorm_snapshot(variant), **details))
            precision_payloads = {}
            for precision in ['fp32', 'amp_float16']:
                autocast = None if precision == 'fp32' else _cuda_autocast_factory(torch.float16)
                for split, loader in [('train_subset', probe_loader), ('val', loaders['val'])]:
                    print(f'DIAG {case} {precision} {split}', flush=True)
                    payload = evaluate(variant, loader, criterion, device, 'classification', autocast_factory=autocast)
                    logits, labels = payload['logits'], payload['labels']
                    if not np.isfinite(logits).all() or not np.isfinite(payload['loss']):
                        atomic_json(output / 'failure.json', dict(case=case, precision=precision, split=split,
                                    reason='Nonfinite inference; controller review required'))
                        raise ValueError('Nonfinite diagnostic inference')
                    metrics = classification_metrics(logits, labels, info['angle_values'])
                    key = f'{case}_{precision}_{split}'
                    all_metrics[key] = metrics
                    np.savez_compressed(output / f'{key}.npz', logits=logits, labels=labels)
                    rows.append(dict(case=case, precision=precision, split=split, n=len(labels),
                                     loss=payload['loss'], **{k: metrics[k] for k in METRICS}))
                    precision_payloads[(precision, split)] = (logits, labels)
            for split in ['train_subset', 'val']:
                fp, labels = precision_payloads[('fp32', split)]
                amp, other_labels = precision_payloads[('amp_float16', split)]
                if not np.array_equal(labels, other_labels):
                    raise ValueError('Diagnostic precision comparison changed sample order')
                comparisons.append(dict(case=case, split=split,
                    argmax_disagreements=int(np.sum(fp.argmax(1) != amp.argmax(1))),
                    argmax_disagreement_rate=float(np.mean(fp.argmax(1) != amp.argmax(1))),
                    max_absolute_logit_difference=float(np.max(np.abs(fp - amp)))))
            if variant is not model:
                del variant
        del model
        gc.collect()
        torch.cuda.empty_cache()
    same_sources = all(sha256(path) == h for path, h in sources.items())
    if not same_sources:
        raise RuntimeError('A source file changed during read-only diagnosis')
    write_csv(output / 'diagnostic_summary.csv', rows)
    write_csv(output / 'precision_comparison.csv', comparisons)
    atomic_json(output / 'all_metrics.json', all_metrics)
    atomic_json(output / 'bn_reestimation.json', bn_checks)
    baseline = json.loads((BASELINE / 'metadata.json').read_text(encoding='utf-8'))
    replay = all_metrics['best_original_amp_float16_val']
    replay_deltas = {k: replay[k] - baseline['metrics']['validation'][k] for k in METRICS}
    verification = dict(status='complete', training_started=False, optimizer_steps=0, test_evaluated=False,
        source_files_unchanged=same_sources, source_sha256=sources, normalizer=normalizer,
        train_probe_events=len(probe_indices), calibration_events=len(calibration_indices),
        validation_events=len(loaders['val'].dataset), bn_calibration_split='train_only',
        bn_calibration_views='original; not the four augmented training views',
        replay_minus_saved_metrics=replay_deltas, torch=str(torch.__version__),
        device=torch.cuda.get_device_name(), all_non_bn_state_preserved=all(r['non_bn_state_unchanged'] for r in bn_checks),
        ready_for_controller_review=True, training_auto_authorized=False)
    atomic_json(output / 'verification.json', verification)
    lines = ['# R42 stability diagnostic (not training)', '',
        'Only best epoch3 and last epoch11 are available; epoch6/9 cannot be reconstructed.',
        'Original files unchanged; no test inference; BN probes use train original views only.',
        'BN-reestimated metrics are sensitivity diagnostics, not replacement formal model scores.', '',
        '| Checkpoint / BN | Precision | Split | N | Acc % | MAE deg | Macro-F1 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['case']} | {r['precision']} | {r['split']} | {r['n']} | {100*r['accuracy']:.4f} | {r['mae_argmax']:.6f} | {r['macro_f1']:.6f} |")
    lines += ['', 'Protocol change: subsequent S42 training selects Val Acc, then lower Val MAE, then Macro-F1, then earlier epoch.',
        'Historical R42 and CDC/APDC remain MAE-selected. S42 vs R42 changes both LR and selection/early-stop rule.',
        'Cumulative BN batch-moment averaging is not an exact pooled variance estimate or proof of causation.',
        'Training-subset scores and validation scores have different sampling distributions; do not interpret their raw gap as causal.']
    (output / 'diagnostic_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'DIAG COMPLETE: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT / 'diagnostic')
    args = parser.parse_args()
    run(args.data_root, args.output)


if __name__ == '__main__':
    main()
