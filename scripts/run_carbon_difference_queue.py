#!/usr/bin/env python
"""Six independent sequential T7 runs; no wall-clock training limit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from timepix.config import load_experiment_config, resolve_project_path
from timepix.config_validation import validate_experiment_config
from timepix.data.normalization import load_frozen_normalizer

GROUP = 'carbon_difference_controls_20260915'
SUMMARY_ROOT = ROOT / 'outputs' / GROUP
RUN_ROOT = ROOT / 'outputs' / 'experiments' / GROUP
BASELINE = ROOT / 'outputs/experiments/carbon_angle_controls_20260909/20260909_014839_carbon_t7_tot_seed42'
SPLIT_SHA256 = '0398729fc7e288f5c8b399ab4b3b9ac6f35e1d33ce684ce5d17c23383ad3e216'
METADATA_SHA256 = 'af8104fc585d1bcb0c87d9da6724e70e48d3f913c9bd00490890322b45119850'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def plan():
    return [dict(id=f'{method}{seed}', method=method, operator=operator, seed=seed,
                 config=str(ROOT / f'configs/experiments/carbon_t7_{operator}_layer1_theta07_seed{seed}.yaml'))
            for seed in (42, 43, 44) for method, operator in [('A', 'cdc'), ('B', 'apdc')]]


def clean_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError('Expected complete config mapping, not empty/truncated config')
    cfg = json.loads(json.dumps(cfg))
    for key in list(cfg):
        if key.startswith('_'):
            cfg.pop(key)
    cfg.get('training', {}).pop('resume_from', None)
    return cfg


def normal_completion(metrics):
    epoch = int(metrics.get('stopped_epoch', 0))
    if metrics.get('early_stopped') and 0 < epoch <= 25:
        return 'early_stopped'
    if epoch == 25 and metrics.get('max_epochs') == 25:
        return 'completed'
    return None


def read_completed(path):
    required = ['metadata.json', 'metrics.json', 'validation_predictions.csv',
                'validation_metrics.json', 'training_log.csv', 'best_model.pth', 'last_checkpoint.pth']
    if not all((path / name).is_file() for name in required):
        return None
    try:
        metrics = json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
        metadata = json.loads((path / 'metadata.json').read_text(encoding='utf-8'))
        json.loads((path / 'validation_metrics.json').read_text(encoding='utf-8'))
        if metadata['metrics'] != metrics:
            return None
    except (ValueError, OSError, KeyError):
        return None
    if metrics.get('test_evaluated', True):
        raise ValueError(f'Test evaluation is forbidden: {path}')
    status = normal_completion(metrics)
    return (status, metrics) if status else None


def matching_runs(cfg):
    matches = []
    for path in sorted(RUN_ROOT.glob(f"*_{cfg['experiment_name']}")):
        config_path = path / 'config.yaml'
        if not config_path.is_file():
            raise ValueError(f'Unidentified partial run; do not delete or restart silently: {path}')
        try:
            saved = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        except yaml.YAMLError as error:
            raise ValueError(f'Malformed run config: {config_path}') from error
        if clean_config(saved) != clean_config(cfg):
            raise ValueError(f'Same-named run has a different config: {path}')
        matches.append(path)
    if len(matches) > 1:
        raise ValueError(f'Multiple matching runs require controller review: {matches}')
    return matches[0] if matches else None


def inspect_item(cfg, item):
    try:
        path = matching_runs(cfg)
        return path, read_completed(path) if path else None
    except (ValueError, OSError) as error:
        item.update(status='failed', ended_at=now(), stop_reason=f'Run identification error: {error}')
        return None, None


def exit_status(code):
    return 'interrupted' if code < 0 or code in {130, 137, 143} else 'failed'


def run_child(command, log, lock_fd):
    # Inherit the same flock open description: a surviving child keeps the lock
    # after a parent SIGKILL, so a restarted queue cannot resume over a live run.
    return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock_fd,))


def resource_check(data_root):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Shared CUDA resource unavailable; CPU training forbidden')
    for angle in ['10', '20', '30', '45', '50', '60', '70']:
        if not (data_root / angle / 'ToT').is_dir():
            raise RuntimeError(f'Shared data path unavailable: {data_root / angle / "ToT"}')
    if shutil.disk_usage(RUN_ROOT).free < 2 * 1024**3:
        raise RuntimeError('Shared output filesystem has less than 2 GiB free; no automatic deletion')


def prepare(data_root):
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    configurations = {}
    for item in plan():
        cfg = load_experiment_config(item['config'])
        cfg['dataset']['root'] = str(data_root)
        validate_experiment_config(cfg)
        configurations[item['id']] = cfg
        path = SUMMARY_ROOT / 'resolved_configs' / f"{item['id']}.yaml"
        path.parent.mkdir(exist_ok=True)
        serialized = yaml.safe_dump(clean_config(cfg), sort_keys=False, allow_unicode=True)
        if path.exists() and path.read_text(encoding='utf-8') != serialized:
            raise ValueError(f'Frozen resolved config differs: {path}')
        if not path.exists():
            path.write_text(serialized, encoding='utf-8')
    cfg = configurations['A42']
    normalizer, provenance = load_frozen_normalizer(
        resolve_project_path(cfg['data']['normalizer_metadata']), ['ToT'], cfg['normalization'],
        resolve_project_path(cfg['split']['path']), cfg['dataset'], cfg['data'])
    if provenance['split_sha256'] != SPLIT_SHA256 or provenance['metadata_sha256'] != METADATA_SHA256:
        raise ValueError('Original T7 split/baseline metadata identity mismatch')
    if (normalizer.stats['ToT'].mean != 637.2197424470774
            or normalizer.stats['ToT'].std != 1189.814612768605):
        raise ValueError('Unexpected original normalizer values')
    baseline = json.loads((BASELINE / 'metadata.json').read_text(encoding='utf-8'))
    if baseline['metrics'].get('test_evaluated', True) or not (BASELINE / 'validation_predictions.csv').is_file():
        raise ValueError('Baseline validation-only reference missing')
    atomic_json(SUMMARY_ROOT / 'protocol_provenance.json', dict(
        baseline=str(BASELINE), normalizer=provenance, training_seeds=[42, 43, 44], split_seed=42,
        split_counts=baseline['data_info']['split_counts'], test_evaluated=False,
        wall_clock_training_limit=None, baseline_initialization=False))
    return configurations


def execute(data_root, prepare_only=False, lock_fd=None):
    configs = prepare(data_root)
    if prepare_only:
        print('Six full configs resolved; fixed normalizer/reference verified. No training started.')
        return 0
    manifest_path = SUMMARY_ROOT / 'queue_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else dict(
        group=GROUP, created_at=now(), baseline=dict(id='R42', method='R', seed=42, run_dir=str(BASELINE), status='reused'),
        runs=[dict(**item, status='pending', run_dir=None, started_at=None, ended_at=None,
                   completed_epochs=0, stop_reason=None, split_sha256=SPLIT_SHA256) for item in plan()])
    if [r['id'] for r in manifest['runs']] != [r['id'] for r in plan()]:
        raise ValueError('Queue manifest does not match approved six-cell order')
    manifest.pop('blocked_reason', None)
    atomic_json(manifest_path, manifest)
    for item in manifest['runs']:
        cfg = configs[item['id']]
        if item['status'] == 'failed':
            continue
        path, completed = inspect_item(cfg, item)
        if item['status'] == 'failed':
            atomic_json(manifest_path, manifest)
            continue
        if completed:
            status, metrics = completed
            item.update(status='reused', run_dir=str(path), completed_epochs=metrics['stopped_epoch'],
                        stop_reason=f'Existing normal completion: {status}', ended_at=item['ended_at'] or now())
            atomic_json(manifest_path, manifest)
            continue
        try:
            resource_check(data_root)
        except RuntimeError as error:
            manifest['blocked_reason'] = str(error)
            atomic_json(manifest_path, manifest)
            print(f'BLOCKED: {error}', flush=True)
            return 2
        if path and not (path / 'last_checkpoint.pth').is_file():
            item.update(status='failed', run_dir=str(path), ended_at=now(),
                        stop_reason='Partial run has no compatible checkpoint; controller review required')
            atomic_json(manifest_path, manifest)
            continue
        command = [sys.executable, '-u', str(ROOT / 'scripts/train.py'), '--config',
                   str(SUMMARY_ROOT / 'resolved_configs' / f"{item['id']}.yaml")]
        if path:
            command += ['--resume', str(path / 'last_checkpoint.pth')]
        item.update(status='running', started_at=item['started_at'] or now(), command=command,
                    run_dir=str(path) if path else None)
        attempt = dict(started_at=now(), resumed=bool(path))
        item.setdefault('attempts', []).append(attempt)
        atomic_json(manifest_path, manifest)
        print(f"START {item['id']} {now()}", flush=True)
        log_path = SUMMARY_ROOT / f"{item['id']}.console.log"
        try:
            with log_path.open('a', encoding='utf-8') as log:
                # No timeout: the child exits only on its epoch/patience rule or an error.
                result = run_child(command, log, lock_fd)
        except KeyboardInterrupt:
            path = matching_runs(cfg)
            item.update(status='interrupted', run_dir=str(path) if path else None,
                        ended_at=now(), stop_reason='External interruption; resume same checkpoint')
            atomic_json(manifest_path, manifest)
            raise
        path, completed = inspect_item(cfg, item)
        attempt.update(ended_at=now(), exit_code=result.returncode)
        item.update(ended_at=now(), run_dir=str(path) if path else None)
        if item['status'] == 'failed':
            atomic_json(manifest_path, manifest)
            continue
        if result.returncode == 0 and completed:
            status, metrics = completed
            item.update(status=status, completed_epochs=metrics['stopped_epoch'],
                        stop_reason='patience=8' if status == 'early_stopped' else 'max_epochs=25')
        else:
            item.update(status=exit_status(result.returncode), stop_reason=f'Exit {result.returncode}; see {log_path.name}')
        atomic_json(manifest_path, manifest)
        print(f"END {item['id']} {item['status']} {now()}", flush=True)
    from scripts.summarize_carbon_difference import summarize
    summarize(SUMMARY_ROOT)
    manifest['finished_at'] = now()
    atomic_json(manifest_path, manifest)
    # Include the final queue completion marker in the lightweight package.
    from scripts.summarize_carbon_difference import package
    package(SUMMARY_ROOT, manifest)
    return 1 if any(r['status'] in {'failed', 'interrupted'} for r in manifest['runs']) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        return execute(args.data_root.resolve(), True)
    # OS-held lock is released after crashes, preventing duplicate experimenters.
    import fcntl
    with (SUMMARY_ROOT / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(args.data_root.resolve(), lock_fd=lock.fileno())


if __name__ == '__main__':
    raise SystemExit(main())
