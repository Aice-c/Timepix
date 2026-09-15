#!/usr/bin/env python
"""One approved fresh lr1e-4/seed42 run, after controller reviews diagnosis."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from scripts.diagnose_carbon_stability import CONFIG, GROUP, OUTPUT, frozen_config
from scripts.run_carbon_difference_queue import atomic_json, clean_config, read_completed, now
from timepix.config_validation import validate_experiment_config


def run(data_root, approved):
    if not approved:
        raise ValueError('Controller must review diagnostic results before authorizing training')
    verification = json.loads((OUTPUT / 'diagnostic/verification.json').read_text(encoding='utf-8'))
    if (verification.get('status') != 'complete' or not verification.get('source_files_unchanged')
            or not verification.get('all_non_bn_state_preserved') or verification.get('test_evaluated')):
        raise ValueError('Diagnostic integrity check incomplete; stop for controller review')
    cfg, _ = frozen_config(data_root)
    validate_experiment_config(cfg)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_root = ROOT / 'outputs/experiments' / GROUP
    run_root.mkdir(parents=True, exist_ok=True)
    # New single-run driver is deliberately conservative: no automatic recovery,
    # no deletion of partial runs, no duplicate seed42 attempts.
    matches = sorted(run_root.glob(f"*_{cfg['experiment_name']}"))
    status_path = OUTPUT / 'training_status.json'
    if matches:
        if len(matches) == 1 and read_completed(matches[0]):
            old = yaml.safe_load((matches[0] / 'config.yaml').read_text(encoding='utf-8'))
            if clean_config(old) == clean_config(cfg):
                print(f'Already completed; no retraining: {matches[0]}')
                return 0
        raise RuntimeError('Existing partial/conflicting run requires controller review; not restarted')
    if status_path.exists():
        raise RuntimeError('A previous launch record exists; controller review required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training')
    if shutil.disk_usage(run_root).free < 5 * 1024**3:
        raise RuntimeError('Less than 5 GiB free; no automatic cleanup')
    resolved = OUTPUT / 'resolved_config.yaml'
    import fcntl
    with (OUTPUT / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Repeat checks while holding the process lock, inherited by the child.
        if status_path.exists() or list(run_root.glob(f"*_{cfg['experiment_name']}")):
            raise RuntimeError('Another launch won the lock; refusing duplicate')
        with resolved.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(clean_config(cfg), stream, sort_keys=False)
        status = dict(id='S42', started_at=now(), status='running',
            cancelled_same_time_lr3e4_repeat=True, training_seed=42, learning_rate=1e-4,
            selection_rule=cfg['task'], historical_reference_selection='Val MAE then Macro-F1',
            warning='LR and checkpoint/early-stop rule both changed; not LR-only causal comparison')
        atomic_json(status_path, status)
        command = [sys.executable, '-u', str(ROOT / 'scripts/train.py'), '--config', str(resolved),
                   '--data-root', str(data_root)]
        with (OUTPUT / 'training.console.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    pass_fds=(lock.fileno(),))
        matches = list(run_root.glob(f"*_{cfg['experiment_name']}"))
        status.update(ended_at=now(), exit_code=result.returncode,
                      run_dir=str(matches[0]) if len(matches) == 1 else None)
        complete = read_completed(matches[0]) if len(matches) == 1 else None
        status['status'] = 'complete' if result.returncode == 0 and complete else 'failed'
        atomic_json(status_path, status)
        if status['status'] != 'complete':
            return result.returncode or 2
        return subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_carbon_stability.py')], cwd=ROOT).returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--diagnostic-approved', action='store_true')
    args = parser.parse_args()
    raise SystemExit(run(args.data_root, args.diagnostic_approved))
