"""Opt-in strict lexicographic checkpoint selection."""
import math


def selection_key(metrics: dict, task: str, task_cfg: dict) -> tuple[float, ...]:
    names = [task_cfg.get('primary_metric', 'val_accuracy')] + list(task_cfg.get('tie_break_metrics', []))
    values = []
    for name in names:
        if name.startswith('test_'):
            raise ValueError('Checkpoint selection must not use test metrics')
        key = name.removeprefix('val_')
        if key not in metrics:
            raise ValueError(f'Missing checkpoint-selection metric: {name}')
        value = float(metrics[key])
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite checkpoint-selection metric: {name}')
        if any(token in key for token in ('mae', 'error', 'rmse', 'loss')):
            value = -value
        values.append(value)
    return tuple(values)
