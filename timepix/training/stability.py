"""Opt-in, observational training diagnostics and train-only BN probes."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


def stratified_indices(records, per_class, seed):
    rng = np.random.default_rng(seed)
    chosen = []
    for label in sorted({r.label for r in records}):
        candidates = [i for i, r in enumerate(records) if r.label == label]
        chosen.extend(rng.choice(candidates, min(per_class, len(candidates)), replace=False).tolist())
    rng.shuffle(chosen)
    return chosen


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


class StepAudit:
    """Observe actual optimizer calls without unscaling/clipping/changing gradients."""
    def __init__(self, optimizer, path, epoch):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.stream = Path(path).open('a', encoding='utf-8')
        self.epoch = epoch
        self.batch = 0
        self.calls = 0
        self.hook = optimizer.register_step_post_hook(self._stepped)

    def _stepped(self, optimizer, args, kwargs):
        self.calls += 1

    def before_step(self, model, scaler):
        self.batch += 1
        self.before_calls = self.calls
        self.scaler = scaler
        scale = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.
        norms = [torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float64)
                 for p in model.parameters() if p.grad is not None]
        norm = float(torch.linalg.vector_norm(torch.stack(norms)).item()) / scale if norms else 0.
        self.row = dict(epoch=self.epoch, batch=self.batch, scale_before=scale,
                        gradient_l2_unscaled=_finite(norm), gradient_finite=math.isfinite(norm))

    def after_step(self):
        scaler = self.scaler
        self.row.update(optimizer_step_executed=self.calls > self.before_calls,
                        scale_after=float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.)
        self.stream.write(json.dumps(self.row, allow_nan=False) + '\n')
        if self.batch % 100 == 0:
            self.stream.flush()

    def close(self):
        self.hook.remove()
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def batchnorm_snapshot(model):
    result = {}
    for name, bn in model.named_modules():
        if isinstance(bn, nn.modules.batchnorm._BatchNorm) and bn.track_running_stats:
            mean, var = bn.running_mean.detach(), bn.running_var.detach()
            result[name] = dict(mean_min=_finite(mean.min()), mean_max=_finite(mean.max()),
                                var_min=_finite(var.min()), var_max=_finite(var.max()),
                                finite=bool(torch.isfinite(mean).all() and torch.isfinite(var).all()),
                                batches=int(bn.num_batches_tracked))
    return result


@torch.no_grad()
def reestimate_batchnorm(model, train_loader, device):
    """Caller supplies a copy and TRAIN-only records; dropout stays in eval mode.

    Cumulative batch-moment averaging is a sensitivity probe, not an exact pooled
    population variance estimator or a new formal checkpoint-selection strategy.
    """
    model.eval()
    norms = [(name, m) for name, m in model.named_modules()
             if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    if not norms:
        raise ValueError('No tracked BatchNorm layers to diagnose')
    buffers = {f'{name}.{suffix}' for name, _ in norms
               for suffix in ('running_mean', 'running_var', 'num_batches_tracked')}
    unchanged = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k not in buffers}
    old_momenta = [m.momentum for _, m in norms]
    batches = examples = 0
    # Even a nonshuffled DataLoader consumes RNG for its iterator base seed.
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            for _, bn in norms:
                bn.reset_running_stats()
                bn.momentum = None
                bn.train()
            for batch in train_loader:
                images = batch[0].to(device)
                model(images)
                batches += 1
                examples += len(images)
    finally:
        for (_, bn), momentum in zip(norms, old_momenta):
            bn.momentum = momentum
        model.eval()
    if batches == 0:
        raise ValueError('Empty train-only BN calibration subset')
    preserved = all(torch.equal(v, model.state_dict()[k].detach().cpu()) for k, v in unchanged.items())
    if not preserved:
        raise RuntimeError('BN probe changed non-BN state')
    return dict(batches=batches, examples=examples, non_bn_state_unchanged=preserved,
                method='FP32, train original views only, cumulative average of batch moments; dropout eval')
