#!/usr/bin/env python
"""One synthetic forward/backward batch per new model; never trains a run."""
import argparse
import gc
import hashlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from timepix.config import load_experiment_config
from timepix.models import build_model
from timepix.models.difference import DifferenceConv2d, LAYER1_PATHS
from timepix.utils.seed import set_seed
from scripts.run_carbon_difference_queue import atomic_json, SUMMARY_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SUMMARY_ROOT / 'new_model_check.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no training or CPU fallback')
    checks = []
    for kind in ['cdc', 'apdc']:
        set_seed(42)
        cfg = load_experiment_config(ROOT / f'configs/experiments/carbon_t7_{kind}_layer1_theta07_seed42.yaml')
        model = build_model(cfg, 1, 7, 'classification', 0).cuda()
        digest = hashlib.sha256()
        for key, value in model.state_dict().items():
            digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        locations = [key for key, m in model.named_modules() if isinstance(m, DifferenceConv2d)]
        assert locations == list(LAYER1_PATHS)
        assert sum(p.numel() for p in model.parameters()) == 11433863
        shapes = {'input': [2, 1, 50, 50]}
        handles = [getattr(model.backbone.model, name).register_forward_hook(
            lambda m, args, out, name=name: shapes.update({name: list(out.shape)}))
                   for name in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']]
        x = torch.randn(2, 1, 50, 50, device='cuda', requires_grad=True)
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(x).logits
            loss = F.cross_entropy(logits, torch.tensor([0, 6], device='cuda'))
        loss.backward()
        assert list(logits.shape) == [2, 7]
        assert [shapes[n][-1] for n in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']] == [49, 49, 25, 13, 7, 1]
        assert torch.isfinite(logits).all() and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(dict(operator=kind, theta=.7, paths=locations, shapes=shapes,
                           output=list(logits.shape), param_count=11433863,
                           initial_raw_state_sha256=digest.hexdigest(), finite_backward=True, optimizer_steps=0))
        for handle in handles:
            handle.remove()
        del model, x, logits, loss
        gc.collect(); torch.cuda.empty_cache()
    assert checks[0]['initial_raw_state_sha256'] == checks[1]['initial_raw_state_sha256']
    atomic_json(args.output, dict(device=torch.cuda.get_device_name(), checks=checks, training_started=False))
    print(f'Two new models passed one-batch checks: {args.output}')


if __name__ == '__main__':
    main()
