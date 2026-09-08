#!/usr/bin/env python
"""Check frozen carbon configs, manifests and model shapes; never train a model."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from timepix.config import load_experiment_config, resolve_project_path
from timepix.config_validation import validate_experiment_config
from timepix.data.dataset import collect_samples
from timepix.data.frame_groups import validate_group_manifest
from timepix.models import build_model
from scripts.prepare_carbon_controls import immutable_text, json_text

CONFIGS = ['carbon_t7_tot_seed42', 'carbon_t7_mask_seed42', 'carbon_v6_base_seed42', 'carbon_v6_hires_seed42']


def pair_protocol(cfg):
    cell = cfg['evaluation']['experiment_id']
    expected = {'T7-ToT': ('T7', 'signal', False), 'T7-Mask': ('T7', 'hit_mask', False),
                'V6-Base': ('V6', 'signal', False), 'V6-HiRes': ('V6', 'signal', True)}
    actual = (cfg['evaluation']['task_id'], cfg['data'].get('input_representation', 'signal'),
              cfg['model'].get('preserve_late_resolution', False))
    if cell not in expected or actual != expected[cell]:
        raise ValueError(f'Incorrect cell-specific representation/stride: {cell}')
    if cfg['normalization']['ToT']['enabled'] != (actual[1] == 'signal'):
        raise ValueError(f'Incorrect cell normalization: {cell}')
    value = copy.deepcopy(cfg)
    for key in ('_config_path', '_config_dir', 'experiment_name'):
        value.pop(key, None)
    value['evaluation'].pop('experiment_id', None)
    if actual[0] == 'V6':
        value['model'].pop('preserve_late_resolution', None)
    else:
        value['data'].pop('input_representation', None)
        value['normalization']['ToT'].pop('enabled', None)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--backward', action='store_true', help='CUDA synthetic memory check only; no optimizer')
    parser.add_argument('--batch-size', type=int, help='Preflight both members with this proposed common batch')
    parser.add_argument('--output', default='outputs/carbon_angle_controls_20260909/local_preflight.json')
    args = parser.parse_args()
    if args.backward and args.device != 'cuda':
        parser.error('Backward memory check is CUDA-only; do not stress local CPU')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; no automatic CPU fallback')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    configs = [load_experiment_config(ROOT / f'configs/experiments/{name}.yaml') for name in CONFIGS]
    for a, b in [(0, 1), (2, 3)]:
        if pair_protocol(configs[a]) != pair_protocol(configs[b]):
            raise ValueError('Paired configs differ beyond controlled representation/stride switches')
    results = []
    manifest_cache = {}
    for cfg in configs:
        validate_experiment_config(cfg)
        task_id = cfg['evaluation']['task_id']
        if cfg['evaluation']['run_test'] or cfg['model']['pretrained'] or cfg['training'].get('resume_from'):
            raise ValueError('Protocol requires test disabled and fresh random initialization')
        if task_id not in manifest_cache:
            records, _ = collect_samples(args.data_root, ['ToT'], class_names=cfg['dataset']['class_names'])
            payload = json.loads(resolve_project_path(cfg['split']['path']).read_text(encoding='utf-8'))
            validate_group_manifest(payload, [r.key for r in records])
            if payload['class_names'] != cfg['dataset']['class_names'] or payload['task_id'] != task_id:
                raise ValueError('Manifest task/class order mismatch')
            # A few actual matrices validate dimensions without a full pixel rescan.
            import numpy as np
            for angle in cfg['dataset']['class_names']:
                r = next(r for r in records if r.angle == angle)
                array = np.loadtxt(r.modalities['ToT'])
                if array.shape != (50, 50) or not np.isfinite(array).all() or (array < 0).any():
                    raise ValueError(f'Unexpected probe matrix: {r.key}')
            manifest_cache[task_id] = payload
        batch = (args.batch_size or cfg['training']['batch_size']) if args.backward else 1
        model = build_model(cfg, 1, len(cfg['dataset']['class_names']), 'classification', 0).to(args.device)
        model.train(args.backward)
        x = torch.zeros(batch, 1, 50, 50, device=args.device)
        shapes = {'input': list(x.shape)}
        handles = []
        for key in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']:
            def capture(m, inputs, output, key=key):
                shapes[key] = list(output.shape)
            handles.append(getattr(model.backbone.model, key).register_forward_hook(capture))
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        if args.backward:
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = model(x).logits.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
        else:
            with torch.inference_mode():
                model(x)
        for h in handles:
            h.remove()
        results.append(dict(experiment_id=cfg['evaluation']['experiment_id'], shapes=shapes,
                            parameter_count=sum(p.numel() for p in model.parameters()),
                            manifest_sha256=manifest_cache[task_id]['manifest_sha256'],
                            peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                            configured_training_batch=cfg['training']['batch_size'], probe_batch=batch))
        del model, x
        if args.backward:
            del loss
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    if results[2]['parameter_count'] != results[3]['parameter_count']:
        raise ValueError('HiRes parameter count differs from Base')
    report = dict(device=args.device, torch=torch.__version__, optimizer_steps=0,
                  memory_check_only=args.backward, training_started=False, checks=results)
    immutable_text(resolve_project_path(args.output), json_text(report))
    print(json_text(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
