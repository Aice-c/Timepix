"""
消融实验主脚本 — 在服务器端运行。

用法:
    python run_ablation.py                          # 运行全部 6 组实验
    python run_ablation.py --configs A B            # 只运行指定配置
    python run_ablation.py --epochs 5 --quick       # 快速验证模式（少 epoch，小 batch）
    python run_ablation.py --data_dir /path/to/data # 指定数据目录

实验配置:
    A: ResNet-18 + CrossEntropy (分类)
    B: ResNet-18 + EMD/Wasserstein (分类)
    C: ShallowResNet + CrossEntropy (分类)
    D: ShallowResNet + EMD/Wasserstein (分类)
    E: ResNet-18 + SmoothL1 (回归)
    F: ShallowResNet + SmoothL1 (回归)
"""

import argparse
import os
import sys
import time
import json
import torch
import numpy as np
import traceback

from main import run_experiment
from Config import config

# ═══════════════════════════════════════════════════
#  6 种实验配置定义
# ═══════════════════════════════════════════════════

EXPERIMENT_CONFIGS = {
    'A': {
        'name': 'ResNet+CE',
        'description': 'ResNet-18 + CrossEntropy + one-hot',
        'overrides': {
            'model_name': 'Resnet18',
            'task': 'classification',
            'loss_type': 'cross_entropy',
            'label_encoding': 'onehot',
        },
        'dir_name': 'experiment_A_resnet_ce',
    },
    'B': {
        'name': 'ResNet+EMD',
        'description': 'ResNet-18 + Wasserstein(p=2) + 高斯软标签(σ=2.0)',
        'overrides': {
            'model_name': 'Resnet18',
            'task': 'classification',
            'loss_type': 'emd',
            'label_encoding': 'gaussian',
            'emd_p': 2,
            'gaussian_sigma': 2.0,
        },
        'dir_name': 'experiment_B_resnet_emd',
    },
    'C': {
        'name': 'Shallow+CE',
        'description': 'ShallowResNet + CrossEntropy + one-hot',
        'overrides': {
            'model_name': 'ShallowResNet',
            'task': 'classification',
            'loss_type': 'cross_entropy',
            'label_encoding': 'onehot',
        },
        'dir_name': 'experiment_C_shallow_ce',
    },
    'D': {
        'name': 'Shallow+EMD',
        'description': 'ShallowResNet + Wasserstein(p=2) + 高斯软标签(σ=2.0)',
        'overrides': {
            'model_name': 'ShallowResNet',
            'task': 'classification',
            'loss_type': 'emd',
            'label_encoding': 'gaussian',
            'emd_p': 2,
            'gaussian_sigma': 2.0,
        },
        'dir_name': 'experiment_D_shallow_emd',
    },
    'E': {
        'name': 'ResNet+MSE',
        'description': 'ResNet-18 + SmoothL1Loss (回归)',
        'overrides': {
            'model_name': 'Resnet18',
            'task': 'regression',
            'loss_type': 'smooth_l1',
            'label_encoding': 'onehot',
        },
        'dir_name': 'experiment_E_resnet_reg',
    },
    'F': {
        'name': 'Shallow+MSE',
        'description': 'ShallowResNet + SmoothL1Loss (回归)',
        'overrides': {
            'model_name': 'ShallowResNet',
            'task': 'regression',
            'loss_type': 'smooth_l1',
            'label_encoding': 'onehot',
        },
        'dir_name': 'experiment_F_shallow_reg',
    },
}

# ═══════════════════════════════════════════════════
#  统一训练超参数
# ═══════════════════════════════════════════════════

COMMON_OVERRIDES = {
    'epoch': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'early_stopping_patience': 10,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'random_seed': 42,
    'rotation': True,
    'dropout_rate': 0.1,
    'max_angle': 90.0,
    'modalities': ['ToT'],
    # 禁用手工特征（简化消融实验）
    'handcrafted_features': {
        'ToT': {'total_energy': False},
        'ToA': {'total_energy': False},
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description='Timepix3 消融实验')
    parser.add_argument('--configs', nargs='+', default=list(EXPERIMENT_CONFIGS.keys()),
                        help='要运行的实验配置, 如 A B C')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/full-angle',
                        help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出根目录 (默认: Program/output/)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖 epoch 数')
    parser.add_argument('--quick', action='store_true',
                        help='快速验证模式: 2 epochs, batch_size=128')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader 工作线程数')
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = args.output_dir or os.path.join(config.output_path, 'ablation_results')
    os.makedirs(output_root, exist_ok=True)

    # 数据划分索引文件路径（所有实验共享）
    split_indices_path = os.path.join(output_root, 'data_split_indices.pkl')

    # 快速模式
    if args.quick:
        COMMON_OVERRIDES['epoch'] = 2
        COMMON_OVERRIDES['batch_size'] = 128
        COMMON_OVERRIDES['early_stopping_patience'] = 0
        print("⚡ 快速验证模式: epoch=2, batch_size=128")

    if args.epochs:
        COMMON_OVERRIDES['epoch'] = args.epochs

    COMMON_OVERRIDES['data_dir'] = args.data_dir
    COMMON_OVERRIDES['num_workers'] = args.num_workers

    # 汇总结果
    all_results = {}
    failed = []

    configs_to_run = [c for c in args.configs if c in EXPERIMENT_CONFIGS]
    print(f"\n{'='*60}")
    print(f"  Timepix3 消融实验 — 共 {len(configs_to_run)} 组")
    print(f"  数据目录: {args.data_dir}")
    print(f"  输出目录: {output_root}")
    print(f"  Epoch: {COMMON_OVERRIDES['epoch']}")
    print(f"  配置: {', '.join(configs_to_run)}")
    print(f"{'='*60}\n")

    for idx, cfg_key in enumerate(configs_to_run):
        exp_cfg = EXPERIMENT_CONFIGS[cfg_key]
        exp_dir = os.path.join(output_root, exp_cfg['dir_name'])
        os.makedirs(exp_dir, exist_ok=True)

        print(f"\n{'═'*60}")
        print(f"  [{idx+1}/{len(configs_to_run)}] 实验 {cfg_key}: {exp_cfg['name']}")
        print(f"  {exp_cfg['description']}")
        print(f"  输出: {exp_dir}")
        print(f"{'═'*60}\n")

        # 合并覆盖参数
        overrides = {**COMMON_OVERRIDES, **exp_cfg['overrides']}
        overrides['_split_indices_path'] = split_indices_path

        # 分类任务不需要 weight，用 None 让 build_loss_function 自动处理
        if overrides.get('loss_type') == 'cross_entropy':
            overrides['weight'] = None  # 不使用固定权重

        start_time = time.time()
        try:
            result = run_experiment(
                overrides=overrides,
                save_plots=True,
                experiment_dir=exp_dir,
            )
            elapsed = time.time() - start_time
            result['elapsed_seconds'] = elapsed
            all_results[cfg_key] = result

            print(f"\n  ✓ 实验 {cfg_key} 完成 ({elapsed:.1f}s)")
            print(f"    Best Epoch: {result['best_epoch']}")
            print(f"    Valid Acc: {result['best_vacc']:.4f}")
            print(f"    Valid MAE: {result['best_vmae_argmax']:.2f}°")
            if 'test_acc' in result:
                print(f"    Test Acc: {result['test_acc']:.4f}")
                print(f"    Test MAE: {result['test_mae_argmax']:.2f}°")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n  ✗ 实验 {cfg_key} 失败 ({elapsed:.1f}s): {e}")
            traceback.print_exc()
            failed.append(cfg_key)

        # 释放 GPU 显存
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════
    #  汇总报告
    # ═══════════════════════════════════════════════════
    print(f"\n{'═'*60}")
    print("  实验汇总")
    print(f"{'═'*60}")

    summary_rows = []
    for cfg_key in configs_to_run:
        if cfg_key in all_results:
            r = all_results[cfg_key]
            exp_name = EXPERIMENT_CONFIGS[cfg_key]['name']
            row = {
                'config': cfg_key,
                'name': exp_name,
                'task': EXPERIMENT_CONFIGS[cfg_key]['overrides'].get('task', 'classification'),
                'best_epoch': r['best_epoch'],
                'valid_acc': round(r['best_vacc'], 4),
                'valid_mae': round(r['best_vmae_argmax'], 2),
                'test_acc': round(r.get('test_acc', 0), 4),
                'test_mae': round(r.get('test_mae_argmax', 0), 2),
                'params_total': r['param_count']['total'],
                'elapsed': round(r['elapsed_seconds'], 1),
            }
            summary_rows.append(row)
            print(f"  {cfg_key} ({exp_name:15s}): Acc={row['test_acc']:.4f}  MAE={row['test_mae']:.2f}°  "
                  f"Params={row['params_total']:,}  Time={row['elapsed']}s")

    if failed:
        print(f"\n  失败的实验: {', '.join(failed)}")

    # 保存汇总 CSV（合并已有结果，避免单组重跑覆盖全部）
    if summary_rows:
        import csv
        csv_path = os.path.join(output_root, 'all_results_summary.csv')
        # 读取已有的汇总
        existing_rows = {}
        if os.path.isfile(csv_path):
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing_rows[row['config']] = row
        # 用本轮结果覆盖对应 config
        for row in summary_rows:
            existing_rows[row['config']] = row
        # 按 config 字母排序写入
        merged = [existing_rows[k] for k in sorted(existing_rows.keys())]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(merged)
        print(f"\n  汇总表已保存到 {csv_path}")

    # 保存完整结果 JSON（合并已有结果）
    json_path = os.path.join(output_root, 'all_results_detail.json')
    existing_json = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing_json = json.load(f)
        except Exception:
            pass
    serializable = {}
    for k, v in all_results.items():
        row = {}
        for rk, rv in v.items():
            if isinstance(rv, (int, float, str, bool, type(None))):
                row[rk] = rv
            elif isinstance(rv, dict):
                row[rk] = {str(dk): dv for dk, dv in rv.items() if isinstance(dv, (int, float, str))}
            elif isinstance(rv, list) and len(rv) < 50:
                row[rk] = rv
        serializable[k] = row
    # 合并旧结果
    existing_json.update(serializable)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing_json, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*60}")
    print("  全部实验完成！")
    print(f"  请将 {output_root} 目录拷回本地进行图表生成和报告撰写。")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
