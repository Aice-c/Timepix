"""
rebuild_summary.py —— 从各实验子目录恢复 all_results_summary.csv 和 all_results_detail.json

当 run_ablation.py 的汇总文件被单组重跑覆盖时，使用此脚本重新生成。

用法:
    python rebuild_summary.py
    python rebuild_summary.py --results_dir /path/to/output
"""

import os
import csv
import json
import yaml
import argparse
import numpy as np
from collections import OrderedDict

# 与 run_ablation.py 保持一致的映射关系
EXPERIMENT_META = OrderedDict({
    'A': {'name': 'ResNet+CE',    'dir_name': 'experiment_A_resnet_ce',    'task': 'classification'},
    'B': {'name': 'ResNet+EMD',   'dir_name': 'experiment_B_resnet_emd',   'task': 'classification'},
    'C': {'name': 'Shallow+CE',   'dir_name': 'experiment_C_shallow_ce',   'task': 'classification'},
    'D': {'name': 'Shallow+EMD',  'dir_name': 'experiment_D_shallow_emd',  'task': 'classification'},
    'E': {'name': 'ResNet+MSE',   'dir_name': 'experiment_E_resnet_reg',   'task': 'regression'},
    'F': {'name': 'Shallow+MSE',  'dir_name': 'experiment_F_shallow_reg',  'task': 'regression'},
})


def extract_from_training_log(log_path):
    """从 training_log.csv 提取最佳 epoch 的指标"""
    best_epoch = 0
    best_vacc = 0.0
    best_vmae_argmax = 999.0
    best_vmae_weighted = 999.0

    with open(log_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vacc = float(row.get('valid_acc', 0))
            vmae = float(row.get('valid_mae_argmax', 999))
            # 与 main.py 一致：以 valid_acc 为主判据
            if vacc > best_vacc or (vacc == best_vacc and vmae < best_vmae_argmax):
                best_vacc = vacc
                best_vmae_argmax = vmae
                best_vmae_weighted = float(row.get('valid_mae_weighted', 999))
                best_epoch = int(row.get('epoch', 0))

    return {
        'best_epoch': best_epoch,
        'best_vacc': best_vacc,
        'best_vmae_argmax': best_vmae_argmax,
        'best_vmae_weighted': best_vmae_weighted,
    }


def extract_from_test_predictions(npz_path):
    """从 test_predictions.npz 提取测试集指标"""
    data = np.load(npz_path, allow_pickle=True)
    true_labels = data['true_labels']
    pred_labels = data['pred_labels']
    angle_values = data['angle_values']

    # Accuracy
    test_acc = float(np.mean(true_labels == pred_labels))

    # MAE (argmax): 用类别索引映射到角度再算差
    true_angles = np.array([angle_values[i] for i in true_labels])
    pred_angles_mapped = np.array([angle_values[i] for i in pred_labels])
    test_mae_argmax = float(np.mean(np.abs(true_angles - pred_angles_mapped)))

    result = {
        'test_acc': test_acc,
        'test_mae_argmax': test_mae_argmax,
    }

    # 回归实验额外提取连续角度预测
    if 'pred_angles' in data and len(data['pred_angles']) > 0:
        pred_angles = data['pred_angles']
        true_angles_cont = data['true_angles']
        if len(pred_angles) > 0 and len(true_angles_cont) > 0:
            result['test_reg_mae'] = float(np.mean(np.abs(pred_angles - true_angles_cont)))

    return result


def extract_param_count(config_yaml_path):
    """从 config.yaml 提取参数量（如果保存了的话）"""
    try:
        with open(config_yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg.get('param_count', {}).get('total', 0)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser(description='从实验子目录恢复汇总文件')
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'ablation_results')
    parser.add_argument('--results_dir', type=str, default=default_dir,
                        help='实验结果根目录 (默认: Program/output/ablation_results/)')
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"错误: 目录不存在 {results_dir}")
        return

    summary_rows = []
    detail_dict = {}
    found = []
    missing = []

    for cfg_key, meta in EXPERIMENT_META.items():
        exp_dir = os.path.join(results_dir, meta['dir_name'])

        log_path = os.path.join(exp_dir, 'training_log.csv')
        npz_path = os.path.join(exp_dir, 'test_predictions.npz')
        cfg_path = os.path.join(exp_dir, 'config.yaml')

        if not os.path.isfile(log_path):
            missing.append(cfg_key)
            print(f"  ⚠ 配置 {cfg_key} ({meta['name']}): 未找到 training_log.csv，跳过")
            continue

        print(f"  ✓ 配置 {cfg_key} ({meta['name']}): 正在恢复...")

        # 从 training_log.csv 恢复验证指标
        train_info = extract_from_training_log(log_path)

        # 从 test_predictions.npz 恢复测试指标
        test_info = {}
        if os.path.isfile(npz_path):
            test_info = extract_from_test_predictions(npz_path)
            print(f"          Test Acc: {test_info['test_acc']:.4f}  MAE: {test_info['test_mae_argmax']:.2f}°")
        else:
            print(f"          ⚠ 无 test_predictions.npz，仅恢复验证指标")

        # 参数量
        params_total = extract_param_count(cfg_path)

        row = {
            'config': cfg_key,
            'name': meta['name'],
            'task': meta['task'],
            'best_epoch': train_info['best_epoch'],
            'valid_acc': round(train_info['best_vacc'], 4),
            'valid_mae': round(train_info['best_vmae_argmax'], 2),
            'test_acc': round(test_info.get('test_acc', 0), 4),
            'test_mae': round(test_info.get('test_mae_argmax', 0), 2),
            'params_total': params_total,
            'elapsed': 0,  # 原始耗时无法恢复
        }
        summary_rows.append(row)

        # 详细结果
        detail_dict[cfg_key] = {
            **train_info,
            **test_info,
            'param_count': {'total': params_total},
        }
        found.append(cfg_key)

    # 保存汇总 CSV
    if summary_rows:
        csv_path = os.path.join(results_dir, 'all_results_summary.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\n  ✓ 汇总 CSV 已保存: {csv_path}")

    # 保存详细 JSON
    if detail_dict:
        json_path = os.path.join(results_dir, 'all_results_detail.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detail_dict, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 详细 JSON 已保存: {json_path}")

    print(f"\n  恢复完成: {len(found)} 组成功 ({', '.join(found)})")
    if missing:
        print(f"  缺失: {len(missing)} 组 ({', '.join(missing)})")


if __name__ == '__main__':
    main()
