"""
综合分析报告生成脚本 — 在本地运行。

用法:
    python generate_report.py                              # 默认 output/ 目录
    python generate_report.py --output_dir path/to/output  # 指定目录
"""

import argparse
import os
import csv
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)

ANGLE_LIST = [10, 20, 30, 45, 50, 60, 70, 80, 90]

EXPERIMENTS = {
    'A': {'name': 'ResNet+CE',      'dir': 'experiment_A_resnet_ce',     'model': 'ResNet-18',      'loss': 'CrossEntropy', 'task': 'cls'},
    'B': {'name': 'ResNet+EMD',     'dir': 'experiment_B_resnet_emd',    'model': 'ResNet-18',      'loss': 'EMD(p=2)+Gaussian', 'task': 'cls'},
    'C': {'name': 'Shallow+CE',     'dir': 'experiment_C_shallow_ce',    'model': 'ShallowResNet',  'loss': 'CrossEntropy', 'task': 'cls'},
    'D': {'name': 'Shallow+EMD',    'dir': 'experiment_D_shallow_emd',   'model': 'ShallowResNet',  'loss': 'EMD(p=2)+Gaussian', 'task': 'cls'},
    'E': {'name': 'ResNet+Reg',     'dir': 'experiment_E_resnet_reg',    'model': 'ResNet-18',      'loss': 'SmoothL1(回归)', 'task': 'reg'},
    'F': {'name': 'Shallow+Reg',    'dir': 'experiment_F_shallow_reg',   'model': 'ShallowResNet',  'loss': 'SmoothL1(回归)', 'task': 'reg'},
}


def load_test_predictions(exp_dir):
    path = os.path.join(exp_dir, 'test_predictions.npz')
    if not os.path.isfile(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def load_config_yaml(exp_dir):
    path = os.path.join(exp_dir, 'config.yaml')
    if not os.path.isfile(path):
        return None
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_summary_csv(output_dir):
    path = os.path.join(output_dir, 'all_results_summary.csv')
    if not os.path.isfile(path):
        return {}
    rows = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows[row['config']] = row
    return rows


def load_training_log(exp_dir):
    path = os.path.join(exp_dir, 'training_log.csv')
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def section(title, level=1):
    if level == 1:
        return f"\n{'═'*60}\n  {title}\n{'═'*60}\n"
    elif level == 2:
        return f"\n{'─'*50}\n  {title}\n{'─'*50}\n"
    else:
        return f"\n  {title}\n  {'·'*40}\n"


def generate_report(output_dir):
    report_lines = []
    W = report_lines.append  # shorthand for write

    summary = load_summary_csv(output_dir)

    # ═══════════════════════════════════════════════════
    W("═" * 60)
    W("  Timepix3 带电粒子入射角度识别 — 消融实验报告")
    W("═" * 60)
    W(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    W("")

    # ─── 一、实验概述 ───
    W(section("一、实验概述"))
    W("  实验目的:")
    W("    在 9 类角度数据集 (10°~90°) 上，系统对比 6 种模型配置的性能，")
    W("    形成完整的消融实验 (Ablation Study)，评估不同损失函数和网络架构的影响。")
    W("")
    W("  6 种实验配置:")
    W(f"    {'编号':>4s}  {'简称':>12s}  {'模型':>14s}  {'损失函数':>18s}  {'任务类型'}")
    W(f"    {'─'*4}  {'─'*12}  {'─'*14}  {'─'*18}  {'─'*8}")
    for key, exp in EXPERIMENTS.items():
        W(f"    {key:>4s}  {exp['name']:>12s}  {exp['model']:>14s}  {exp['loss']:>18s}  {'分类' if exp['task']=='cls' else '回归'}")

    # ─── 二、数据集统计 ───
    W(section("二、数据集统计"))

    # 从实验结果中提取数据集信息
    any_data = None
    for key, exp in EXPERIMENTS.items():
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))
        if data is not None:
            any_data = data
            break

    if any_data is not None:
        label_map_values = [str(v) for v in any_data.get('label_map_values', ANGLE_LIST)]
        W(f"  角度类别数: {len(label_map_values)}")
        W(f"  角度列表: {', '.join([v+'°' for v in label_map_values])}")
    
    # 读取配置
    any_config = None
    for key, exp in EXPERIMENTS.items():
        cfg = load_config_yaml(os.path.join(output_dir, exp['dir']))
        if cfg:
            any_config = cfg
            break
    
    if any_config:
        W(f"\n  数据划分:")
        W(f"    训练集: {any_config.get('train_split', 0.8)*100:.0f}%")
        W(f"    验证集: {any_config.get('val_split', 0.1)*100:.0f}%")
        W(f"    测试集: {any_config.get('test_split', 0.1)*100:.0f}%")
        W(f"    随机种子: {any_config.get('random_seed', 42)}")
        W(f"\n  统一训练超参数:")
        W(f"    Epochs: {any_config.get('epoch', '?')}")
        W(f"    Batch size: {any_config.get('batch_size', '?')}")
        W(f"    Learning rate: {any_config.get('learning_rate', '?')}")
        W(f"    Weight decay: {any_config.get('weight_decay', '?')}")
        W(f"    Scheduler: {any_config.get('scheduler', '?')}")
        W(f"    Dropout: {any_config.get('dropout_rate', '?')}")

    # ─── 三、实验结果 ───
    W(section("三、实验结果"))

    # 3.1 综合性能对比表
    W(section("3.1 综合性能对比表", 2))
    W(f"  {'配置':>4s} | {'模型':>14s} | {'损失函数':>18s} | {'准确率(%)':>9s} | {'Macro-F1':>8s} | {'MAE(°)':>7s} | {'参数量':>12s}")
    W(f"  {'─'*4} | {'─'*14} | {'─'*18} | {'─'*9} | {'─'*8} | {'─'*7} | {'─'*12}")

    results_cache = {}  # 缓存各实验详细结果
    
    for key, exp in EXPERIMENTS.items():
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))
        acc_str, f1_str, mae_str, param_str = '—', '—', '—', '—'
        
        if summary and key in summary:
            param_str = f"{int(float(summary[key].get('params_total', 0))):,}"

        result = {}
        if data is not None:
            label_map_values = [float(v) for v in data.get('label_map_values', ANGLE_LIST)]
            
            if exp['task'] == 'cls':
                true = data['true_labels'].astype(int)
                pred = data['pred_labels'].astype(int)
                acc = accuracy_score(true, pred)
                macro_f1 = f1_score(true, pred, average='macro', zero_division=0)
                true_ang = np.array([label_map_values[l] for l in true])
                pred_ang = np.array([label_map_values[l] for l in pred])
                mae = np.mean(np.abs(pred_ang - true_ang))
                acc_str = f"{acc*100:.1f}"
                f1_str = f"{macro_f1:.3f}"
                mae_str = f"{mae:.1f}"
                result = {'acc': acc, 'macro_f1': macro_f1, 'mae': mae, 'true': true, 'pred': pred,
                          'true_ang': true_ang, 'pred_ang': pred_ang, 'label_map_values': label_map_values}
            else:
                if 'true_angles' in data and len(data['true_angles']) > 0:
                    true_ang = data['true_angles'].astype(float)
                    pred_ang = data['pred_angles'].astype(float)
                    mae = np.mean(np.abs(pred_ang - true_ang))
                    mae_str = f"{mae:.1f}"
                    # 等效分类准确率
                    pred_cls = np.array([ANGLE_LIST[np.argmin(np.abs(np.array(ANGLE_LIST) - p))] for p in pred_ang])
                    true_cls = np.array([ANGLE_LIST[np.argmin(np.abs(np.array(ANGLE_LIST) - t))] for t in true_ang])
                    equiv_acc = np.mean(pred_cls == true_cls)
                    acc_str = f"{equiv_acc*100:.1f}*"
                    result = {'mae': mae, 'equiv_acc': equiv_acc, 'true_ang': true_ang, 'pred_ang': pred_ang}

        results_cache[key] = result
        W(f"  {key:>4s} | {exp['model']:>14s} | {exp['loss']:>18s} | {acc_str:>9s} | {f1_str:>8s} | {mae_str:>7s} | {param_str:>12s}")

    W("\n  注: 标 * 的为回归方案的等效分类准确率（预测值四舍五入到最近类别）")

    # 3.2 分类实验详细结果
    W(section("3.2 分类实验详细结果", 2))
    
    for key in ['A', 'B', 'C', 'D']:
        exp = EXPERIMENTS[key]
        r = results_cache.get(key, {})
        W(f"\n  ▸ 配置 {key}: {exp['name']} ({exp['model']} + {exp['loss']})")
        
        log = load_training_log(os.path.join(output_dir, exp['dir']))
        if log:
            best_row = max(log, key=lambda x: x.get('valid_acc', 0))
            W(f"    最佳验证准确率: {best_row.get('valid_acc', 0)*100:.2f}% (Epoch {int(best_row.get('epoch', 0))})")

        if 'true' in r and 'pred' in r:
            true = r['true']
            pred = r['pred']
            label_map_values = r.get('label_map_values', ANGLE_LIST)
            n_classes = len(label_map_values)
            
            W(f"\n    Classification Report:")
            report = str(classification_report(true, pred, 
                                           target_names=[f"{int(v)}°" for v in label_map_values],
                                           zero_division=0))
            for line in report.split('\n'):
                W(f"      {line}")

            # 最容易混淆的角度对
            cm = confusion_matrix(true, pred, labels=list(range(n_classes)))
            np.fill_diagonal(cm, 0)
            if cm.max() > 0:
                i, j = np.unravel_index(cm.argmax(), cm.shape)
                W(f"\n    最容易混淆的角度对: {int(label_map_values[i])}° → {int(label_map_values[j])}° ({cm[i,j]} 次)")

    # 3.3 回归实验详细结果
    W(section("3.3 回归实验详细结果", 2))
    
    for key in ['E', 'F']:
        exp = EXPERIMENTS[key]
        r = results_cache.get(key, {})
        W(f"\n  ▸ 配置 {key}: {exp['name']} ({exp['model']} + {exp['loss']})")
        
        if 'true_ang' in r and 'pred_ang' in r:
            true_ang = r['true_ang']
            pred_ang = r['pred_ang']
            errors = pred_ang - true_ang
            
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((true_ang - np.mean(true_ang)) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            max_err = np.max(np.abs(errors))
            
            W(f"    MAE:  {mae:.2f}°")
            W(f"    RMSE: {rmse:.2f}°")
            W(f"    R²:   {r2:.4f}")
            W(f"    最大误差: {max_err:.2f}°")
            
            if 'equiv_acc' in r:
                W(f"    等效分类准确率: {r['equiv_acc']*100:.1f}%")
            
            W(f"\n    各角度分段 MAE:")
            W(f"      {'角度':>6s}  {'MAE(°)':>8s}  {'样本数':>6s}")
            for angle_val in ANGLE_LIST:
                mask = np.abs(true_ang - angle_val) < 1.0
                if np.any(mask):
                    seg_mae = np.mean(np.abs(pred_ang[mask] - true_ang[mask]))
                    W(f"      {angle_val:>5d}°  {seg_mae:>8.2f}  {int(np.sum(mask)):>6d}")

    # ─── 四、对比分析 ───
    W(section("四、对比分析"))

    # 4.1 损失函数的影响
    W(section("4.1 损失函数的影响", 2))
    for pair_label, k1, k2 in [("ResNet-18 下 CE vs EMD", 'A', 'B'), ("ShallowResNet 下 CE vs EMD", 'C', 'D')]:
        W(f"\n  ▸ {pair_label} (配置 {k1} vs {k2}):")
        r1, r2_data = results_cache.get(k1, {}), results_cache.get(k2, {})
        if 'acc' in r1 and 'acc' in r2_data:
            diff_acc = (r2_data['acc'] - r1['acc']) * 100
            diff_mae = r1['mae'] - r2_data['mae']
            W(f"    准确率: {r1['acc']*100:.1f}% → {r2_data['acc']*100:.1f}% (差异: {diff_acc:+.1f}%)")
            W(f"    MAE:    {r1['mae']:.1f}° → {r2_data['mae']:.1f}° (差异: {diff_mae:+.1f}°)")
            if diff_acc > 1:
                W(f"    → EMD 损失带来了显著改善")
            elif diff_acc < -1:
                W(f"    → CE 损失表现更好")
            else:
                W(f"    → 两者性能接近")

    # 4.2 网络架构的影响
    W(section("4.2 网络架构的影响", 2))
    for pair_label, k1, k2 in [("CE 下 ResNet-18 vs ShallowResNet", 'A', 'C'), ("EMD 下 ResNet-18 vs ShallowResNet", 'B', 'D')]:
        W(f"\n  ▸ {pair_label} (配置 {k1} vs {k2}):")
        r1, r2_data = results_cache.get(k1, {}), results_cache.get(k2, {})
        if 'acc' in r1 and 'acc' in r2_data:
            W(f"    准确率: {r1['acc']*100:.1f}% vs {r2_data['acc']*100:.1f}%")
            W(f"    MAE:    {r1['mae']:.1f}° vs {r2_data['mae']:.1f}°")
    
    if summary:
        a_params = int(float(summary.get('A', {}).get('params_total', 0)))
        c_params = int(float(summary.get('C', {}).get('params_total', 0)))
        if a_params > 0 and c_params > 0:
            W(f"\n  参数量对比:")
            W(f"    ResNet-18:      {a_params:,}")
            W(f"    ShallowResNet:  {c_params:,}")
            W(f"    参数量减少:     {(1-c_params/a_params)*100:.1f}%")

    # 4.3 分类 vs 回归
    W(section("4.3 分类 vs 回归", 2))
    cls_maes = {k: results_cache[k]['mae'] for k in ['A','B','C','D'] if 'mae' in results_cache.get(k, {})}
    reg_maes = {k: results_cache[k]['mae'] for k in ['E','F'] if 'mae' in results_cache.get(k, {})}
    if cls_maes and reg_maes:
        best_cls = min(cls_maes, key=lambda x: cls_maes[x])
        best_reg = min(reg_maes, key=lambda x: reg_maes[x])
        W(f"  分类最佳方案: 配置 {best_cls} ({EXPERIMENTS[best_cls]['name']}), MAE = {cls_maes[best_cls]:.2f}°")
        W(f"  回归最佳方案: 配置 {best_reg} ({EXPERIMENTS[best_reg]['name']}), MAE = {reg_maes[best_reg]:.2f}°")
        if cls_maes[best_cls] < reg_maes[best_reg]:
            W(f"  → 分类方案的 MAE 更低，更适合此问题")
        else:
            W(f"  → 回归方案的 MAE 更低，连续预测更有优势")

    # 4.4 近垂直角度的表现
    W(section("4.4 近垂直角度 (80°/90°) 的表现", 2))
    for key in ['A', 'B', 'C', 'D']:
        r = results_cache.get(key, {})
        if 'true' not in r:
            continue
        true = r['true']
        pred = r['pred']
        label_map_values = r.get('label_map_values', ANGLE_LIST)
        n_classes = len(label_map_values)
        
        # 找到 80° 和 90° 的索引
        idx_80 = None
        idx_90 = None
        for i, v in enumerate(label_map_values):
            if int(v) == 80: idx_80 = i
            if int(v) == 90: idx_90 = i
        
        if idx_80 is not None and idx_90 is not None:
            mask_80 = true == idx_80
            mask_90 = true == idx_90
            if np.any(mask_80) and np.any(mask_90):
                acc_80 = np.mean(pred[mask_80] == idx_80)
                acc_90 = np.mean(pred[mask_90] == idx_90)
                cross_80_90 = np.mean(pred[mask_80] == idx_90)
                cross_90_80 = np.mean(pred[mask_90] == idx_80)
                W(f"  配置 {key} ({EXPERIMENTS[key]['name']}): 80°准确率={acc_80*100:.1f}%, 90°准确率={acc_90*100:.1f}%, "
                  f"80°→90°误判={cross_80_90*100:.1f}%, 90°→80°误判={cross_90_80*100:.1f}%")

    # ─── 五、结论与建议 ───
    W(section("五、结论与建议"))
    
    W(section("5.1 主要发现", 2))
    all_maes = {**cls_maes, **reg_maes}
    if all_maes:
        best_overall = min(all_maes, key=lambda x: all_maes[x])
        W(f"  最佳配置: {best_overall} ({EXPERIMENTS[best_overall]['name']}), MAE = {all_maes[best_overall]:.2f}°")

    W(section("5.2 对论文撰写的建议", 2))
    W("  - 以综合性能对比表（表格 6）作为核心结果")
    W("  - 混淆矩阵和 F1-per-angle 图展示分类方案的角度分辨能力")
    W("  - 回归散点图展示连续预测的精度")
    W("  - 80°-90° 的分析作为角度分辨极限的讨论")

    W(section("5.3 后续可能的改进方向", 2))
    W("  - 引入 ToA 模态数据（多模态融合）")
    W("  - 数据增强策略（旋转之外的增强方法）")
    W("  - 集成学习（结合分类和回归的优势）")
    W("  - 更精细的角度划分（如 5° 间隔）")

    # ─── 六、附录 ───
    W(section("六、附录"))
    
    W(section("6.1 生成的图表文件列表", 2))
    fig_dir = os.path.join(output_dir, 'figures')
    if os.path.isdir(fig_dir):
        for f in sorted(os.listdir(fig_dir)):
            W(f"  - figures/{f}")

    # 写入文件
    report_text = '\n'.join(report_lines)
    report_path = os.path.join(output_dir, 'experiment_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n报告已保存到 {report_path}")
    return report_text


def main():
    parser = argparse.ArgumentParser(description='生成消融实验分析报告')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, 'output', 'ablation_results')

    generate_report(output_dir)


if __name__ == '__main__':
    main()
