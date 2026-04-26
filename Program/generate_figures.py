"""
图表生成脚本 — 在本地运行，读取服务器回传的实验结果，生成全部 8 组学术图表。

用法:
    python generate_figures.py                              # 默认 output/ 目录
    python generate_figures.py --output_dir path/to/output  # 指定目录

前置条件:
    将服务器端的 output/ 目录完整拷回本地 Program/output/
"""

import argparse
import os
import sys
import csv
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from scipy.stats import gaussian_kde

# ═══════════════════════════════════════════════════
#  全局绑定格式设定
# ═══════════════════════════════════════════════════
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['figure.dpi'] = 300

# ═══════════════════════════════════════════════════
#  实验配置元信息
# ═══════════════════════════════════════════════════

ANGLE_LIST = [10, 20, 30, 45, 50, 60, 70, 80, 90]

EXPERIMENTS = {
    'A': {'name': 'ResNet+CE',      'dir': 'experiment_A_resnet_ce',     'color': '#4472C4', 'ls': '-',  'task': 'cls'},
    'B': {'name': 'ResNet+EMD',     'dir': 'experiment_B_resnet_emd',    'color': '#4472C4', 'ls': '--', 'task': 'cls'},
    'C': {'name': 'Shallow+CE',     'dir': 'experiment_C_shallow_ce',    'color': '#ED7D31', 'ls': '-',  'task': 'cls'},
    'D': {'name': 'Shallow+EMD',    'dir': 'experiment_D_shallow_emd',   'color': '#ED7D31', 'ls': '--', 'task': 'cls'},
    'E': {'name': 'ResNet+Reg',     'dir': 'experiment_E_resnet_reg',    'color': '#4472C4', 'ls': '-.', 'task': 'reg'},
    'F': {'name': 'Shallow+Reg',    'dir': 'experiment_F_shallow_reg',   'color': '#ED7D31', 'ls': '-.', 'task': 'reg'},
}

CLS_KEYS = ['A', 'B', 'C', 'D']
REG_KEYS = ['E', 'F']


def savefig(fig, fig_dir, name):
    """保存为 PDF 和 PNG 两种格式。"""
    fig.savefig(os.path.join(fig_dir, f'{name}.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, f'{name}.png'), bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  ✓ {name}.pdf / .png")


def load_training_log(exp_dir):
    """加载 training_log.csv。"""
    path = os.path.join(exp_dir, 'training_log.csv')
    if not os.path.isfile(path):
        return None
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_test_predictions(exp_dir):
    """加载 test_predictions.npz。"""
    path = os.path.join(exp_dir, 'test_predictions.npz')
    if not os.path.isfile(path):
        return None
    return dict(np.load(path, allow_pickle=True))


def load_summary_csv(output_dir):
    """加载 all_results_summary.csv。"""
    path = os.path.join(output_dir, 'all_results_summary.csv')
    if not os.path.isfile(path):
        return None
    rows = {}
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row['config']] = row
    return rows


# ═══════════════════════════════════════════════════
#  图表 1: 数据集分布条形图
# ═══════════════════════════════════════════════════

def plot_dataset_distribution(output_dir, fig_dir):
    """从任一实验的 test_predictions 中获取角度分布。"""
    # 尝试获取实际样本数据
    counts = {}
    for key, exp in EXPERIMENTS.items():
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))
        if data is not None:
            angle_values = data.get('angle_values', ANGLE_LIST)
            true_labels = data['true_labels']
            label_map_values = data.get('label_map_values', [str(a) for a in ANGLE_LIST])
            # 这里只有测试集的分布，需要推断总体
            for lbl in true_labels:
                angle = float(label_map_values[int(lbl)]) if int(lbl) < len(label_map_values) else lbl
                counts[angle] = counts.get(angle, 0) + 1
            break

    if not counts:
        print("  ⚠ 无法获取数据集分布信息，跳过图表 1")
        return

    angles = sorted(counts.keys())
    values = [counts[a] for a in angles]
    labels = [f"{int(a)}°" for a in angles]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, values, color='#4472C4', edgecolor='white', height=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{v}', va='center', ha='left', fontsize=11)
    ax.set_xlabel('Number of Samples (Test Set)')
    ax.set_ylabel('Incident Angle (°)')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0, max(values) * 1.15)
    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, fig_dir, 'dataset_distribution')


# ═══════════════════════════════════════════════════
#  图表 2: 训练曲线对比图
# ═══════════════════════════════════════════════════

def plot_training_curves(output_dir, fig_dir):
    logs = {}
    for key, exp in EXPERIMENTS.items():
        log = load_training_log(os.path.join(output_dir, exp['dir']))
        if log:
            logs[key] = log

    if not logs:
        print("  ⚠ 无训练日志，跳过图表 2")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上图: 训练损失
    ax1 = axes[0]
    for key in logs:
        exp = EXPERIMENTS[key]
        epochs = [r['epoch'] for r in logs[key]]
        losses = [r['train_loss'] for r in logs[key]]
        ax1.plot(epochs, losses, color=exp['color'], linestyle=exp['ls'], label=exp['name'], linewidth=1.5)
    ax1.set_ylabel('Training Loss')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3)

    # 下图: 验证指标
    ax2 = axes[1]
    ax2_right = ax2.twinx()
    
    for key in logs:
        exp = EXPERIMENTS[key]
        epochs = [r['epoch'] for r in logs[key]]
        if exp['task'] == 'cls':
            accs = [r['valid_acc'] for r in logs[key]]
            ax2.plot(epochs, accs, color=exp['color'], linestyle=exp['ls'], label=f"{exp['name']} (Acc)", linewidth=1.5)
        else:
            maes = [r['valid_mae_argmax'] for r in logs[key]]
            ax2_right.plot(epochs, maes, color=exp['color'], linestyle=exp['ls'], label=f"{exp['name']} (MAE)", linewidth=1.5)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2_right.set_ylabel('Validation MAE (°)')
    
    # 合并图例
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    savefig(fig, fig_dir, 'training_curves')


# ═══════════════════════════════════════════════════
#  图表 3: 混淆矩阵（分类实验 A/B/C/D）
# ═══════════════════════════════════════════════════

def plot_confusion_matrices(output_dir, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    titles = {
        'A': '(a) ResNet-18 + CE',
        'B': '(b) ResNet-18 + EMD',
        'C': '(c) ShallowResNet + CE',
        'D': '(d) ShallowResNet + EMD',
    }

    plotted = 0
    for idx, key in enumerate(CLS_KEYS):
        ax = axes[idx // 2][idx % 2]
        exp = EXPERIMENTS[key]
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))

        if data is None:
            ax.set_title(f"{titles[key]} (无数据)")
            ax.axis('off')
            continue

        true_labels = data['true_labels'].astype(int)
        pred_labels = data['pred_labels'].astype(int)
        label_map_values = [str(v) for v in data.get('label_map_values', ANGLE_LIST)]
        n_classes = len(label_map_values)
        tick_labels = [f"{v}°" for v in label_map_values]

        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n_classes)))
        # 归一化
        cm_norm = cm.astype(float)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums * 100

        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                    xticklabels=tick_labels, yticklabels=tick_labels,
                    vmin=0, vmax=100, cbar_kws={'label': '%'})
        ax.set_title(titles[key])
        ax.set_xlabel('Predicted Angle (°)')
        ax.set_ylabel('True Angle (°)')
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("  ⚠ 无分类实验数据，跳过图表 3")
        return

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    savefig(fig, fig_dir, 'confusion_matrices')


# ═══════════════════════════════════════════════════
#  图表 4: 回归实验散点图
# ═══════════════════════════════════════════════════

def plot_regression_scatter(output_dir, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = {
        'E': '(a) ResNet-18 + Regression',
        'F': '(b) ShallowResNet + Regression',
    }

    plotted = 0
    for idx, key in enumerate(REG_KEYS):
        ax = axes[idx]
        exp = EXPERIMENTS[key]
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))

        if data is None or 'true_angles' not in data or len(data['true_angles']) == 0:
            ax.set_title(f"{titles[key]} (无数据)")
            ax.axis('off')
            continue

        true_angles = data['true_angles'].astype(float)
        pred_angles = data['pred_angles'].astype(float)

        ax.scatter(true_angles, pred_angles, alpha=0.3, s=10, color='#4472C4', zorder=2)
        ax.plot([0, 100], [0, 100], 'r--', linewidth=1.5, label='Ideal', zorder=3)

        # 计算 MAE 和 R²
        mae = np.mean(np.abs(pred_angles - true_angles))
        ss_res = np.sum((true_angles - pred_angles) ** 2)
        ss_tot = np.sum((true_angles - np.mean(true_angles)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)

        textstr = f'MAE = {mae:.2f}°\n$R^2$ = {r2:.4f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

        ax.set_title(titles[key])
        ax.set_xlabel('True Angle (°)')
        ax.set_ylabel('Predicted Angle (°)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xticks(ANGLE_LIST)
        ax.set_yticks(ANGLE_LIST)
        ax.legend()
        ax.grid(alpha=0.3)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("  ⚠ 无回归实验数据，跳过图表 4")
        return

    fig.tight_layout()
    savefig(fig, fig_dir, 'regression_scatter')


# ═══════════════════════════════════════════════════
#  图表 5: 各角度 MAE 分段对比
# ═══════════════════════════════════════════════════

def plot_mae_by_angle(output_dir, fig_dir):
    """对每个真实角度，展示 6 种配置的 MAE。"""
    mae_data = {}  # {key: {angle: mae}}

    for key, exp in EXPERIMENTS.items():
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))
        if data is None:
            continue

        label_map_values = [str(v) for v in data.get('label_map_values', ANGLE_LIST)]
        
        if exp['task'] == 'reg':
            if 'true_angles' not in data or len(data['true_angles']) == 0:
                continue
            true_angles = data['true_angles'].astype(float)
            pred_angles = data['pred_angles'].astype(float)
            per_angle_mae = {}
            for angle_val in ANGLE_LIST:
                mask = np.abs(true_angles - angle_val) < 1.0
                if np.any(mask):
                    per_angle_mae[angle_val] = np.mean(np.abs(pred_angles[mask] - true_angles[mask]))
            mae_data[key] = per_angle_mae
        else:
            true_labels = data['true_labels'].astype(int)
            pred_labels = data['pred_labels'].astype(int)
            angle_values = np.array([float(v) for v in label_map_values])
            
            true_angles = angle_values[true_labels]
            pred_angles = angle_values[pred_labels]
            per_angle_mae = {}
            for i, angle_val in enumerate(ANGLE_LIST):
                mask = true_labels == i
                if np.any(mask):
                    per_angle_mae[angle_val] = np.mean(np.abs(pred_angles[mask] - true_angles[mask]))
            mae_data[key] = per_angle_mae

    if not mae_data:
        print("  ⚠ 无数据，跳过图表 5")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    n_configs = len(mae_data)
    bar_width = 0.8 / n_configs
    x = np.arange(len(ANGLE_LIST))

    for i, (key, per_angle) in enumerate(sorted(mae_data.items())):
        exp = EXPERIMENTS[key]
        values = [per_angle.get(a, 0) for a in ANGLE_LIST]
        offset = (i - n_configs / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=exp['name'], color=exp['color'],
               alpha=0.85 if exp['ls'] == '-' else 0.6,
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel('True Angle (°)')
    ax.set_ylabel('Mean Absolute Error (°)')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}°" for a in ANGLE_LIST])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    savefig(fig, fig_dir, 'mae_by_angle')


# ═══════════════════════════════════════════════════
#  图表 6: 综合性能对比表格图
# ═══════════════════════════════════════════════════

def plot_performance_comparison(output_dir, fig_dir):
    summary = load_summary_csv(output_dir)
    
    # 收集详细指标
    rows = []
    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        exp = EXPERIMENTS[key]
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))
        
        row = {'Config': key, 'Model': '', 'Loss': '', 'Acc(%)': '—', 'Macro-F1': '—', 'MAE(°)': '—', 'Params': '—'}
        
        if summary and key in summary:
            s = summary[key]
            row['Acc(%)'] = s.get('test_acc', '—')
            row['MAE(°)'] = s.get('test_mae', '—')
            row['Params'] = f"{int(float(s.get('params_total', 0))):,}"

        if exp['task'] == 'cls':
            row['Model'] = 'ResNet-18' if 'ResNet' in exp['name'] and 'Shallow' not in exp['name'] else 'ShallowResNet'
            row['Loss'] = 'CE' if 'CE' in exp['name'] else 'EMD'
            if data is not None:
                true = data['true_labels'].astype(int)
                pred = data['pred_labels'].astype(int)
                from sklearn.metrics import accuracy_score, f1_score as sk_f1
                row['Acc(%)'] = f"{accuracy_score(true, pred)*100:.1f}"
                row['Macro-F1'] = f"{sk_f1(true, pred, average='macro', zero_division=0):.3f}"
                
                # MAE weighted
                label_map_values = [float(v) for v in data.get('label_map_values', ANGLE_LIST)]
                true_ang = np.array([label_map_values[l] for l in true])
                pred_ang = np.array([label_map_values[l] for l in pred])
                row['MAE(°)'] = f"{np.mean(np.abs(pred_ang - true_ang)):.1f}"
        else:
            row['Model'] = 'ResNet-18' if 'ResNet' in exp['name'] and 'Shallow' not in exp['name'] else 'ShallowResNet'
            row['Loss'] = 'SmoothL1 (Reg)'
            if data is not None and 'true_angles' in data and len(data['true_angles']) > 0:
                true_ang = data['true_angles'].astype(float)
                pred_ang = data['pred_angles'].astype(float)
                row['MAE(°)'] = f"{np.mean(np.abs(pred_ang - true_ang)):.1f}"

        rows.append(row)

    if not rows:
        print("  ⚠ 无数据，跳过图表 6")
        return

    # 打印文本版
    print("\n  综合性能对比表:")
    header = f"  {'Config':>6s} | {'Model':>14s} | {'Loss':>14s} | {'Acc(%)':>7s} | {'Macro-F1':>8s} | {'MAE(°)':>7s} | {'Params':>12s}"
    print(header)
    print(f"  {'─'*len(header)}")
    for r in rows:
        print(f"  {r['Config']:>6s} | {r['Model']:>14s} | {r['Loss']:>14s} | {r['Acc(%)']:>7s} | {r['Macro-F1']:>8s} | {r['MAE(°)']:>7s} | {r['Params']:>12s}")

    # matplotlib 表格图
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis('off')
    
    col_labels = ['Config', 'Model', 'Loss', 'Acc(%)', 'Macro-F1', 'MAE(°)', 'Params']
    cell_text = [[r[c] for c in col_labels] for r in rows]
    
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    
    # 表头样式
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    fig.tight_layout()
    savefig(fig, fig_dir, 'performance_comparison')


# ═══════════════════════════════════════════════════
#  图表 7: 各角度 F1 对比折线图
# ═══════════════════════════════════════════════════

def plot_f1_by_angle(output_dir, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0

    for key in CLS_KEYS:
        exp = EXPERIMENTS[key]
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))
        if data is None:
            continue

        true_labels = data['true_labels'].astype(int)
        pred_labels = data['pred_labels'].astype(int)
        label_map_values = [str(v) for v in data.get('label_map_values', ANGLE_LIST)]
        n_classes = len(label_map_values)

        f1_per_class = np.asarray(f1_score(true_labels, pred_labels, labels=list(range(n_classes)),
                                average=None, zero_division=0))

        angles = [float(v) for v in label_map_values]
        ax.plot(angles, f1_per_class, color=exp['color'], linestyle=exp['ls'],
                marker='o', markersize=5, label=exp['name'], linewidth=1.5)
        for a, f in zip(angles, f1_per_class):
            ax.annotate(f'{f:.2f}', (a, f), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=8)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("  ⚠ 无分类实验数据，跳过图表 7")
        return

    ax.set_xlabel('Incident Angle (°)')
    ax.set_ylabel('F1-score')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(ANGLE_LIST)
    ax.set_xticklabels([f"{a}°" for a in ANGLE_LIST])
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    savefig(fig, fig_dir, 'f1_by_angle')


# ═══════════════════════════════════════════════════
#  图表 8: 回归误差分布直方图
# ═══════════════════════════════════════════════════

def plot_regression_error_distribution(output_dir, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = {
        'E': '(a) ResNet-18 + Regression',
        'F': '(b) ShallowResNet + Regression',
    }

    plotted = 0
    for idx, key in enumerate(REG_KEYS):
        ax = axes[idx]
        exp = EXPERIMENTS[key]
        data = load_test_predictions(os.path.join(output_dir, exp['dir']))

        if data is None or 'true_angles' not in data or len(data['true_angles']) == 0:
            ax.set_title(f"{titles[key]} (无数据)")
            ax.axis('off')
            continue

        true_angles = data['true_angles'].astype(float)
        pred_angles = data['pred_angles'].astype(float)
        errors = pred_angles - true_angles

        ax.hist(errors, bins=50, density=True, alpha=0.7, color='#4472C4', edgecolor='white')
        
        # KDE
        if len(errors) > 10:
            try:
                kde = gaussian_kde(errors)
                x_range = np.linspace(errors.min() - 5, errors.max() + 5, 200)
                ax.plot(x_range, kde(x_range), color='#E74C3C', linewidth=1.5, label='KDE')
            except Exception:
                pass

        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        textstr = f'Mean = {mean_err:.2f}°\nStd = {std_err:.2f}°'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        ax.set_title(titles[key])
        ax.set_xlabel('Prediction Error (°)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(alpha=0.3)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("  ⚠ 无回归实验数据，跳过图表 8")
        return

    fig.tight_layout()
    savefig(fig, fig_dir, 'regression_error_distribution')


# ═══════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='生成消融实验图表')
    parser.add_argument('--output_dir', type=str, default=None, help='输出根目录')
    args = parser.parse_args()

    # 定位 ablation_results 目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, 'output', 'ablation_results')

    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  图表生成")
    print(f"  输入目录: {output_dir}")
    print(f"  图表目录: {fig_dir}")
    print(f"{'='*60}\n")

    # 检查实验目录
    for key, exp in EXPERIMENTS.items():
        exp_path = os.path.join(output_dir, exp['dir'])
        exists = os.path.isdir(exp_path)
        print(f"  {'✓' if exists else '✗'} {key}: {exp['dir']}/" + ('' if exists else ' (未找到)'))

    print()

    plot_dataset_distribution(output_dir, fig_dir)
    plot_training_curves(output_dir, fig_dir)
    plot_confusion_matrices(output_dir, fig_dir)
    plot_regression_scatter(output_dir, fig_dir)
    plot_mae_by_angle(output_dir, fig_dir)
    plot_performance_comparison(output_dir, fig_dir)
    plot_f1_by_angle(output_dir, fig_dir)
    plot_regression_error_distribution(output_dir, fig_dir)

    print(f"\n{'='*60}")
    print(f"  全部图表生成完成")
    print(f"  保存位置: {fig_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
