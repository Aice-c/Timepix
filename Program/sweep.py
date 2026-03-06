"""
Optuna 自动超参搜索脚本。

用法:
    python sweep.py                      # 默认搜索 30 次
    python sweep.py --n_trials 50        # 搜索 50 次
    python sweep.py --study_name my_exp  # 指定实验名称
"""

import argparse
import optuna
from main import run_experiment


def objective(trial: optuna.Trial) -> float:
    """Optuna 目标函数：采样超参 → 训练 → 返回验证准确率。"""

    overrides = {
        # 学习率：对数均匀采样
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),

        # 权重衰减
        'weight_decay': trial.suggest_float('weight_decay', 0, 1e-2),

        # Dropout
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.05),

        # Batch size
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),

        # 卷积核大小
        'kernel_size': trial.suggest_int('kernel_size', 2, 5),

        # 卷积特征大小
        'cnn_feature_size': trial.suggest_categorical('cnn_feature_size', [64, 128, 256, 512]),

        # 输出特征大小
        'out_feature_size': trial.suggest_categorical('out_feature_size', [128, 256, 512, 1024]),

        # 模型选择（取消注释以加入搜索空间）
        # 'model_name': trial.suggest_categorical('model_name', [
        #     'Resnet18', 'Resnet34', 'Resnet50', 'Densenet201',
        #     'Efficientnetb0', 'Shufflenet', 'CNN',
        # ]),
    }

    # sweep 时关闭绘图以加速
    best_vacc = run_experiment(overrides=overrides, save_plots=False)
    return best_vacc


def main():
    parser = argparse.ArgumentParser(description='Optuna 超参搜索')
    parser.add_argument('--n_trials', type=int, default=30, help='搜索次数')
    parser.add_argument('--study_name', type=str, default='timepix_sweep', help='实验名称')
    args = parser.parse_args()

    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',              # 最大化验证准确率
        storage=None,                      # 内存存储；换成 sqlite:///sweep.db 可持久化
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    # ---- 输出搜索结果汇总 ----
    print('\n' + '=' * 60)
    print('搜索完成！')
    print(f'最佳验证准确率: {study.best_value:.5f}')
    print(f'最佳超参组合:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    print('=' * 60)

    # 保存所有 trial 到 CSV
    df = study.trials_dataframe()
    df.to_csv('output/sweep_results.csv', index=False)
    print('所有 trial 结果已保存到 output/sweep_results.csv')


if __name__ == '__main__':
    main()
