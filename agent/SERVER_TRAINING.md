# 服务器持久化训练说明

VSCode Remote SSH 断开时，前台终端里的训练进程可能会一起结束。长时间训练建议同时使用两层保护：

1. 用 `tmux` 或 `nohup` 让训练进程不依赖 VSCode 终端。
2. 让代码每个 epoch 保存 `last_checkpoint.pth`，万一进程真的被杀，可以从最近完成的 epoch 恢复。

## 推荐方式：tmux

在服务器上：

```bash
cd ~/Timepix
tmux new -s timepix
```

进入 tmux 后运行训练：

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha
```

断开 tmux 但保持训练运行：

```text
Ctrl+b 然后按 d
```

重新进入：

```bash
tmux attach -t timepix
```

查看已有 tmux 会话：

```bash
tmux ls
```

训练时会显示当前 `Epoch x/y`，并分别显示 train/val batch 进度条。每个 epoch 结束后会输出一行摘要：

```text
Epoch summary | train_loss=... val_loss=... val_acc=... val_mae=... val_p90=...
```

## 备选方式：nohup

```bash
mkdir -p logs
PYTHONUNBUFFERED=1 nohup python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha \
  > logs/alpha_resnet18_tot.log 2>&1 &
```

查看日志：

```bash
tail -f logs/alpha_resnet18_tot.log
```

## Checkpoint 恢复

新系统默认每个 epoch 原子更新：

```text
last_checkpoint.pth
```

验证集指标刷新最好结果时，会同步更新：

```text
best_model.pth
```

例如：

```text
outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/last_checkpoint.pth
```

如果训练中断，可以恢复：

```bash
python scripts/train.py \
  --resume outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/last_checkpoint.pth
```

新的 checkpoint 里已经保存了训练配置，所以 `--resume` 可以单独使用。恢复时会继续使用原实验目录，并把 `training_log.csv` 截断到 checkpoint 对应 epoch 后再追加，避免重复 epoch 记录。

如果你要恢复旧 checkpoint，或训练数据移动到了新路径，可以显式加回配置和数据路径：

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha \
  --resume outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/last_checkpoint.pth
```

## 配置项

实验 YAML 中可以控制：

```yaml
training:
  progress_bar: true
  save_last_checkpoint: true
  mixed_precision: false
  mixed_precision_dtype: float16
```

进度条和 checkpoint 默认建议保持开启；混合精度先保持 `false` 作为 FP32 基准，确认精度损失可接受后再改成 `true`。

如果服务器 GPU 利用率高但训练仍然耗时，可以用 CUDA AMP 做速度对比：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml
```

开启 `training.mixed_precision: true` 后，训练、验证和测试都会使用 autocast，FP16 训练会使用 GradScaler。`last_checkpoint.pth` 会保存 scaler 状态，因此 AMP 实验也可以正常 `--resume`。汇总结果中的 `fit_seconds` 可以用于比较训练速度。

## 超参数搜索

代表性训练超参数搜索也建议放在 tmux 中运行：

```bash
python scripts/search_hparams.py \
  --config configs/search/a2_alpha_resnet18_tot_training.yaml
```

每个 trial 都会保存普通实验输出，Optuna study 默认写入 `outputs/optuna/` 的 SQLite 文件。VSCode 或 tmux 中断后，用同一条命令重新运行即可继续已有 study。

## 三 seed 认证

A2 最优训练超参可以用固定 `split.seed`、切换 `training.seed` 的 grid 认证：

```bash
python scripts/run_grid.py \
  --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml \
  --skip-existing \
  --continue-on-error
```

跑完后汇总并聚合：

```bash
python scripts/summarize.py --group a2_best_3seed --out outputs/a2_best_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a2_best_3seed_runs.csv --out outputs/a2_best_3seed_mean_std.csv
```

## B1-best tmux 示例

B1-best 是 `Proton_C_7` 最佳训练配置的三 seed 认证，建议使用专用 `tmux` 会话：

```bash
cd ~/Timepix
tmux new -s b1_best
```

进入会话后一次性运行完整链路：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_best_3seed --out outputs/b1_proton_c7_resnet18_tot_best_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b1_proton_c7_resnet18_tot_best_3seed_runs.csv --out outputs/b1_proton_c7_resnet18_tot_best_3seed_mean_std.csv
```

断开但保持训练运行：

```text
Ctrl+b 然后按 d
```

重新进入：

```bash
tmux attach -t b1_best
```

## A4c-4 tmux 示例

A4c-4 `warm_started_expert_gate` 属于长时间三 seed 对比实验，建议使用专用 `tmux` 会话：

```bash
cd ~/Timepix
tmux new -s a4c_warm_gate
```

进入会话后运行完整链路：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4c_warm_started_expert_gate --out outputs/a4c_warm_started_expert_gate_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a4c_warm_started_expert_gate_runs.csv --out outputs/a4c_warm_started_expert_gate_mean_std.csv
```

断开但保持训练运行：

```text
Ctrl+b 然后按 d
```

重新进入：

```bash
tmux attach -t a4c_warm_gate
```
