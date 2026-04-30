# 服务器持久化训练指南

本文档说明在 Linux 服务器上运行 Timepix 训练任务时，如何避免 VSCode Remote SSH 断开导致训练中断，并在必要时从 checkpoint 恢复。旧版 `agent/SERVER_TRAINING.md` 因编码损坏已归档为 `agent/SERVER_TRAINING.old.md`。

## 1. 基本原则

长时间训练不应直接运行在 VSCode 前台终端中。推荐流程是：

1. 使用 `tmux` 创建持久会话。
2. 在 `tmux` 内运行训练、网格实验或超参数搜索。
3. 训练配置开启 `save_last_checkpoint`。
4. 每个对比实验同时生成 run summary；三 seed 实验还要生成 mean/std 聚合表。

当前服务器常用数据路径：

```bash
/root/autodl-tmp/Alpha_100
/root/autodl-tmp/Proton_C_7
```

全量 `Proton_C` 仅用于论文数据分析，不作为训练主线。

## 2. tmux 基本用法

创建会话：

```bash
cd /root/Timepix
tmux new -s timepix
```

断开但保持任务运行：

```text
Ctrl+b, then d
```

重新进入：

```bash
tmux attach -t timepix
```

查看会话：

```bash
tmux ls
```

## 3. 单次训练示例

Alpha_100 单次训练：

```bash
cd /root/Timepix
tmux new -s alpha_train

python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_100
```

Proton_C_7 单次训练：

```bash
cd /root/Timepix
tmux new -s proton_train

python scripts/train.py \
  --config configs/experiments/proton_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Proton_C_7
```

## 4. 网格实验标准写法

每个网格实验建议按“dry-run -> run -> summarize -> aggregate”的顺序写成一条链。

三 seed 训练示例：

```bash
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b3b_proton_c7_expected_mae_3seed --out outputs/b3b_proton_c7_expected_mae_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b3b_proton_c7_expected_mae_3seed_runs.csv --out outputs/b3b_proton_c7_expected_mae_3seed_mean_std.csv
```

单 seed screening 示例：

```bash
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a6a_alpha_tot_ordinal_loss_seed42 --out outputs/a6a_alpha_tot_ordinal_loss_seed42_runs.csv
```

## 5. checkpoint 恢复

新训练系统默认保存：

```text
last_checkpoint.pth
best_model.pth
```

如果进程中断，可从 `last_checkpoint.pth` 恢复：

```bash
python scripts/train.py \
  --resume outputs/experiments/<group>/<run>/last_checkpoint.pth
```

如果 checkpoint 中保存的历史路径已经不适配当前服务器，可显式补充 `--data-root`：

```bash
python scripts/train.py \
  --resume outputs/experiments/<group>/<run>/last_checkpoint.pth \
  --data-root /root/autodl-tmp/Alpha_100
```

恢复训练会继续写回原 run 目录，并截断/续写 `training_log.csv`，避免重复 epoch 记录。

## 6. 中继延长训练

如果某一批 run 已经完成但 epoch 不足，可以使用 `scripts/extend_runs.py` 将其复制到新实验组后继续训练。例如 B1-1 曾使用 from20 方式补跑到 25 epoch：

```bash
python scripts/extend_runs.py \
  --source-group b1_proton_c7_resnet18_tot_lr_batch \
  --target-group b1_proton_c7_resnet18_tot_lr_batch_ep25_from20 \
  --target-epochs 25 \
  --data-root /root/autodl-tmp/Proton_C_7 \
  --skip-completed \
  --skip-early-stopped \
  --resume-target-existing \
  --continue-on-error
```

这类结果必须标注为 continuation / from20，因为 cosine scheduler 的时间尺度可能与从头训练到相同 epoch 不完全等价。

## 7. AMP 与速度对比

AMP 由配置控制：

```yaml
training:
  mixed_precision: true
  mixed_precision_dtype: float16
```

AMP checkpoint 会保存 `GradScaler` 状态，因此可以正常 `--resume`。如果需要比较 FP32 与 AMP：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group compare_mixed_precision --out outputs/compare_mixed_precision_runs.csv
```

当前实验决策是：AMP 已证明有效且不明显降低准确率，后续正式训练可默认启用，除非某个配置出现异常速度或数值问题。

## 8. 当前常用 tmux 会话命名

建议使用具有实验编号的会话名：

```bash
tmux new -s a6a
tmux new -s a6b
tmux new -s b3b
tmux new -s a5d_gated
```

不要复用含义模糊的长期会话名保存多个不同实验，以免后续无法追踪任务来源。
