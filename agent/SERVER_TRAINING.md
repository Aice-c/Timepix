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
  --data-root /root/autodl-tmp/Alpha_Clean
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
  --data-root /root/autodl-tmp/Alpha_Clean \
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
  --data-root /root/autodl-tmp/Alpha_Clean \
  --resume outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/last_checkpoint.pth
```

## 配置项

实验 YAML 中可以控制：

```yaml
training:
  progress_bar: true
  save_last_checkpoint: true
```

默认建议都保持开启。
