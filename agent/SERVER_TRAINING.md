# 服务器持久化训练说明

VSCode Remote SSH 断开时，前台终端里的训练进程可能会一起结束。为了长时间训练，建议同时使用两层保护：

1. 用 `tmux` 或 `nohup` 让训练进程不依赖 VSCode 终端。
2. 让代码每个 epoch 保存 `last_checkpoint.pth`，万一进程真的被杀，可以恢复。

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

## 备选方式：nohup

```bash
mkdir -p logs
nohup python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_Clean \
  > logs/alpha_resnet18_tot.log 2>&1 &
```

查看日志：

```bash
tail -f logs/alpha_resnet18_tot.log
```

## Checkpoint 恢复

新系统默认每个 epoch 保存：

```text
last_checkpoint.pth
```

例如：

```text
outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/last_checkpoint.pth
```

如果训练中断，可以恢复：

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --resume outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/last_checkpoint.pth
```

恢复时会继续使用原实验目录，并继续往原来的 `training_log.csv` 后面追加记录。

## 配置项

实验 YAML 中可以控制：

```yaml
training:
  progress_bar: true
  save_last_checkpoint: true
```

默认建议都保持开启。

