# 新实验系统使用指南

本文档说明当前 Timepix 新实验系统的日常使用方式。旧版 `agent/NEW_SYSTEM_GUIDE.md` 因编码损坏已归档为 `agent/NEW_SYSTEM_GUIDE.old.md`；当前版只保留稳定、可执行、与最新实验状态一致的操作说明。

## 1. 基本原则

新实验系统以 YAML 配置和 CLI 命令为主，不再通过修改 legacy `Program/Config.py` 来定义实验。常规训练链路为：

```text
configs/experiments/*.yaml
  -> scripts/train.py 或 scripts/run_grid.py
  -> timepix/
  -> outputs/experiments/
```

新增对比实验时必须同步记录：

- 实验编号和阶段目的。
- 关键固定项与变量项。
- 服务器训练命令。
- 汇总命令。
- 三 seed 实验的 mean/std 聚合命令。
- validation-only 模型选择规则。

权威实验状态见 `agent/EXPERIMENT_LOG.md`；配置与命令索引见 `configs/README.md`。

## 2. 数据集与路径

当前训练主线：

| 数据集 | 用途 | 模态 |
| --- | --- | --- |
| `Alpha_100` | Alpha 正式训练主线 | `ToT`, `ToA` |
| `Alpha_50` | 历史/对照数据集 | `ToT`, `ToA` |
| `Proton_C_7` | Proton/C 七分类训练主线 | `ToT` |
| `Proton_C` | 论文数据分析与近垂直分辨极限分析 | `ToT` |

本地 Windows 路径：

```text
Alpha_100  -> D:\Project\Timepix\Data\Alpha_100
Proton_C   -> E:\C1Analysis\Proton_C
Proton_C_7 -> E:\C1Analysis\Proton_C_7
```

训练脚本的 `--data-root` 指向具体数据集目录：

```powershell
python scripts\run_grid.py --config configs\experiments\a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root D:\Project\Timepix\Data\Alpha_100 --dry-run
```

服务器命令使用 Linux 路径：

```bash
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run
```

数据分析脚本的 `--data-root` 指向父目录：

```powershell
python scripts\analyze_datasets.py --data-root E:\C1Analysis --datasets Proton_C
```

## 3. 单次训练

单次实验使用 `scripts/train.py`：

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_100
```

临时覆盖配置字段使用 `--set`：

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_100 \
  --set training.epochs=2 \
  --set training.batch_size=32
```

输出目录位于：

```text
outputs/experiments/<experiment_group>/<timestamp>_<experiment_name>/
```

单次 run 通常包含：

```text
config.yaml
metadata.json
metrics.json
training_log.csv
best_model.pth
last_checkpoint.pth
predictions.csv
confusion_matrix.csv
```

## 4. 网格实验

网格实验使用 `scripts/run_grid.py`：

```bash
python scripts/run_grid.py \
  --config configs/experiments/b3a_proton_c7_ordinal_loss_seed42.yaml \
  --data-root /root/autodl-tmp/Proton_C_7 \
  --dry-run
```

正式运行：

```bash
python scripts/run_grid.py \
  --config configs/experiments/b3a_proton_c7_ordinal_loss_seed42.yaml \
  --data-root /root/autodl-tmp/Proton_C_7 \
  --skip-existing \
  --continue-on-error
```

`--skip-existing` 用于跳过已经完成的 run，`--continue-on-error` 用于让某个 run 失败后继续执行后续组合。

## 5. 汇总与三 seed 聚合

按实验组汇总：

```bash
python scripts/summarize.py \
  --group b3a_proton_c7_ordinal_loss_seed42 \
  --out outputs/b3a_proton_c7_ordinal_loss_seed42_runs.csv
```

三 seed 聚合：

```bash
python scripts/aggregate_seeds.py \
  --summary outputs/b3b_proton_c7_expected_mae_3seed_runs.csv \
  --out outputs/b3b_proton_c7_expected_mae_3seed_mean_std.csv
```

A4b 冻结 expert 诊断脚本通常不写入标准 `outputs/experiments/<group>/`，需要显式指定 `--output-summary`、`--output-by-class`、`--output-json`，并使用 `scripts/aggregate_selector_fusion.py` 聚合。

## 6. 服务器持久化训练

长时间训练必须使用 `tmux`：

```bash
cd /root/Timepix
tmux new -s a6a
```

在 `tmux` 中运行：

```bash
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a6a_alpha_tot_ordinal_loss_seed42 --out outputs/a6a_alpha_tot_ordinal_loss_seed42_runs.csv
```

断开会话：

```text
Ctrl+b, then d
```

恢复会话：

```bash
tmux attach -t a6a
```

## 7. checkpoint 恢复

如果训练进程中断，可以从 `last_checkpoint.pth` 恢复：

```bash
python scripts/train.py \
  --resume outputs/experiments/<group>/<run>/last_checkpoint.pth
```

checkpoint 会保存训练配置、optimizer、scheduler、AMP scaler 和当前 epoch。若服务器数据路径变化，可显式附加 `--data-root`。

若需要把旧 run 复制到新实验组并继续到更长 epoch，使用：

```bash
python scripts/extend_runs.py \
  --source-group <source_group> \
  --target-group <target_group> \
  --target-epochs 25 \
  --data-root /root/autodl-tmp/Proton_C_7 \
  --skip-completed \
  --skip-early-stopped \
  --resume-target-existing \
  --continue-on-error
```

这种 continuation 结果应明确标注为中继训练，因为 cosine scheduler 的时间尺度可能与从头训练不同。

## 8. 当前常用实验命令

Alpha A6a ordered-loss screening：

```bash
cd /root/Timepix
tmux new -s a6a
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a6a_alpha_tot_ordinal_loss_seed42 --out outputs/a6a_alpha_tot_ordinal_loss_seed42_runs.csv
```

Proton B3b main：

```bash
cd /root/Timepix
tmux new -s b3b
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b3b_proton_c7_expected_mae_3seed --out outputs/b3b_proton_c7_expected_mae_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b3b_proton_c7_expected_mae_3seed_runs.csv --out outputs/b3b_proton_c7_expected_mae_3seed_mean_std.csv
```

A5d handcrafted gated verification：

```bash
cd /root/Timepix
tmux new -s a5d_gated
python scripts/run_grid.py --config configs/experiments/a5d_alpha_handcrafted_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a5d_alpha_handcrafted_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a5d_alpha_handcrafted_gated_3seed --out outputs/a5d_alpha_handcrafted_gated_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a5d_alpha_handcrafted_gated_3seed_runs.csv --out outputs/a5d_alpha_handcrafted_gated_3seed_mean_std.csv
```

## 9. 当前已收束和待推进事项

已收束：

- A4/A4b/A4c 多模态结构探索。
- A5 手工物理标量特征融合。
- B1/B2/B3 Proton_C_7 训练超参、手工特征和有序损失实验。

待推进：

- A6a 结果整理。
- 根据 A6a validation 结果撰写 A6b 三 seed 配置。
- 若 A6b 证明 best loss 有价值，再适配 A6c，将 best loss 迁移到 A4c-2 `dual_stream_gmu_aux`。

## 10. 选择与报告规则

- 模型、阈值、特征组、loss 参数和 gate 策略只能根据 validation 选择。
- test set 只用于最终泛化报告。
- 三 seed 结果报告 mean +/- std，不从单个 seed 中挑最高值作为正式结果。
- 若 test 诊断发现有趣现象，只能作为解释或后续假设，不可作为反向选择依据。
- 论文中选择 GMU 作为最终端到端多模态架构时，应强调 validation 侧 Macro-F1 接近最优、Val MAE 更好、稳定性和物理机制更合理，而不是使用 test 结果反选。
