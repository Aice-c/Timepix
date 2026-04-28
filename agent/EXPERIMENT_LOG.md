# Timepix 实验日志与配置说明

本文档记录当前实验序列、配置文件、关键结果和实验设计决策。训练命令默认按 Linux 服务器环境书写；只有本地检查才使用 Windows PowerShell 路径。

## 当前数据约定

- 标准数据集名称：`Alpha_100`、`Alpha_50`、`Proton_C`。
- 当前正式 Alpha 实验主线统一使用 `Alpha_100`；`Alpha_50` 曾短暂尝试，但效果不佳，不能支撑完整故事线，因此后续 A3/A4/A5/A6 默认回到 `Alpha_100`。
- `configs/experiments/alpha_resnet18_tot.yaml` 指向 `configs/datasets/alpha_100.yaml`；`configs/datasets/alpha.yaml` 仅作为兼容别名保留，也指向 `Alpha_100`。
- A1/A2 当时实际使用的 ToT split 已从本地 `outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json` 恢复，并确认与旧 `alpha_clean_ToT_seed42_0.8_0.1_0.1.json` 哈希一致。
- 服务器上这条历史 split 的规范文件名为 `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json`；所有继承 A2 best base、且基于 `Alpha_100 + ToT` 的实验应显式复用它。
- A2 best base 来自 `Alpha_100` 数据集的超参搜索结果，后续实验复用这组超参；当前不新增 `Alpha_50` 专用 A2 best base。
- `Alpha_100` 中 ToT 与 ToA 文件完全一一对应，split manifest 保存的是去掉 ToT/ToA 标记后的归一化 sample key。
- A4 的 ToT+ToA 双模态实验使用单独文件名的 paired split：`outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`，但该文件应从历史 `Alpha_100_ToT` split 复制得到，不重新随机生成。
- 默认 split 命名逻辑是 `dataset.name + modalities + split.seed + ratios`；如果显式提供 `split.path`，文件不存在时会按该路径创建。
- 如果 Alpha 数据内容发生变化，例如删除异常样本，需要删除或更换对应 split manifest 后再重跑相关实验。

## 随机性与划分策略

- `split.seed` 控制 train/validation/test 的分层划分。
- `training.seed` 控制模型初始化、DataLoader shuffle 和训练随机性。
- 正式对比实验应显式设置 `split.seed: 42`。
- 模型、模态、损失等对比应尽量复用同一 split。
- 多 seed 认证应报告 `mean ± std`，不应挑选三个 seed 中最高的结果作为正式结果。

## A2 Best Base

配置文件：

```text
configs/experiments/alpha_tot_a2_best_base.yaml
```

用途：

- 作为 A2 后续消融和对比实验的统一 base。
- 避免在 A3/A4/A5/A6 中手工复制超参数。

固定设置：

- Dataset: `Alpha_100`
- Modality: `ToT`
- Task: classification
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Fusion: `none`
- AMP: enabled, `float16`
- `split.seed=42`
- `split.path=outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json`
- `training.seed=42`
- `epochs=25`

A2 最佳超参数：

```text
learning_rate = 4.3878e-05
weight_decay  = 4.7324e-04
batch_size    = 32
eta_min       = 1.6433e-07
dropout       = 0.1
scheduler     = cosine
```

A2 最佳 trial 记录：

```text
trial          = 12
val accuracy   = 0.6953
test accuracy  = 0.7048
val MAE        = 6.279 deg
test MAE       = 5.964 deg
test macro-F1  = 0.6461
best epoch     = 24
```

## A1 结构适配实验

配置文件：

```text
configs/experiments/a1_resnet18_original_baseline.yaml
configs/experiments/a1_structure_adaptation.yaml
```

实验目的：

- 在 Alpha-ToT 单模态任务上确定 ResNet18 适配 Timepix 稀疏矩阵的结构。
- 比较原始 ResNet18、去除第一层 maxpool 的 ResNet18、保留 maxpool 的 ResNet18。

固定设置：

- Dataset: `Alpha_100`
- Modality: `ToT`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Seed: fixed

比较因素：

- 原始 ResNet18 baseline
- `resnet18_no_maxpool` vs `resnet18_maxpool`
- `conv1_kernel_size`: `2`, `3`, `5`
- `conv1_stride`: `1`, `2`
- `dropout`: `0.0`, `0.1`, `0.3`

决策备注：

- 原始 ResNet18 只作为 baseline，不参与结构网格搜索。
- 当时 A1 观察到 `resnet18_no_maxpool + kernel_size=2 + stride=1 + dropout=0.3` 表现最好。
- A2 训练超参搜索之后，统一 base 使用 `dropout=0.1`，因为 dropout 被纳入了训练超参数搜索。
- 本地拉取到的 A1 结果没有 grid manifest。A1 的 metadata 缺少 `command`、`git`、`environment` 字段，说明 A1 运行时使用的是较早 runtime；当时 manifest 和增强 metadata 还未生效，或 A1 进程启动后代码才更新。

服务器命令：

```bash
python scripts/train.py --config configs/experiments/a1_resnet18_original_baseline.yaml
python scripts/run_grid.py --config configs/experiments/a1_structure_adaptation.yaml
```

## AMP 对比实验

配置文件：

```text
configs/experiments/compare_mixed_precision.yaml
```

实验目的：

- 在当前最佳 ResNet18 结构上比较 FP32 与 CUDA AMP。

决策备注：

- AMP 对比结果显示混合精度有效，且没有明显降低准确率。
- 后续正式训练默认可以使用 `training.mixed_precision: true`。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml
```

## A2 训练超参数搜索

配置文件：

```text
configs/search/a2_alpha_resnet18_tot_training.yaml
```

实验目的：

- 固定 A1 得到的 ResNet18 结构后，搜索训练过程相关超参数。
- 将最优训练配置固定为后续消融和模型对比的默认训练设置。

固定设置：

- Dataset: `Alpha_100`
- Modality: `ToT`
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Scheduler: `cosine`
- Objective: `validation.accuracy`

搜索项：

```text
training.learning_rate
training.weight_decay
training.batch_size
training.eta_min
model.dropout
```

决策备注：

- 搜索目标只使用 validation 指标。
- test 指标只记录和报告，不用于选择超参数。
- A2 搜索结果已沉淀为 `alpha_tot_a2_best_base.yaml`。

服务器命令：

```bash
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml
```

## A2 Best 三 Seed 认证

配置文件：

```text
configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml
```

实验目的：

- 对 A2 最佳训练配置做三 seed 稳定性认证。
- 固定数据划分，只改变训练随机性。

当前设置：

```text
training.seed = 42, 43, 44
split.seed    = 42
split.path    = outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json
```

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml --continue-on-error
python scripts/summarize.py --group a2_best_3seed --out outputs/a2_best_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a2_best_3seed_runs.csv --out outputs/a2_best_3seed_mean_std.csv
```

决策备注：

- 增加该实验是因为单次训练结果存在随机波动。
- A3 曾因算力不足改为单 seed；后来单 seed 排名出现波动，因此 A3 改回三 seed 验证。

## A3 主干模型对比

配置文件：

```text
configs/experiments/a3_backbone_comparison.yaml
configs/experiments/a3_backbone_comparison_seed42.yaml
```

实验目的：

- 在 Alpha-ToT 单模态任务上比较不同模型主干。
- 选择后续多模态、手工特征和损失函数实验的主干。

固定设置：

- Base: `configs/experiments/alpha_tot_a2_best_base.yaml`
- Dataset: `Alpha_100`
- Modality: `ToT`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Fusion: `none`
- Training config: A2 best
- Split: `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json`
- Seed: `42`, `43`, `44`
- `epochs=25`

快速版：

- `a3_backbone_comparison_seed42.yaml` 继承完整 A3，仅固定 `training.seed=42`。
- 计划实验数为 7：7 个模型主干各跑一次。

比较主干：

```text
shallow_cnn
shallow_resnet
resnet18_no_maxpool
densenet121
efficientnet_b0
convnext_tiny
vit_tiny
```

决策备注：

- A3 初版使用单 seed，原因是当前算力不足。
- A3 单 seed 结果出现明显排名波动，ResNet18 变差、shallow 模型升高，因此 A3 改为三 seed 验证。
- ViT-Tiny 使用 `image_size=100`、`patch_size=10`，保持 `10x10=100` 个 patch token。
- 不将 ViT resize 到 `224x224`，避免改变公平比较条件。
- `model.dropout=0.1` 指统一 Timepix task head dropout。
- Torchvision backbone 内部默认正则保持模型默认，不在 A3 中单独调参。
- A3 早期观察显示 `resnet18_no_maxpool` 准确率最高。
- ViT-Tiny 预期不会很好，因为样本量有限、激活像素稀疏，CNN 的局部归纳偏置更适合该数据。
- A3 继承 A2 best base，因此 ToT 单模态比较显式复用恢复出的 `Alpha_100_ToT` 历史 split。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml --continue-on-error
python scripts/summarize.py --group a3_backbone_comparison --out outputs/a3_backbone_comparison_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a3_backbone_comparison_runs.csv --out outputs/a3_backbone_comparison_mean_std.csv
```

时间紧张时先跑 seed42 快速版：

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml --continue-on-error
```

当前结果记录（2026-04-29 用户汇报）：

- A3 支持 `resnet18_no_maxpool` 作为当前最佳主干模型。
- `resnet18_no_maxpool` 相比第二梯队高约 1.49 个百分点，同时 Test MAE 最低、Test Macro-F1 最高。

总体结果：

| Rank | Model | Test Acc | Val Acc | Test MAE | Test Macro-F1 | Params | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `resnet18_no_maxpool` | 70.48% | 69.53% | 5.96 deg | 0.646 | 11.43M | 10.0 min |
| 2 | `convnext_tiny` | 68.99% | 67.03% | 6.29 deg | 0.612 | 28.15M | 23.2 min |
| 3 | `shallow_resnet` | 68.99% | 67.03% | 6.26 deg | 0.634 | 1.34M | 4.1 min |
| 4 | `densenet121` | 68.69% | 67.73% | 6.40 deg | 0.610 | 7.34M | 24.8 min |
| 5 | `shallow_cnn` | 65.01% | 62.74% | 6.86 deg | 0.485 | 0.52M | 1.8 min |
| 6 | `efficientnet_b0` | 64.51% | 63.84% | 6.95 deg | 0.616 | 4.47M | 24.5 min |
| 7 | `vit_tiny` | 35.19% | 35.16% | 14.96 deg | 0.130 | 5.56M | 10.8 min |

每类 F1：

| Model | 15 deg F1 | 30 deg F1 | 45 deg F1 | 60 deg F1 |
| --- | ---: | ---: | ---: | ---: |
| `resnet18_no_maxpool` | 0.763 | 0.402 | 0.751 | 0.669 |
| `shallow_resnet` | 0.732 | 0.410 | 0.762 | 0.632 |
| `convnext_tiny` | 0.756 | 0.306 | 0.747 | 0.638 |
| `densenet121` | 0.749 | 0.315 | 0.751 | 0.623 |
| `efficientnet_b0` | 0.679 | 0.418 | 0.709 | 0.658 |

## A4 模态对比

配置文件：

```text
configs/experiments/a4_modality_comparison.yaml
configs/experiments/a4_modality_comparison_seed42.yaml
```

实验目的：

- 验证 `Alpha_100` 数据集中 ToT 与 ToA 对极角识别的贡献。
- 比较 ToT、ToA、ToT+ToA。

固定设置：

- Base: `configs/experiments/alpha_tot_a2_best_base.yaml`
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Fusion: `none`
- Training config: A2 best
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Seed: `42`, `43`, `44`

快速版：

- `a4_modality_comparison_seed42.yaml` 继承完整 A4，仅固定 `training.seed=42`。
- 计划实验数为 3：ToT+ToA、ToT、ToA 各跑一次。

比较模态：

```text
[ToT, ToA]
[ToT]
[ToA]
```

决策备注：

- `Alpha_100` 中 ToT 与 ToA 文件完全一一对应，ToT 单模态样本集合与 ToT+ToA 双模态样本集合一致。
- A4 paired split 从历史 ToT split 复制得到，保证 A4 与 A1/A2/A3 的数据划分严格一致；不让程序重新随机生成 A4 split。
- ToT-only、ToA-only 和 ToT+ToA 复用同一份 split 内容，保证样本集合和划分一致。
- A4 与 A3 一样改为三 seed 验证，避免单次训练波动影响模态结论。
- A4 split manifest:

```text
outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

- 服务器上创建 A4 paired split 的命令：

```bash
cp outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json \
   outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

- ToA normalization 使用 `log1p: true`、`ignore_zero: true`。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml --continue-on-error
python scripts/summarize.py --group a4_modality_comparison --out outputs/a4_modality_comparison_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4_modality_comparison_runs.csv --out outputs/a4_modality_comparison_mean_std.csv
```

时间紧张时先跑 seed42 快速版：

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml --continue-on-error
```

当前结果记录（2026-04-29 用户汇报）：

- 当前实现下，ToT 单模态最好。
- ToT+ToA 没有提升，测试准确率和误差指标均低于 ToT 单模态。
- ToA 单独效果低于 ToT，并且 ToA 对 30 deg 类别几乎失效。

总体结果：

| Modality | Val Acc | Test Acc | Test MAE | Test P90 | Test Macro-F1 | Best Epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ToT | 69.53% | 70.48% | 5.96 deg | 15 deg | 0.646 | 24 |
| ToT + ToA | 64.04% | 65.90% | 6.92 deg | 30 deg | 0.553 | 8 |
| ToA | 59.34% | 60.14% | 8.81 deg | 30 deg | 0.477 | 4 |

相对 ToT：

| Comparison | Test Acc Change | Test MAE Change | Macro-F1 Change |
| --- | ---: | ---: | ---: |
| ToT+ToA vs ToT | -4.57% | +0.95 deg | -0.093 |
| ToA vs ToT | -10.34% | +2.85 deg | -0.169 |

测试集每类 F1：

| Modality | 15 deg | 30 deg | 45 deg | 60 deg |
| --- | ---: | ---: | ---: | ---: |
| ToT | 0.763 | 0.402 | 0.751 | 0.669 |
| ToT + ToA | 0.733 | 0.178 | 0.730 | 0.572 |
| ToA | 0.675 | 0.000 | - | - |

## A4b ToA 表达方式对比

配置文件：

```text
configs/experiments/a4b_toa_transform.yaml
configs/experiments/a4b_toa_transform_seed42.yaml
```

实验目的：

- 在 A4 已确认 raw/log1p ToA early channel concat 不如 ToT 单模态的基础上，先检查 ToA 表达方式是否是主要问题。
- 不引入 dual-stream、GMU、FiLM 或 MMTM 等新模型结构，保持第一阶段只改数据表达，便于控制变量。

固定设置：

- Base: `configs/experiments/alpha_tot_a2_best_base.yaml`
- Dataset: `Alpha_100`
- Modalities: `[ToT, ToA]`
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Fusion: `none`
- Training config: A2 best
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- ToA normalization: `enabled: true`, `log1p: false`, `ignore_zero: true`

新增代码支持：

- `data.toa_transform`
  - `none`
  - `raw_log1p`
  - `relative_minmax`
  - `relative_centered`
  - `relative_rank`
- `data.add_hit_mask`
  - `false`: 输入通道为 `[ToT, transformed_ToA]`
  - `true`: 输入通道为 `[ToT, transformed_ToA, hit_mask]`

第一阶段网格：

```yaml
grid:
  data.toa_transform:
    - relative_minmax
    - relative_centered
    - relative_rank
  data.add_hit_mask:
    - false
    - true
  training.seed:
    - 42
    - 43
    - 44
```

决策备注：

- A4b 第一阶段不重复 A4 的 raw/log1p baseline；A4 已经提供 ToT、ToA、ToT+ToA raw/log1p 结果。
- 对 relative ToA 变换，配置中显式关闭 `normalization.ToA.log1p`，避免对相对时间再次做 log transform。
- `compute_normalizer` 与 `TimepixDataset` 共用同一套 ToA transform，保证训练输入和归一化统计一致。
- `add_hit_mask: true` 会让 `data_info.input_channels` 比 `dataset.modalities` 多 1，runner 已改为使用 `input_channels` 构建模型。
- `model.fusion_mode` 继续表示图像特征与手工特征融合；A4b 第一阶段不新增多模态模型融合语义。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --continue-on-error
python scripts/summarize.py --group a4b_toa_transform_seed42 --out outputs/a4b_toa_transform_seed42_runs.csv
```

若 seed42 结果值得继续，再运行三 seed 版本：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml --continue-on-error
python scripts/summarize.py --group a4b_toa_transform --out outputs/a4b_toa_transform_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4b_toa_transform_runs.csv --out outputs/a4b_toa_transform_mean_std.csv
```

当前结果记录（用户汇报）：

- 相对 ToA 表达相比 A4 的 raw/log1p ToT+ToA 有明显改善。
- 但所有 A4b-1 early fusion 变体仍未超过 ToT 单模态 baseline。
- `relative_centered, no mask` 的 Test Acc 最高，为 68.79%，但仍低于 ToT baseline 的 70.48%。
- `relative_minmax, no mask` 的 30 deg F1 最高，为 0.447，高于 ToT baseline 的 0.402；这说明 ToA 表达可能对 30 deg 类别有局部帮助，但总体验证/测试指标不足以支持替代 ToT。

| Experiment | Val Acc | Test Acc | Test MAE | Test P90 | Test Macro-F1 | 30 deg F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 ToT baseline | **69.53%** | **70.48%** | **5.96 deg** | **15 deg** | **0.646** | 0.402 |
| A4 ToT+ToA raw/log1p | 64.04% | 65.90% | 6.92 deg | 30 deg | 0.553 | 0.178 |
| A4b relative_centered, no mask | 67.83% | **68.79%** | **6.49 deg** | 30 deg | 0.612 | 0.290 |
| A4b relative_minmax, mask | 68.03% | 67.99% | 6.80 deg | 30 deg | 0.625 | 0.341 |
| A4b relative_centered, mask | 67.23% | 67.59% | 6.64 deg | 30 deg | 0.568 | 0.261 |
| A4b relative_rank, no mask | 66.53% | 67.20% | 6.80 deg | 30 deg | 0.617 | 0.362 |
| A4b relative_minmax, no mask | 67.13% | 67.10% | 6.86 deg | 30 deg | 0.635 | **0.447** |
| A4b relative_rank, mask | **69.23%** | 66.00% | 7.05 deg | 30 deg | 0.564 | 0.241 |

阶段性结论：

- A4b-1 支持“原始 ToA 表达方式不理想”这一判断。
- 但在当前 early fusion 框架下，改用相对 ToA 表达仍不能证明 ToA+ToT 优于 ToT。
- 后续如果继续使用 ToA，应优先考虑更保守的辅助方式，而不是简单 early channel concat。

## A4b-2.5 预测互补性诊断

脚本：

```text
scripts/analyze_prediction_complementarity.py
```

实验目的：

- 不训练新模型，只使用已有 `predictions.csv` 分析 ToA 或 ToT+ToA 是否在 ToT 出错样本上提供互补预测。
- 在继续实现 GMU/residual 等复杂融合前，先估计 ToA 可挖掘信息的上限。

默认输入：

- ToT baseline: `outputs/experiments/a4_modality_comparison_seed42`
- ToA baseline: `outputs/experiments/a4_modality_comparison_seed42`
- Relative ToT+ToA candidates: `outputs/experiments/a4b_toa_transform_seed42`

运行命令：

```bash
python scripts/analyze_prediction_complementarity.py --seed 42
```

输出：

```text
outputs/a4b_prediction_complementarity_seed42.json
outputs/a4b_prediction_complementarity_seed42_summary.csv
outputs/a4b_prediction_complementarity_seed42_by_class.csv
```

总体互补性结果：

| Comparator | Other Acc | ToT Wrong + Other Correct | Other Better When ToT Wrong | Oracle Acc | Oracle Gain | Oracle MAE | MAE Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ToA | 60.14% | 74 | 77 / 297 | 77.83% | +7.36% | 4.443 deg | +1.521 deg |
| relative_minmax, no mask | 67.10% | 111 | 125 / 297 | **81.51%** | **+11.03%** | **3.698 deg** | **+2.266 deg** |
| relative_minmax, mask | 67.99% | 81 | 84 / 297 | 78.53% | +8.05% | 4.309 deg | +1.655 deg |
| relative_centered, no mask | 68.79% | 82 | 84 / 297 | 78.63% | +8.15% | 4.324 deg | +1.640 deg |
| relative_centered, mask | 67.59% | 101 | 106 / 297 | 80.52% | +10.04% | 3.862 deg | +2.102 deg |
| relative_rank, no mask | 67.20% | 104 | 113 / 297 | 80.82% | +10.34% | 3.862 deg | +2.102 deg |
| relative_rank, mask | 66.00% | 72 | 75 / 297 | 77.63% | +7.16% | 4.488 deg | +1.476 deg |

30 deg 类别互补性结果：

| Comparator | ToT Acc | Other Acc | ToT Wrong + Other Correct | Oracle Acc | Oracle Gain | Other Better Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ToA | 29.66% | 0.00% | 0 | 29.66% | 0.00% | 0 |
| relative_minmax, no mask | 29.66% | 46.21% | 37 | **55.17%** | **+25.52%** | 40 |
| relative_minmax, mask | 29.66% | 26.90% | 12 | 37.93% | +8.28% | 12 |
| relative_centered, no mask | 29.66% | 20.00% | 8 | 35.17% | +5.52% | 8 |
| relative_centered, mask | 29.66% | 16.55% | 8 | 35.17% | +5.52% | 9 |
| relative_rank, no mask | 29.66% | 28.97% | 20 | 43.45% | +13.79% | 22 |
| relative_rank, mask | 29.66% | 15.86% | 4 | 32.41% | +2.76% | 4 |

阶段性结论：

- ToA 单模态虽然总体弱，但 oracle ToT/ToA 相比 ToT 有 +7.36% accuracy 上限和 1.52 deg MAE 上限改善，说明错误并非完全重叠。
- 原始 ToA 对 30 deg 没有补救能力；30 deg 的互补性主要来自相对 ToA early-fusion 候选，尤其是 `relative_minmax, no mask`。
- `relative_minmax, no mask` 的自身 Test Acc 不如 ToT，但 oracle 可达 81.51%，30 deg oracle 可达 55.17%，说明“什么时候信它”比“直接替代 ToT”更关键。
- 该诊断支持继续尝试 gated/residual/选择性融合，但不支持简单 early fusion 或固定 late fusion。

## A4b 阶段 2：Late Logit Fusion 评估

脚本：

```text
scripts/evaluate_logit_fusion.py
```

实验目的：

- 直接使用 A4 已训练完成的 ToT 与 ToA 单模态 checkpoint，评估 decision-level late fusion 是否能让弱 ToA 以较小权重补充 ToT。
- 不重新训练模型，不使用 test set 选择融合权重。

融合公式：

```text
logits = (1 - alpha_toa) * logits_tot + alpha_toa * logits_toa
```

默认搜索：

```text
alpha_toa = 0, 0.05, 0.10, 0.20, 0.30, 0.50
```

选择规则：

- 只在 validation set 上选择 `alpha_toa`。
- 主规则：最大 validation accuracy。
- 平局时依次使用更低 validation MAE、更高 validation macro-F1。
- 选定 `alpha_toa` 后再报告 test accuracy、test MAE、test P90、test macro-F1。

决策备注：

- 该阶段优先作为评估脚本实现，而不是新增训练模型；这样可以复用 A4 结果，快速判断 ToA 在 decision level 是否有补充信息。
- 脚本支持自动扫描实验组中同一 `training.seed` 的 `[ToT]` 与 `[ToA]` run，也支持手动传入两个 run 目录。
- 默认扫描 `a4_modality_comparison_seed42`，适配当前已经拉取到本地/服务器的 A4 seed42 结果；完整三 seed A4 完成后可改用 `--group a4_modality_comparison`。

服务器命令：

```bash
python scripts/evaluate_logit_fusion.py \
  --group a4_modality_comparison_seed42 \
  --output-csv outputs/a4b_late_logit_fusion_seed42.csv \
  --output-json outputs/a4b_late_logit_fusion_seed42.json
```

如果 A4 三 seed 结果存在：

```bash
python scripts/evaluate_logit_fusion.py \
  --group a4_modality_comparison \
  --output-csv outputs/a4b_late_logit_fusion_runs.csv \
  --output-json outputs/a4b_late_logit_fusion_runs.json
```

当前结果记录（用户汇报）：

- Late logit fusion 没有可靠证明 ToA 能提升效果。
- 验证集选择 `alpha_toa=0.00`，即完全不用 ToA。
- 虽然 test 上 `alpha_toa=0.30` 的 Test Acc 达到 70.97%，比 ToT baseline 高 0.50 个百分点，但该权重没有被 validation 选中，因此不能作为正式结论或调参依据。
- `alpha_toa=0.05` 的 Test Acc/MAE 也略有改善，但 validation accuracy 低于 `alpha_toa=0.00`，同样不能用 test 结果反向选择。

| alpha_toa | Selected by val | Val Acc | Test Acc | Test Acc Change | Test MAE | Test Macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00 | **yes** | **69.53%** | 70.48% | 0 | 5.96 deg | **0.646** |
| 0.05 | no | 69.23% | 70.68% | +0.20% | **5.93 deg** | 0.646 |
| 0.10 | no | 69.23% | 70.58% | +0.10% | 5.98 deg | 0.641 |
| 0.20 | no | 68.83% | 70.58% | +0.10% | 6.01 deg | 0.636 |
| 0.30 | no | 68.93% | **70.97%** | +0.50% | 5.95 deg | 0.636 |
| 0.50 | no | 67.53% | 69.38% | -1.09% | 6.38 deg | 0.599 |

阶段性结论：

- 按预先设定的验证集选择规则，late logit fusion 的最佳策略退化为 ToT-only。
- Test set 上个别非零 alpha 的小幅提升只能作为现象记录，不能用于选择模型或宣称 ToA 稳定提升。
- A4b-1 与 A4b-2 合起来说明：ToA 可能包含局部补充信息，尤其可能与 30 deg 类别有关，但当前 raw/relative early fusion 与 late logit fusion 都不足以支撑“ToA 带来可靠总体提升”的结论。

## 过渡或旧配置

- `configs/experiments/compare_models.yaml`: 早期主干对比配置。正式 A3 优先使用 `a3_backbone_comparison.yaml`。
- `configs/search/alpha_resnet18_tot_training.yaml`: 旧超参搜索配置。正式 A2 优先使用 `a2_alpha_resnet18_tot_training.yaml`。
- `configs/experiments/alpha_resnet18_tot.yaml`: baseline 模板。A2 后的新实验通常应继承 `alpha_tot_a2_best_base.yaml`。

## 常用汇总命令

A3:

```bash
python scripts/summarize.py --group a3_backbone_comparison --out outputs/a3_backbone_comparison_summary.csv
```

A4:

```bash
python scripts/summarize.py --group a4_modality_comparison --out outputs/a4_modality_comparison_summary.csv
```

全部实验：

```bash
python scripts/summarize.py --all --out outputs/experiment_summary.csv
```
