# Timepix 实验日志与配置说明

本文档记录当前实验序列、配置文件、关键结果和实验设计决策。训练命令默认按 Linux 服务器环境书写；只有本地检查才使用 Windows PowerShell 路径。

## 当前数据约定

- 标准训练数据集名称：`Alpha_100`、`Alpha_50`、`Proton_C_7`。
- 标准数据分析数据集名称：`Alpha_100`、全量 `Proton_C`。
- `Proton_C_7` 是服务器上的正式质子/C 7 分类训练数据集名称，后续所有 Proton/C 训练只使用该数据集。
- `Proton_C` 保留为全量质子/C 数据集名称，只用于论文数据分析和近垂直分辨极限分析。
- `configs/datasets/proton_c.yaml` 仅作为旧入口兼容，内容也指向 `Proton_C_7`；新配置统一使用 `configs/datasets/proton_c_7.yaml`。
- 当前正式 Alpha 实验主线统一使用 `Alpha_100`；`Alpha_50` 曾短暂尝试，但效果不佳，不能支撑完整故事线，因此后续 A3/A4/A5/A6 默认回到 `Alpha_100`。
- `configs/experiments/alpha_resnet18_tot.yaml` 指向 `configs/datasets/alpha_100.yaml`；`configs/datasets/alpha.yaml` 仅作为兼容别名保留，也指向 `Alpha_100`。
- A1/A2 当时实际使用的 ToT split 已从本地 `outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json` 恢复，并确认与旧 `alpha_clean_ToT_seed42_0.8_0.1_0.1.json` 哈希一致。
- 服务器上这条历史 split 的规范文件名为 `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json`；所有继承 A2 best base、且基于 `Alpha_100 + ToT` 的实验应显式复用它。
- A2 best base 来自 `Alpha_100` 数据集的超参搜索结果，后续实验复用这组超参；当前不新增 `Alpha_50` 专用 A2 best base。
- `Alpha_100` 中 ToT 与 ToA 文件完全一一对应，split manifest 保存的是去掉 ToT/ToA 标记后的归一化 sample key。
- A4 的 ToT+ToA 双模态实验使用单独文件名的 paired split：`outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`，但该文件应从历史 `Alpha_100_ToT` split 复制得到，不重新随机生成。
- 默认 split 命名逻辑是 `dataset.name + modalities + split.seed + ratios`；如果显式提供 `split.path`，文件不存在时会按该路径创建。
- 如果 Alpha 数据内容发生变化，例如删除异常样本，需要删除或更换对应 split manifest 后再重跑相关实验。

## 本地验证环境与数据路径

本地 Windows 笔记本已新增独立 conda 环境：

```powershell
conda activate timepix-local
```

也可以直接调用解释器：

```powershell
& 'D:\Program\Anaconda\envs\timepix-local\python.exe' <script> ...
```

该环境用于本地验证、checkpoint 推理诊断、数据分析和论文图表生成，不作为正式训练环境。

本地数据路径：

```text
Alpha_100    -> D:\Project\Timepix\Data\Alpha_100
Proton_C     -> E:\C1Analysis\Proton_C
Proton_C_7   -> E:\C1Analysis\Proton_C_7
```

路径使用注意：

- 训练/评估脚本的 `--data-root` 覆盖的是具体数据集目录，例如 `D:\Project\Timepix\Data\Alpha_100` 或 `E:\C1Analysis\Proton_C_7`。
- 数据分析脚本的 `--data-root` 是包含数据集文件夹的父目录；Alpha 用 `D:\Project\Timepix\Data`，Proton 用 `E:\C1Analysis`。
- 由于 Alpha 与 Proton 位于不同盘符，`scripts/analyze_datasets.py --datasets Alpha_100 Proton_C` 不能直接用单个本地 `--data-root` 同时覆盖两者；本地可分别运行，或建立本地数据链接目录后再合并分析。
- 文档中的正式训练命令仍默认按 Linux 服务器环境书写；Windows 路径只用于本地检查和分析。

## 随机性与划分策略

- `split.seed` 控制 train/validation/test 的分层划分。
- `training.seed` 控制模型初始化、DataLoader shuffle 和训练随机性。
- 正式对比实验应显式设置 `split.seed: 42`。
- 模型、模态、损失等对比应尽量复用同一 split。
- 多 seed 认证应报告 `mean ± std`，不应挑选三个 seed 中最高的结果作为正式结果。

## 对比实验命令与汇总记录规范

- 从 2026-04-29 起，任何新对比实验、消融实验或诊断实验，只要写入配置或计划文档，就必须同时记录运行命令和汇总命令。
- 服务器正式训练/评估命令默认使用 Linux bash 写法；只有本地验证、数据分析或笔记本复现实验才使用 Windows PowerShell 写法。
- 单 seed 实验应至少给出 dry-run/正式运行命令，以及对应的 `scripts/summarize.py` 或诊断脚本输出 CSV 命令。
- 三 seed 实验应同时给出三 seed 运行命令、逐 run 汇总命令，以及 `scripts/aggregate_seeds.py` 或 `scripts/aggregate_selector_fusion.py` 的 mean/std 聚合命令。
- 不经过 `outputs/experiments/<group>/` 的诊断脚本也要明确输出文件，例如 summary CSV、by-class CSV、JSON、sample-level CSV 或 distribution CSV，避免只有运行入口而没有可追溯结果入口。
- 每次新增实验配置、修改实验编号、废弃旧结果或改变选择标准，都要同步更新本文档中的关键决策与命令块。

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

### A4b-3a/b oracle 控制诊断

新增脚本：

```text
scripts/evaluate_oracle_complementarity.py
```

实验目的：

- A4b-3a：做 `ToT-vs-ToT` 多 seed oracle control，判断 A4b-2.5 中较高 oracle 上限是否主要来自随机种子/模型多样性。
- A4b-3b：在 validation/test 上复查 `ToT` 与 `relative_minmax/no mask` candidate 的互补性，避免只根据 test-set oracle 做后续决策。

关键实现：

- 脚本重新加载已有 `best_model.pth`，而不是只读 test-only `predictions.csv`。
- 复用 run 目录中的 `config.yaml`、split manifest、normalization、ToA transform 和模型结构。
- 使用 `build_dataloaders(..., eval_mode=True)` 构造确定性推理 dataloader：train split 也不做 `rotation_90` 扩增，且不 shuffle。
- 对齐检查使用 sample key、label map 和 labels，不只依赖行号。
- 输出 summary CSV、by-class CSV 和 JSON。

服务器命令：

A4b-3a 纯 ToT 多 seed oracle control 使用 `a2_best_3seed`，而不是 A4 的 seed42 快速组。原因是 `a2_best_3seed` 是当前唯一已经完成的 ToT 三 seed 结果，并且固定了与 A3/A4 主线一致的 `Alpha_100 + ToT + resnet18_no_maxpool + A2 best` 架构和训练超参数：

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-seed-control \
  --tot-group a2_best_3seed \
  --splits val,test \
  --seeds 42 43 44 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --output-json outputs/a4b_3a_tot_seed_control.json \
  --output-summary outputs/a4b_3a_tot_seed_control_summary.csv \
  --output-by-class outputs/a4b_3a_tot_seed_control_by_class.csv
```

A4b-3b 先做 seed42 的 `ToT` vs `relative_minmax/no mask` 复查。这里 ToT 侧也优先从 `a2_best_3seed` 中取 seed42，使 ToT baseline 与 A4b-3a 控制实验来自同一三 seed 基准组；candidate 侧来自 `a4b_toa_transform_seed42`：

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-vs-candidate \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --splits val,test \
  --seeds 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_3b_tot_vs_relative_minmax.json \
  --output-summary outputs/a4b_3b_tot_vs_relative_minmax_summary.csv \
  --output-by-class outputs/a4b_3b_tot_vs_relative_minmax_by_class.csv
```

决策备注：

- A4b-3a 的正式输入组改为 `a2_best_3seed`。该组比 `a4_modality_comparison_seed42` 更适合做 seed control，因为它包含 seed 42/43/44 的纯 ToT 结果。
- A4b-3b 目前只对 seed42 做 `ToT` vs `relative_minmax/no mask` 复查；如果后续补跑 A4b ToA transform 三 seed，再扩展到 seed 42/43/44。
- A4b-3b 优先选择 `relative_minmax/no mask`，不是因为它自身 Test Acc 最高，而是因为它在 A4b-2.5 互补性诊断中最值得进一步复查：与 ToT baseline 的 oracle Test Acc 达到 81.51%，oracle gain 为 +11.03%，并且 30 deg 类别 oracle accuracy 从 ToT 的 29.66% 提高到 55.17%。这说明它最能代表“整体弱于 ToT、但可能在 ToT 错误样本上提供补充”的候选。
- A4b-3 的 `eval_mode=True` 只用于确定性推理和逐样本 oracle 对齐，不改变正常训练入口。
- 本地 `timepix-local` conda 环境可以运行脚本 smoke test；但如果本地 `Data/Alpha_100` 与服务器训练数据/checkpoint 不完全一致，本地数值只作为脚本验证，正式结果应在 Linux 服务器同一数据与 checkpoint 环境下生成。

数据集命名兼容备注：

- `a2_best_3seed` 是早期历史 run，当时 `config.yaml` 中记录的 dataset 名称为 `Alpha`，路径为 `/root/autodl-tmp/Alpha`，但本质数据对应当前主线的 `Alpha_100`。
- 服务器当前正式数据目录为 `/root/autodl-tmp/Alpha_100`，因此 A4b-3a/b 重放评估必须显式传入 `--data-root /root/autodl-tmp/Alpha_100`。
- 旧 A2 run 没有显式 `split.path`，会按旧 dataset name 查找 `outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json`。为避免程序自动重新生成 split，应先把当前正式 split `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json` 复制为这个旧名称的兼容别名，并确认二者 hash 一致。
- 不修改历史 run 的 `config.yaml` 或 `metadata.json`；兼容别名只服务于历史 checkpoint 的确定性重放。

服务器准备命令：

```bash
cd /root/Timepix

test -d /root/autodl-tmp/Alpha_100
test -f outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json
test -f outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json

cp -n \
  outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json \
  outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json

sha256sum \
  outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json \
  outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json \
  outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

如果上述三个 split hash 不一致，先不要继续运行 A4b-3b，因为 ToT 与 ToT+ToA 的逐样本 oracle 对齐会失去严格意义。

当前结果记录（用户汇报）：

A4b-3a：ToT-vs-ToT 随机 seed 控制。

| Split | ToT-vs-ToT Oracle Acc | Oracle Gain | ToT 错时另一个 seed 更好 |
| --- | ---: | ---: | ---: |
| Val | 71.56% ± 0.38% | +2.33% ± 0.15% | 8.22% ± 0.61% |
| Test | 73.06% ± 0.00% | +2.55% ± 0.06% | 9.33% ± 0.20% |

30 deg 类别：

| Split | ToT-vs-ToT 30 deg Oracle Gain |
| --- | ---: |
| Val | +2.55% ± 0.80% |
| Test | +1.15% ± 0.80% |

A4b-3b：ToT vs `relative_minmax/no mask`。

| Split | ToT Acc | Candidate Acc | Oracle Acc | Oracle Gain | MAE Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Val | 69.53% | 67.13% | 79.72% | +10.19% | +2.29 deg |
| Test | 70.48% | 67.10% | 81.51% | +11.03% | +2.27 deg |

Test set 中 ToT baseline 错了 297 个样本，其中 candidate 在角度误差意义上更好的有 125 个，占 42.09%。

30 deg 类别：

| Split | ToT 30 deg Acc | Candidate 30 deg Acc | Oracle 30 deg Acc | Oracle Gain |
| --- | ---: | ---: | ---: | ---: |
| Val | 24.31% | 46.53% | 51.39% | +27.08% |
| Test | 29.66% | 46.21% | 55.17% | +25.52% |

阶段性结论：

- 普通 ToT 随机 seed 多样性确实带来少量 oracle 上限，但幅度较小：test oracle gain 约 +2.55%，30 deg oracle gain 约 +1.15%。
- `relative_minmax/no mask` 与 ToT 的互补性明显更强：test oracle gain 为 +11.03%，30 deg oracle gain 为 +25.52%。
- 因此 A4b-2.5 中观察到的强互补性不能简单解释为“换一个随机种子也会这样”。更合理的解释是：相对 ToA early-fusion candidate 捕捉到一部分 ToT baseline 未能正确利用的样本/类别局部信息。
- A4b 的后续重点应从“确认是否存在互补性”转向“互补性能否被验证集可学习的 selector/gate 稳定利用”。test oracle 仍只能作为上限诊断，不能作为模型选择依据。

### A4b-4 Selector Fusion

新增脚本：

```text
scripts/evaluate_selector_fusion.py
```

实验目的：

- 在 A4b-3 证明 `ToT` 与 `relative_minmax/no mask` candidate 存在强互补性之后，验证这种 oracle 互补性能否由规则或轻量 selector 学出来。
- 该实验不训练新的图像主干。ToT baseline 与 candidate checkpoint 全部冻结。

固定输入：

- Primary expert: `a2_best_3seed` 中的 ToT seed42。
- Candidate expert: `a4b_toa_transform_seed42` 中的 `relative_minmax/no mask`。
- Dataset: `Alpha_100`。
- Split: 复用历史 `Alpha_100_ToT` / paired split；旧 `Alpha` split 名称按 A4b-3 的兼容策略处理。

关键实现：

- 重新加载两个已有 `best_model.pth`，用 `eval_mode=True` 在 train/val/test 上做确定性推理。
- A4b-4a：`--selector-mode rule`，不训练 selector，只在 validation 上选择简单置信度、margin、entropy、30 deg 相关规则。
- A4b-4b：`--selector-mode trained --selector-fit train`，在 train split 的 logits/probabilities/confidence/margin/entropy/disagreement 特征上训练 logistic selector。这一版保留为探索性对照，因为 expert 在 train split 上可能过度自信。
- A4b-4c：`--selector-mode trained --selector-fit val-cv`，在 validation 内做 K-fold cross-fitting 产生 out-of-fold selector 分数，用这些分数选择 threshold；随后在完整 validation 上训练最终 selector，并只在 test 上评估。这是更严格的正式 selector 版本。
- 默认 target 为 `lower-error`，即当 candidate 的角度误差严格小于 ToT 时标记为 1，否则保守选择 ToT。
- 所有版本都把 `primary_only` 作为 validation 可选策略，因此如果规则/selector 没有带来验证集收益，会自动退回 ToT baseline。
- test split 只做最终报告，不参与规则、selector、阈值或策略选择。
- 输出 primary-only、candidate-only、规则/selector 候选和 oracle 的 summary；per-class CSV 默认保留 primary、candidate、oracle 和 validation-selected strategy，重点观察 30 deg。

废弃记录：

- 先前泛称为 A4b-4 的初版结果作废，不纳入正式结论。后续重新按 A4b-4a/4b/4c 三个编号运行和汇总。

服务器命令：

A4b-4a rule-based selector：

```bash
cd /root/Timepix

python scripts/evaluate_selector_fusion.py \
  --selector-mode rule \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_4a_rule_selector_seed42.json \
  --output-summary outputs/a4b_4a_rule_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4a_rule_selector_seed42_by_class.csv
```

A4b-4b train-logit selector：

```bash
cd /root/Timepix

python scripts/evaluate_selector_fusion.py \
  --selector-mode trained \
  --selector-fit train \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --selector-target lower-error \
  --selector-epochs 500 \
  --selector-lr 0.01 \
  --selector-weight-decay 0.0001 \
  --output-json outputs/a4b_4b_train_logit_selector_seed42.json \
  --output-summary outputs/a4b_4b_train_logit_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4b_train_logit_selector_seed42_by_class.csv
```

A4b-4c validation-CV selector：

```bash
cd /root/Timepix

python scripts/evaluate_selector_fusion.py \
  --selector-mode trained \
  --selector-fit val-cv \
  --cv-folds 5 \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --selector-target lower-error \
  --selector-epochs 500 \
  --selector-lr 0.01 \
  --selector-weight-decay 0.0001 \
  --output-json outputs/a4b_4c_val_cv_selector_seed42.json \
  --output-summary outputs/a4b_4c_val_cv_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4c_val_cv_selector_seed42_by_class.csv
```

决策备注：

- A4b-4 是从 oracle 诊断进入“可学习选择”的第一步，不改变 A4/A4b-1 已训练模型。
- A4b-4a 先检验简单规则能否利用互补性，解释性最好。
- A4b-4b 保留 train-logit selector，作为与常规训练后处理的探索性对照；但由于 train logits 可能包含 expert 过拟合/过度自信，不作为最严格主结论。
- A4b-4c 是更严格的 selector 版本：validation 内部 cross-fitting 选择 threshold，test 只评估一次。
- 默认使用 logistic selector；如 logistic 选择器无法利用互补性，再考虑 `--selector-hidden-dim` 的小 MLP，但应把这作为后续变体并记录。
- 如果 A4b-4a/4b/4c 都不能超过 ToT baseline，而 oracle 上限仍很高，则说明仅基于 logits/confidence 的可观测特征不足以判断何时切换，需要再考虑图像/物理特征级 selector 或 gated model。

当前结果记录（用户汇报）：

| Experiment | Val-selected strategy | Test Acc | vs ToT | Test MAE | Test Macro-F1 | Test selection rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ToT baseline | `primary_only` | 70.48% | 0 | 5.964 deg | 0.646 | 0% |
| A4b-4a rule | `entropy_adv_0p03` | 70.97% | +0.50% | 5.905 deg | 0.658 | 14.51% |
| A4b-4b train selector | `threshold=0.95` | 71.17% | +0.70% | 5.890 deg | 0.654 | 6.96% |
| A4b-4c val-CV selector | `threshold=0.95` | 70.38% | -0.10% | 6.009 deg | 0.644 | 1.39% |
| Oracle | ideal switch | 81.51% | +11.03% | 3.698 deg | 0.784 | 12.43% |

阶段性结论：

- A4b-4a 和 A4b-4b 说明仅基于 frozen logits/confidence 的后处理选择器可以获得小幅 test 改善：+0.50% 到 +0.70% accuracy，同时 MAE 和 macro-F1 也略有改善。
- A4b-4b 的 train-logit selector 表现最好，但它使用 train split 的 expert logits 训练 selector，可能受到 expert 在训练集上过度自信/过拟合模式影响，因此更适合作为探索性上限而非最严格主结论。
- 更严格的 A4b-4c validation-CV selector 未超过 ToT baseline，说明当前可观测 logits 特征并不能稳定学会 oracle 切换规则。
- A4b-4 总体结论应谨慎表述：选择性利用 `relative_minmax/no mask` 的确能带来小幅真实收益，但距离 oracle 上限仍很远；互补性存在，但可靠可学习性仍是后续 A4b-5/A4b-6 或物理特征 selector 需要解决的问题。
- A4b-4a 的 `entropy_adv_0p03` 有解释性价值：当 candidate 相比 ToT 呈现更有利的不确定性/熵关系时切换，能以 14.51% 的选择率获得小幅增益，可作为论文中简单 selector baseline。

## 数据分析链路：数据集与近垂直分辨极限

新增目的：

- 为本科论文生成两层数据分析结果：第一层解释 `Alpha_100` 与全量 `Proton_C` 数据集的来源、样本分布、事件级特征和代表性样本；第二层分析 C/质子近垂直角度 `80, 82, 84, 86, 88, 90` 在当前 `ToT` 单模态表示下的可分性限制。
- 该链路只做离线分析，不修改训练主链路，不改变现有 A1-A4/A4b 训练配置。

新增代码：

```text
timepix/analysis/
scripts/analyze_datasets.py
scripts/analyze_resolution_limit.py
scripts/make_analysis_report.py
agent/DATA_ANALYSIS_GUIDE.md
```

输出约定：

```text
outputs/data_analysis/
outputs/resolution_limit/
outputs/analysis_report.md
```

核心决策：

- 全量 `Proton_C` 当前只按 `ToT` 单模态分析，不实现或假设 C/质子 `ToA`。
- `Proton_C_7` 只用于训练实验主线；不要把数据分析默认数据集从 `Proton_C` 改成 `Proton_C_7`。
- 代表性样本由固定 seed 和自动距离规则选择，避免人工挑选样本。
- 统计检验必须同时报告效应量，包括 KS statistic、Wasserstein distance、Cliff's delta、median difference 和 IQR overlap ratio。
- 近垂直结论使用限定性表述：在当前探测器设置、事件提取方法、ToT 单模态矩阵表示和已测试模型/特征族条件下，近垂直 `80-90 deg`、`2 deg` 间隔数据没有表现出足够可分性；不写成“深度学习绝对无法区分”。
- Windows 本地环境中 UMAP/numba 和 sklearn 默认 `lbfgs` 后端可能不稳定，因此 UMAP 在 Windows 默认跳过，LogisticRegression 基线使用更稳的 `liblinear`；Linux 服务器上仍会尝试 UMAP。
- 服务器运行近垂直分析时如出现 `Skipped UMAP: No module named 'umap'`，说明缺少 `umap-learn`；已新增 `requirements-analysis.txt`，可用 `pip install -r requirements-analysis.txt` 安装完整分析依赖，或仅执行 `pip install umap-learn` 补齐 UMAP。
- 绘图风格已调整为论文友好的 Matplotlib 默认参数：300 dpi PNG、PDF 字体嵌入、统一字号、色盲友好配色，并在保存后关闭 figure 以降低批量绘图内存占用。
- 数据分析长循环已加入 `tqdm` 进度条：扫描样本/读取 shape、事件特征提取、传统 ML 基线、pairwise AUC。`requirements-analysis.txt` 已补充 `tqdm>=4.66`；缺少 tqdm 时脚本会回退为普通迭代。

服务器命令：

```bash
python scripts/analyze_datasets.py \
  --data-root Data \
  --output-root outputs/data_analysis \
  --datasets Alpha_100 Proton_C \
  --sample-cap-plot 5000 \
  --seed 42

python scripts/analyze_resolution_limit.py \
  --data-root Data \
  --dataset Proton_C \
  --angles 80 82 84 86 88 90 \
  --modality ToT \
  --output-root outputs/resolution_limit \
  --sample-cap-plot 5000 \
  --sample-cap-ml 10000 \
  --seeds 42 43 44 45 46

python scripts/make_analysis_report.py \
  --data-analysis-root outputs/data_analysis \
  --resolution-root outputs/resolution_limit \
  --out outputs/analysis_report.md
```

本地验证：

- 使用 `D:\Program\Anaconda\envs\timepix\python.exe` 完成 `compileall`、三个脚本 `--help`、空数据 smoke test、合成小数据的 dataset/resolution/report 全流程测试。
- 本地没有完整 `Data/Alpha_100` 和 `Data/Proton_C`，因此未在真实数据上生成最终论文表图；真实运行应放到服务器或数据完整的环境。

## 过渡或旧配置

- `configs/experiments/compare_models.yaml`: 早期主干对比配置。正式 A3 优先使用 `a3_backbone_comparison.yaml`。
- `configs/search/alpha_resnet18_tot_training.yaml`: 旧超参搜索配置。正式 A2 优先使用 `a2_alpha_resnet18_tot_training.yaml`。
- `configs/experiments/alpha_resnet18_tot.yaml`: baseline 模板。A2 后的新实验通常应继承 `alpha_tot_a2_best_base.yaml`。

## B1 Proton/C 训练超参搜索

### B1-1 learning rate × batch size

配置文件：

```text
configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml
```

实验目的：

- 在 `Proton_C_7` 数据集上固定 alpha A1 得到的 ResNet18 stem/variant，先对训练过程中的 `learning_rate` 和 `batch_size` 做小范围搜索。
- 为后续质子/C 消融实验沉淀默认训练配置。

固定设置：

- Dataset: `Proton_C_7`
- Modality: `ToT`
- Task: classification
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Fusion: `none`
- Scheduler: `cosine`
- `eta_min=1e-7`
- `dropout=0.1`
- `weight_decay=1e-4`
- `epochs=25`
- `early_stopping_patience=5`
- Primary metric: `val_accuracy`
- Split: `outputs/splits/Proton_C_7_ToT_seed42_0.8_0.1_0.1.json`
- AMP: enabled

网格：

```yaml
grid:
  training.learning_rate:
    - 0.0001
    - 0.0003
    - 0.001
  training.batch_size:
    - 64
    - 128
    - 256
```

共 9 组实验。

决策备注：

- B1-1 固定的是 A1 结构结论中的 ResNet18 variant/stem：`resnet18_no_maxpool + conv1 2/1/0`。
- A1 当时观察到 `dropout=0.3` 表现最好，但 A2 后续将 dropout 视为训练超参并搜索到 `dropout=0.1`。B1-1 暂固定 `dropout=0.1`，作为保守训练默认值，不把它表述为 A1 结构参数。
- 原 B1-1 使用 `epochs=20`，运行中观察到部分组合在停止时准确率仍在上升，因此将训练预算调整为 `epochs=25`，`early_stopping_patience` 保持 5。
- 为避免 20 epoch 历史结果与 25 epoch 正式结果混在同一汇总中，当前配置的 `experiment_name` 与 `experiment_group` 改为 `b1_proton_c7_resnet18_tot_lr_batch_ep25`；旧 `b1_proton_c7_resnet18_tot_lr_batch` 结果仅作为被替换的诊断记录。
- `batch_size=256` 保留在第一轮搜索中；服务器运行时建议使用 `--continue-on-error`，如果显存不足不会中断整组。
- B1-2 将在 B1-1 选出最佳 `learning_rate + batch_size` 后，只搜索 `weight_decay = [0, 1e-5, 1e-4]`。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_lr_batch_ep25 --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_runs.csv
```

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

## A4b planning update after selector results

Date: 2026-04-29.

Context:
- A4b-4a rule selector, A4b-4b train-logit selector, and A4b-4c validation-CV selector have been reported.
- A4b-4a and A4b-4b show small real test gains over ToT, but A4b-4c does not beat ToT.
- Oracle remains far higher than all practical selectors, so the bottleneck is no longer proving complementarity; it is explaining and improving switch reliability.

Current interpretation:
- A4b-4a is the cleanest positive baseline because the rule is selected on validation and test is final-only.
- A4b-4b is exploratory because it trains on expert outputs from the train split, where frozen experts may be overconfident.
- A4b-4c is the stricter learned-selector result and is currently negative/neutral.
- Therefore, A4b should not jump directly to complex end-to-end fusion. The next step is switch diagnostics and then low-cost soft-gate/residual variants.

Updated A4b numbering:
- A4b-4d: switch diagnostics for the A4b-4a selected rule `entropy_adv_0p03`. No training. Report switch precision, switch recall, harmful/neutral switch rates, per-class switch behavior, and score distributions.
- A4b-4e: optional three-seed confirmation for the A4b-4a result. Only rerun the key `relative_minmax/no mask` candidate for seeds 43 and 44; reuse existing `a2_best_3seed` ToT baselines.
- A4b-5: entropy soft gate based on the A4b-4a entropy advantage signal, with validation-selected threshold/slope.
- A4b-6: constrained residual interpolation using the same entropy gate and a validation-selected beta grid.
- A4b-7: compact ToA-only relative controls.
- A4b-8: ToT image plus ToA scalar physical features.
- A4b-9: optional end-to-end gated expert fusion after low-cost selectors are understood.

Deferred:
- GMU, FiLM, MMTM, ordinary feature concat, and larger mask/transform grids are deferred until A4b-4d/A4b-5/A4b-6 clarify whether selector/gate signals are reliable.

### A4b-4d switch diagnostics implementation

Implemented script:

```text
scripts/analyze_selector_switches.py
```

Purpose:
- Reproduce the fixed A4b-4a validation-selected rule `entropy_adv_0p03`.
- Explain whether the small A4b-4a gain is limited by low switch precision, low switch recall, harmful switches, missed 30 deg beneficial samples, or overlapping selector-score distributions.
- This is a no-training diagnostic; it does not select a new rule or threshold.

Rule definition used by the existing selector implementation:

```text
entropy_adv_0p03 switches to candidate when:
  primary prediction and candidate prediction disagree
  candidate_entropy <= primary_entropy - 0.03
```

Server command:

```bash
cd /root/Timepix

python scripts/analyze_selector_switches.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --rule entropy_adv_0p03 \
  --output-json outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42.json \
  --output-summary outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_summary.csv \
  --output-by-class outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_by_class.csv \
  --output-samples outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_samples.csv \
  --output-distribution outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_distribution.csv
```

Outputs:
- summary CSV: overall switch precision/recall, harmful/neutral switch rate, final metrics, oracle metrics.
- by-class CSV: same diagnostics per true angle class, especially 30 deg.
- samples CSV: per-sample predictions, errors, selected flag, oracle-beneficial flag, and switch outcome.
- distribution CSV: score distributions for selected-beneficial, selected-harmful, selected-neutral, missed-beneficial, and no-benefit groups.

Local verification:
- `python scripts\analyze_selector_switches.py --help`
- `python -m py_compile scripts\analyze_selector_switches.py`
- Small synthetic helper smoke test for switch precision/recall calculations.

## B1-1 epoch-20 recovery plan

Date: 2026-04-29.

Issue:
- B1-1 was accidentally run with the older 20-epoch setup.
- The intended B1-1 budget is now 25 epochs with `early_stopping_patience=5`.
- Re-running the full 9-run grid from scratch is expensive.

Decision:
- If each old run has `last_checkpoint.pth`, it can be continued from that checkpoint with `training.epochs=25`.
- This continuation is valid as an epoch-budget rescue, but it is not perfectly identical to a fresh 25-epoch run from scratch, because the cosine scheduler had already followed the 20-epoch schedule before the resume. Record this as `from20` continuation rather than pretending it was originally trained with `T_max=25`.
- Runs that already triggered early stopping before epoch 20 should usually be skipped, because increasing `max_epochs` from 20 to 25 would not have changed a run that had already stopped by patience.
- Keep the old 20-epoch group intact and copy resumed runs into a new group to avoid mixing old and rescued results.

Implemented support:

```text
scripts/extend_runs.py
```

Recommended server dry-run:

```bash
cd /root/Timepix

python scripts/extend_runs.py \
  --source-group b1_proton_c7_resnet18_tot_lr_batch \
  --target-group b1_proton_c7_resnet18_tot_lr_batch_ep25_from20 \
  --target-epochs 25 \
  --data-root /root/autodl-tmp/Proton_C_7 \
  --skip-completed \
  --skip-early-stopped \
  --resume-target-existing \
  --dry-run
```

Recommended server execution:

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

Summary command:

```bash
python scripts/summarize.py \
  --group b1_proton_c7_resnet18_tot_lr_batch_ep25_from20 \
  --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_from20_runs.csv
```

Local verification:
- `python scripts\extend_runs.py --help`
- `python -m py_compile scripts\extend_runs.py`
- `D:\Program\Anaconda\envs\timepix-local\python.exe scripts\extend_runs.py ... --dry-run` on local B1 outputs.

## A4b-4e three-seed selector confirmation

Date: 2026-04-29.

Purpose:
- A4b-4a produced a small positive seed42 result, but the gain is only about +0.50% test accuracy.
- To use it as more than a diagnostic, we need a three-seed confirmation.
- The expensive part is the candidate expert; ToT seed42/43/44 already exist in `a2_best_3seed`.

Decision:
- Do not retrain the seed42 candidate. Reuse the existing `a4b_toa_transform_seed42` run for `relative_minmax/no mask`.
- Train only the key candidate `ToT + relative_minmax ToA, no mask` for seeds 43 and 44.
- Then evaluate oracle complementarity and A4b-4a rule selector for seeds 42/43/44.

New files:

```text
configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml
scripts/aggregate_selector_fusion.py
```

Candidate training:

```bash
cd /root/Timepix

python scripts/run_grid.py \
  --config configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml \
  --dry-run

python scripts/run_grid.py \
  --config configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml \
  --continue-on-error

python scripts/summarize.py \
  --group a4b_4e_relative_minmax_no_mask_seed43_44 \
  --out outputs/a4b_4e_relative_minmax_no_mask_seed43_44_runs.csv
```

Three-seed oracle confirmation:

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-vs-candidate \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
  --seeds 42 43 44 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_4e_oracle_3seed.json \
  --output-summary outputs/a4b_4e_oracle_3seed_summary.csv \
  --output-by-class outputs/a4b_4e_oracle_3seed_by_class.csv
```

Three-seed rule-selector confirmation:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_selector_fusion.py \
    --selector-mode rule \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_4e_rule_selector_seed${seed}.json" \
    --output-summary "outputs/a4b_4e_rule_selector_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_4e_rule_selector_seed${seed}_by_class.csv"
done
```

Aggregate selector result:

```bash
python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_4e_rule_selector_seed42_summary.csv \
    outputs/a4b_4e_rule_selector_seed43_summary.csv \
    outputs/a4b_4e_rule_selector_seed44_summary.csv \
  --out outputs/a4b_4e_rule_selector_mean_std.csv
```

Interpretation:
- If the validation-selected rule selector improves mean test accuracy, MAE, and macro-F1 over `primary_only`, A4b-4a can be reported as a small but stable selector baseline.
- If the mean improvement disappears or variance is high, A4b-4a remains a seed42 diagnostic; the stronger conclusion is still the oracle-level complementarity, not a reliable deployed fusion gain.

## A4b-5 sample-wise gated late fusion

Date: 2026-04-29.

Decision:
- A4b-5 and A4b-6 are no longer treated as sequential "whether to continue" checks.
- They are formal selective-fusion comparison families. Soft/constrained variants are diagnostic ablations, and learned variants are compared in the same run.
- A4b-5 is implemented first because it is a direct extension of A4b-4: replace hard selection/global alpha with a sample-wise gate.

Implementation:

```text
scripts/evaluate_gated_late_fusion.py
```

Fixed constraints:
- Dataset/split: `Alpha_100`, paired split `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`.
- Primary expert: ToT baseline from `a2_best_3seed`.
- Candidate expert: `ToT + relative_minmax ToA, no mask`.
- Primary/candidate ResNet experts are frozen; only the gate is trained/calibrated.
- Test set is not used for choosing gate type, threshold, slope, fit mode, or regularization.

Implemented A4b-5 variants:
- A4b-5a: entropy soft gate, probability fusion.
- A4b-5b: learned scalar gate, probability fusion.
- A4b-5c: learned scalar gate, logit fusion.
- A4b-5d: class-aware probability gate.
- A4b-5e: conservative scalar probability gate, initialized toward ToT and penalized for high mean gate.

Gate features:
- ToT logits, candidate logits, logit differences.
- ToT probabilities, candidate probabilities, probability differences.
- Top1 confidence, top1-top2 margin, entropy for each expert.
- Disagreement flag and predicted angle difference.
- ToT-predicts-30 and candidate-predicts-30 flags.

Gate fitting:
- `train`: exploratory/optimistic reference, because expert outputs on train can be overconfident.
- `val-cv`: stricter variant using validation cross-fitting for validation metrics and final validation fit for test reporting.
- A4b-5a uses validation-grid selection over entropy threshold and sigmoid slope.

Seed42 command:

```bash
cd /root/Timepix

python scripts/evaluate_gated_late_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_5_gated_late_fusion_seed42.json \
  --output-summary outputs/a4b_5_gated_late_fusion_seed42_summary.csv \
  --output-by-class outputs/a4b_5_gated_late_fusion_seed42_by_class.csv
```

Three-seed command after A4b-4e candidate seeds are available:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_gated_late_fusion.py \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_5_gated_late_fusion_seed${seed}.json" \
    --output-summary "outputs/a4b_5_gated_late_fusion_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_5_gated_late_fusion_seed${seed}_by_class.csv"
done

python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_5_gated_late_fusion_seed42_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed43_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed44_summary.csv \
  --out outputs/a4b_5_gated_late_fusion_mean_std.csv
```

Output:
- Summary CSV includes ToT baseline, candidate-only, A4b-4a rule, all A4b-5 variants, and oracle.
- Per-class CSV includes baselines, rule, selected A4b-5 variant, and oracle.
- Summary rows include validation/test Acc, MAE, P90, macro-F1, mean gate, high-gate rate, true-30 mean gate, beneficial high-gate count, and harmful high-gate count.

Local verification:
- `python scripts\evaluate_gated_late_fusion.py --help`
- `python -m py_compile scripts\evaluate_gated_late_fusion.py`
- Synthetic logits smoke test using `D:\Program\Anaconda\envs\timepix-local\python.exe`.
