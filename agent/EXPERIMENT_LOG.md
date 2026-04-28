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

## A4 模态对比

配置文件：

```text
configs/experiments/a4_modality_comparison.yaml
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
