# 5.5 Pro 研究交接说明

本文档用于让研究型模型快速理解本课题、当前实验状态，以及下一步可协助的文献调研、论文大纲、创新点和结果分析任务。

## 一句话课题

本课题研究如何利用 Timepix/Timepix3 探测器记录的带电粒子像素响应矩阵，识别粒子的入射极角。当前主线是 alpha 粒子数据集上的离散角度分类任务，输入为 ToT 和/或 ToA 稀疏矩阵，输出为角度类别。

## 建议先阅读的文档包

必须阅读：

1. `agent/RESEARCH_HANDOFF_5_5_PRO.md`
2. `agent/PHYSICS_CONTEXT.md`
3. `agent/EXPERIMENT_LOG.md`
4. `configs/README.md`
5. `agent/FILE_MAP.md`

需要工程细节时再读：

1. `agent/CODE_CONTEXT.md`
2. `agent/ARCHITECTURE.md`
3. `agent/NEW_SYSTEM_GUIDE.md`
4. `agent/EXPERIMENT_GROUPS.md`
5. `agent/SERVER_TRAINING.md`

实验结果完成后应额外提供：

```text
outputs/a3_backbone_comparison_runs.csv
outputs/a3_backbone_comparison_mean_std.csv
outputs/a4_modality_comparison_runs.csv
outputs/a4_modality_comparison_mean_std.csv
```

如果当前只跑了 seed 42 快速版，应提供：

```text
outputs/a3_backbone_comparison_seed42_summary.csv
outputs/a4_modality_comparison_seed42_summary.csv
```

也可以直接提供对应实验目录：

```text
outputs/experiments/a3_backbone_comparison/
outputs/experiments/a4_modality_comparison/
outputs/experiments/a3_backbone_comparison_seed42/
outputs/experiments/a4_modality_comparison_seed42/
```

## 当前状态

当前日期：2026-04-29。

- A1 结构适配实验已完成，用于确定 ResNet18 在 Timepix 稀疏矩阵上的结构适配方式。
- A2 训练超参数搜索已完成，已得到后续实验统一使用的 A2 best base。
- AMP 对比已完成，结论是混合精度有效且没有明显降低准确率，后续正式训练可以开启 AMP。
- A3 主干模型对比已有当前结果记录，支持 `resnet18_no_maxpool` 作为当前最佳主干模型。
- A4 模态对比已有当前结果记录，当前实现下 ToT 单模态最好，ToT+ToA 没有提升。
- A4b ToA 融合策略已有结果：相对 ToA 表达优于 raw/log1p early fusion，但仍未超过 ToT；late logit fusion 在 validation 上选择 `alpha_toa=0`。后续互补性诊断显示 ToA/relative ToT+ToA 与 ToT 错误并非完全重叠，存在 oracle 上限提升，尤其 `relative_minmax, no mask` 对 30 deg 有明显 oracle 改善。
- B1 Proton/C 第一轮训练超参搜索配置已准备；正式质子/C 数据集名称已切换为 `Proton_C_7`，固定 A1 ResNet18 stem/variant 后搜索 `learning_rate × batch_size`。
- 论文数据分析链路与训练链路分开：数据分析默认使用全量 `Proton_C`，训练实验默认使用 7 分类子集 `Proton_C_7`。
- 时间紧张时已准备 A3/A4 的 seed 42 快速版配置，但正式论文结论优先使用三 seed mean/std。

## 数据集主线

当前正式实验主线使用：

```text
Alpha_100
```

曾短暂切换到 `Alpha_50`，但效果不佳，不能支撑完整实验故事线，因此后续 A3/A4/A5/A6 默认回到 `Alpha_100`。

当前数据集配置：

```text
configs/datasets/alpha_100.yaml
configs/datasets/alpha_50.yaml
configs/datasets/proton_c_7.yaml
```

其中：

- `Alpha_100`：正式 alpha 主线，100x100，支持 ToT 和 ToA。
- `Alpha_50`：保留为对照/历史配置，不作为当前正式主线。
- `Proton_C_7`：C/质子 7 分类数据集，目前只有 ToT；后续 Proton/C 训练只使用该数据集。
- `Proton_C`：C/质子全量数据集，目前只用于论文数据分析和近垂直分辨极限分析，不作为训练配置入口。

## Split 决策

A1/A2 当时使用的历史 ToT split 已恢复，并规范命名为：

```text
outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json
```

后续基于 `Alpha_100 + ToT` 的实验应复用这份 split，以保持与历史 A1/A2/A3 的数据划分一致。

A4 双模态 split 的决策：

```text
outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

该文件不应重新随机生成，而应从历史 ToT split 复制得到：

```bash
cp outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json \
   outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

原因是 `Alpha_100` 中 ToT 与 ToA 文件完全一一对应，split manifest 保存的是去掉 ToT/ToA 标记后的归一化 sample key。因此同一份划分内容可用于 ToT、ToA 和 ToT+ToA。

## 主要实验设计

### A1 结构适配

目的：确定 ResNet18 如何适配 Timepix 稀疏矩阵。

固定：`Alpha_100`、ToT、CE、one-hot、无手工特征、固定 seed。

比较：

- 原始 ResNet18 baseline。
- `resnet18_no_maxpool` vs `resnet18_maxpool`。
- `conv1_kernel_size`: 2 / 3 / 5。
- `conv1_stride`: 1 / 2。
- `dropout`: 0 / 0.1 / 0.3。

已观察到的结论：`resnet18_no_maxpool + kernel_size=2 + stride=1 + dropout=0.3` 在 A1 中表现最好。A2 后续将 dropout 纳入超参搜索，因此最终 base 使用 A2 搜索得到的 `dropout=0.1`。

### A2 训练超参数搜索

目的：固定 A1 结构后，搜索训练过程相关超参数，作为后续消融和模型对比的统一训练配置。

A2 best:

```text
learning_rate = 4.3878e-05
weight_decay  = 4.7324e-04
batch_size    = 32
eta_min       = 1.6433e-07
dropout       = 0.1
scheduler     = cosine
epochs        = 25
early_stopping_patience = 8
mixed_precision = true
```

最佳 trial 记录：

```text
trial          = 12
val accuracy   = 0.6953
test accuracy  = 0.7048
val MAE        = 6.279 deg
test MAE       = 5.964 deg
test macro-F1  = 0.6461
best epoch     = 24
```

### A3 主干模型对比

目的：在 `Alpha_100 + ToT` 单模态任务上比较不同主干模型，选择后续实验主干。

配置：

```text
configs/experiments/a3_backbone_comparison.yaml
configs/experiments/a3_backbone_comparison_seed42.yaml
```

固定：

- `Alpha_100`
- ToT
- CE
- one-hot
- 无手工特征
- A2 best 训练超参
- 历史 `Alpha_100_ToT` split

比较模型：

```text
shallow_cnn
shallow_resnet
resnet18_no_maxpool
densenet121
efficientnet_b0
convnext_tiny
vit_tiny
```

ViT 设置：

```text
image_size = 100
patch_size = 10
```

理由：保持 `10 x 10 = 100` 个 patch token，不 resize 到 224x224。

当前结果记录（2026-04-29 用户汇报）：A3 支持 `resnet18_no_maxpool` 作为当前最佳主干模型。

| Rank | Model | Test Acc | Val Acc | Test MAE | Test Macro-F1 | Params | Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `resnet18_no_maxpool` | 70.48% | 69.53% | 5.96 deg | 0.646 | 11.43M | 10.0 min |
| 2 | `convnext_tiny` | 68.99% | 67.03% | 6.29 deg | 0.612 | 28.15M | 23.2 min |
| 3 | `shallow_resnet` | 68.99% | 67.03% | 6.26 deg | 0.634 | 1.34M | 4.1 min |
| 4 | `densenet121` | 68.69% | 67.73% | 6.40 deg | 0.610 | 7.34M | 24.8 min |
| 5 | `shallow_cnn` | 65.01% | 62.74% | 6.86 deg | 0.485 | 0.52M | 1.8 min |
| 6 | `efficientnet_b0` | 64.51% | 63.84% | 6.95 deg | 0.616 | 4.47M | 24.5 min |
| 7 | `vit_tiny` | 35.19% | 35.16% | 14.96 deg | 0.130 | 5.56M | 10.8 min |

每类 F1 记录：

| Model | 15 deg F1 | 30 deg F1 | 45 deg F1 | 60 deg F1 |
| --- | ---: | ---: | ---: | ---: |
| `resnet18_no_maxpool` | 0.763 | 0.402 | 0.751 | 0.669 |
| `shallow_resnet` | 0.732 | 0.410 | 0.762 | 0.632 |
| `convnext_tiny` | 0.756 | 0.306 | 0.747 | 0.638 |
| `densenet121` | 0.749 | 0.315 | 0.751 | 0.623 |
| `efficientnet_b0` | 0.679 | 0.418 | 0.709 | 0.658 |

### A4 模态对比

目的：验证 ToT、ToA、ToT+ToA 对极角识别的贡献。

配置：

```text
configs/experiments/a4_modality_comparison.yaml
configs/experiments/a4_modality_comparison_seed42.yaml
```

固定：

- `Alpha_100`
- `resnet18_no_maxpool`
- A2 best 训练超参
- CE
- one-hot
- 无手工特征
- `fusion_mode: none`
- paired split 从历史 ToT split 复制得到

比较：

```text
[ToT, ToA]
[ToT]
[ToA]
```

当前结果记录（2026-04-29 用户汇报）：当前实现下，ToT 单模态最好，ToT+ToA 没有提升，ToA 单独效果低于 ToT。

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

### A4b ToA 融合策略初步验证

目的：在 A4 显示 raw/log1p ToT+ToA 不如 ToT 单模态之后，检查 ToA 是否可以通过更合适的表达或更保守的融合方式提供稳定补充。

阶段 1：ToA 相对时间表达 + early fusion。

当前结果记录（用户汇报）：

| Experiment | Val Acc | Test Acc | Test MAE | Test P90 | Test Macro-F1 | 30 deg F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A4 ToT baseline | **69.53%** | **70.48%** | **5.96 deg** | **15 deg** | **0.646** | 0.402 |
| A4 ToT+ToA raw/log1p | 64.04% | 65.90% | 6.92 deg | 30 deg | 0.553 | 0.178 |
| A4b relative_centered, no mask | 67.83% | **68.79%** | **6.49 deg** | 30 deg | 0.612 | 0.290 |
| A4b relative_minmax, no mask | 67.13% | 67.10% | 6.86 deg | 30 deg | 0.635 | **0.447** |

阶段 1 结论：相对 ToA 表达确实比 raw/log1p ToT+ToA 好，但仍未超过 ToT 单模态。30 deg F1 的局部改善说明 ToA 可能包含类别局部信息，但不足以改善总体指标。

阶段 2：late logit fusion。

当前结果记录（用户汇报）：

| alpha_toa | Selected by val | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.00 | **yes** | **69.53%** | 70.48% | 5.96 deg | **0.646** |
| 0.05 | no | 69.23% | 70.68% | **5.93 deg** | 0.646 |
| 0.30 | no | 68.93% | **70.97%** | 5.95 deg | 0.636 |

阶段 2 结论：验证集选择 `alpha_toa=0`，即完全不用 ToA。test 上个别非零 alpha 的轻微提升不能用于选择模型或宣称稳定增益。

阶段 2.5：预测互补性诊断。

脚本：

```text
scripts/analyze_prediction_complementarity.py
```

当前 seed-42 结果记录：

| Comparator | ToT Wrong + Other Correct | Other Better When ToT Wrong | Oracle Acc | Oracle Gain | Oracle MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| ToA | 74 | 77 / 297 | 77.83% | +7.36% | 4.443 deg |
| relative_minmax, no mask | 111 | 125 / 297 | **81.51%** | **+11.03%** | **3.698 deg** |
| relative_centered, no mask | 82 | 84 / 297 | 78.63% | +8.15% | 4.324 deg |
| relative_rank, no mask | 104 | 113 / 297 | 80.82% | +10.34% | 3.862 deg |

30 deg 类别中，ToA 单模态对 ToT 错误没有补救能力；但 `relative_minmax, no mask` 可将 oracle accuracy 从 ToT 的 29.66% 提高到 55.17%。这说明 ToA 相关输入存在可挖掘的类别局部互补信息，关键在于设计选择性融合机制，而不是简单拼接或固定权重融合。

### B1 Proton/C 训练超参搜索

目的：在质子/C 数据集上固定 alpha A1 得到的 ResNet18 结构适配方式，仅搜索训练过程超参，为后续质子/C 消融实验确定默认训练配置。

配置：

```text
configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml
```

固定：

- Dataset: `Proton_C_7`
- Modality: `ToT`
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Scheduler: `cosine`
- `eta_min=1e-7`
- `dropout=0.1`
- `epochs=20`
- `early_stopping_patience=5`

B1-1 搜索：

```text
learning_rate = [1e-4, 3e-4, 1e-3]
batch_size    = [64, 128, 256]
```

备注：A1 的结构结论是 no-maxpool、conv1 2/1/0；`dropout=0.1` 是沿用 A2 风格的保守训练默认值，不写成 A1 结构参数。

## 当前可讲的论文故事线

建议主线：

1. Timepix/Timepix3 探测器输出稀疏像素矩阵，轨迹形态和 ToT/ToA 分布携带入射极角信息。
2. 先在 alpha ToT 单模态上确定适合稀疏探测器矩阵的 ResNet18 结构。
3. 再搜索训练超参数，固定统一训练预算，避免后续消融和模型对比混入调参因素。
4. 在统一训练设置下比较多种 CNN/现代视觉主干和 ViT-Tiny；当前 A3 记录支持 `resnet18_no_maxpool` 作为主干。
5. 比较 ToT、ToA、ToT+ToA；当前 A4 记录显示 ToT 单模态最好，ToT+ToA 没有提升。A4b 进一步说明，相对 ToA 表达和 late logit fusion 仍不足以稳定超过 ToT，但 oracle 互补性分析支持继续探索选择性融合。
6. 后续开始 Proton/C 数据集主线，先用 B1 搜索训练默认配置，再讨论质子/C 消融、近垂直角度可分性和跨粒子比较。

## 5.5 Pro 可以重点协助的任务

文献调研：

- Timepix/Timepix3 探测器用于粒子轨迹、角度或粒子识别的研究。
- ToT/ToA 在探测器数据分析中的物理含义和常见处理方式。
- 深度学习用于像素探测器、稀疏事件图像、粒子轨迹重建或角度估计的论文。
- CNN/ResNet/DenseNet/EfficientNet/ConvNeXt/ViT 在小样本稀疏物理图像上的适用性讨论。
- 有序类别分类、EMD/Wasserstein loss、角度 MAE 等指标在角度估计中的使用。

论文大纲：

- 绪论：研究背景、Timepix 探测器、角度识别意义。
- 相关工作：传统特征方法、深度学习方法、多模态探测器信息融合。
- 数据与预处理：Alpha_100、ToT/ToA、split 恢复、归一化、旋转增强。
- 方法：统一实验框架、ResNet18 结构适配、A2 超参搜索、主干模型、多模态输入。
- 实验：A1/A2/A3/A4 及后续消融。
- 讨论：归纳偏置、ToT/ToA 贡献、误差模式、局限性。
- 结论与展望。

可能创新点表达：

- 面向 Timepix 稀疏探测器矩阵的 ResNet18 结构适配。
- 在统一 split 与统一训练预算下系统比较 CNN/现代视觉主干/ViT。
- 对 ToT、ToA 及双模态输入在极角识别中的贡献进行控制变量分析。
- 将角度分类准确率与物理角度误差指标结合，而不仅看 accuracy。
- 构建可复现的实验配置体系，支持 split 复用、多 seed、AMP、超参搜索和结果汇总。

## 重要注意事项

- 不要把 Timepix 矩阵当作自然图像；它是稀疏探测器物理响应。
- 当前正式主线是 `Alpha_100`，不是 `Alpha_50`。
- A4 的 paired split 是从历史 ToT split 复制来的，不是重新生成的。
- A2 best 来自 `Alpha_100`，后续实验借用这组超参是刻意设计。
- test 指标只用于最终报告，不用于选择超参或模型。
- 单 seed 结果可以作为阶段性证据，但正式论文优先报告三 seed mean/std。
- ViT-Tiny 不 resize 到 224x224；当前使用原生 100x100 输入和 patch size 10。
- C/质子数据集目前只有 ToT，不能直接做 ToT+ToA。

## 给 5.5 Pro 的建议初始提示

```text
请先阅读 agent/RESEARCH_HANDOFF_5_5_PRO.md、agent/PHYSICS_CONTEXT.md、
agent/EXPERIMENT_LOG.md、configs/README.md 和 agent/FILE_MAP.md。

我的课题是使用 Timepix/Timepix3 探测器的 ToT/ToA 像素矩阵识别带电粒子
入射极角。当前正式数据主线是 Alpha_100，A1/A2 已完成，A3 主干模型对比
正在进行，A4 模态对比马上完成。我希望你帮助进行文献调研、论文大纲撰写、
创新点提炼、实验结果解释和后续实验设计。请特别注意 split 复用、Alpha_100
主线、A2 best 超参、A3/A4 的控制变量设计，以及单 seed 与三 seed 结论的区别。
```
