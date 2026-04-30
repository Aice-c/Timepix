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

当前日期：2026-04-30。

- A1 结构适配实验已完成，用于确定 ResNet18 在 Timepix 稀疏矩阵上的结构适配方式。
- A2 训练超参数搜索已完成，已得到后续实验统一使用的 A2 best base。
- AMP 对比已完成，结论是混合精度有效且没有明显降低准确率，后续正式训练可以开启 AMP。
- A3 主干模型对比已有当前结果记录，支持 `resnet18_no_maxpool` 作为当前最佳主干模型。
- A4 模态对比已有当前结果记录，当前实现下 ToT 单模态最好，ToT+ToA 没有提升。
- A4b ToA 融合策略已有结果：相对 ToA 表达优于 raw/log1p early fusion，但仍未超过 ToT；late logit fusion 在 validation 上选择 `alpha_toa=0`。后续互补性诊断显示 ToA/relative ToT+ToA 与 ToT 错误并非完全重叠，存在 oracle 上限提升，尤其 `relative_minmax, no mask` 对 30 deg 有明显 oracle 改善。
- A4b 后续选择性融合已形成完整阶段：`A4b-4e` rule selector 三 seed 稳定小幅提升，`A4b-5` frozen-expert sample-wise gated late fusion test 泛化表现强，`A4b-6` residual fusion 是 validation-selected expert-level 后处理融合系统。二者不应混成单一 winner：A4b-6 用于说明 validation accuracy / MAE 口径下 residual expert fusion 的优势，A4b-5 保留为强 test reference。
- `A4c: End-to-end full bimodal fusion models` 第一阶段已完成，包含 `A4c-1 dual_stream_concat_aux`、`A4c-2 dual_stream_gmu_aux`、`A4c-3 toa_conditioned_film`。最终端到端多模态架构选择 `A4c-2 dual_stream_gmu_aux`，选择依据不能使用 test：GMU 的 validation accuracy 不弱，Val Macro-F1 与 FiLM 近似持平，Val MAE 在 A4c 端到端模型中最好，且 gate 机制与 A4b 得到的 “ToA 是选择性辅助模态” 结论一致。test 结果只用于最终泛化报告。第二批 `A4c-4 warm_started_expert_gate` 也已完成；`freeze_experts=true` 优于 ToT 但不超过 A4b-5/A4c-1/2，`freeze_experts=false` 不稳定。`A4c-5 mmtm_lite` 暂为选做。
- A5 物理/手工标量特征融合已完成 A5a/A5b/A5c/A5d 主体链路：不参考 `timepix/analysis/` 的既有特征实现；训练链路已新增 12 维合理保留候选特征、`handcrafted_features.source_modalities` 解耦、`handcrafted_mlp` 和 `scripts/screen_handcrafted_features.py`。A5a 使用 `RandomForest` / one-vs-rest `LogisticRegression` / validation permutation importance 做筛选诊断；其中 `LogisticRegression` 显式使用 `OneVsRestClassifier(LogisticRegression(solver="liblinear"))` 以兼容服务器 `scikit-learn` 多分类行为。A5a 显示 Geometry 最重要，其次是 ToT；ToA 标量有补充但不是主导。A5b 低冗余 seed42 CNN concat pilot 没有证明 handcrafted concat 能稳定提升 Alpha ToT CNN；A5c gated seed42 显示 gated 明显优于 simple concat。
- A5d 正式三 seed 结果已拉取到 `outputs/a5d_alpha_handcrafted_gated_3seed_runs.csv` 和 `outputs/a5d_alpha_handcrafted_gated_3seed_mean_std.csv`。A5d 同时验证五维 `main_5feat` 和 `ToA-only` diagnostic：严格按 validation accuracy，A5d 内部最佳应为 `toa_only_diag`（Val Acc 70.70±0.31%）；`main_5feat` 的 Test MAE / Test Macro-F1 最好（5.875±0.231 / 0.646±0.005），但不能用 test 反选为最佳。两组 A5d 都没有超过 A2 ToT baseline 的 Test Acc，因此 A5 的论文口径应是：低维物理标量有解释性和一定辅助信号，`gated` 融合比 concat 更合适，但在 ResNet18 ToT 图像特征已经较强时，Test Acc 增益有限且不稳定。因此 B2 不再表述为“迁移 Alpha 最优手工特征”，而是低成本验证 Proton_C_7 上 ToT-only 标量是否与 CNN 冗余或能改善弱类别。
- A6 已开始推进，定位为 Alpha 版 B3：固定 `Alpha_100 + ToT + resnet18_no_maxpool + A2 best`，只筛选 angle-ordinal loss / label strategy，不混入 A5 handcrafted 或 A4 multimodal grid。A6a 配置为 `configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml`，包含 Gaussian soft CE `sigma=5/7.5/10`、`ce_expected_mae lambda=0.02/0.05/0.10`、`ce_emd lambda=0.02/0.05/0.10`，共 9 个新增策略 seed42 runs。CE one-hot baseline 不重跑，直接复用同配置 A2-best seed42 和三 seed 结果。pure EMD 和 hybrid regression head 暂不做。A6b/A6c 等 A6a 结果出来后再写具体 best-loss 三 seed 配置。
- 实验编号、阶段目的、完成状态和后续安排的权威索引见 `agent/EXPERIMENT_LOG.md` 中的“实验编号与阶段总览”。
- B1 Proton/C 训练超参搜索与 B1-best 三 seed 认证已完成；正式质子/C 数据集名称为 `Proton_C_7`。B1-1 固定 A1 ResNet18 stem/variant 后搜索 `learning_rate × batch_size`，20 epoch 旧结果和 from20 中继 25 epoch 结果均选择 `learning_rate=3e-4`、`batch_size=128`。B1-2 固定该组合后搜索 `weight_decay=[0,1e-5,1e-4]`，最终仍选择 `weight_decay=1e-4`。正式 B1-best 使用 `early_stopping_patience=8`，配置为 `configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml`，结果为 `Test Acc = 93.26 ± 1.64%`、`MAE = 0.640 ± 0.161`、`Macro-F1 = 0.952 ± 0.011`；旧 patience=5 只作为早停过激诊断。B2 已改为 Proton_C_7 ToT-only 手工特征低成本验证，不再做主干/结构迁移验证；B2a seed42 concat 已完成，`geometry_lowcorr` 对 B1-best seed42 只有极小提升（Test Acc 94.09% -> 94.26%，MAE 0.562 -> 0.545），`tot_lowcorr` 明显变差（Test Acc 91.63%）。B2b seed42 gated 也已完成：`geometry_lowcorr_gated` 为 94.17%，`tot_lowcorr_gated` 为 94.13%。B2b 说明 gated 可以抑制 `ToT_density` 的负面影响，但没有证明手工特征能显著提升 Proton_C_7；B2c 三 seed 当前不优先推进。
- B3a Proton_C_7 有序损失 seed42 screening 已完成，配置为 `configs/experiments/b3a_proton_c7_ordinal_loss_seed42.yaml`。B3a 固定 B1-best patience=8、ToT-only、无 handcrafted，只比较 loss / label strategy。结果显示 `CE+ExpectedMAE lambda=0.05` 是最优候选：Val Acc 94.26%、Test Acc 94.32%、Test MAE 0.540、high-angle F1 0.930，均优于 B1-best seed42。`CE+EMD lambda=0.05` 非常接近且 Test MAE 最低 0.537，但 validation accuracy 低于 `CE+ExpectedMAE lambda=0.05`，因此仅作为 optional 对照。Gaussian soft label 不继续推进，尤其 `sigma=15` 明显软化分类边界。B3b-main 配置已新增 `configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml`；B3b-optional 配置为 `configs/experiments/b3b_proton_c7_ce_emd_optional_3seed.yaml`。
- 论文数据分析链路与训练链路分开：数据分析默认使用全量 `Proton_C`，训练实验默认使用 7 分类子集 `Proton_C_7`。
- 本地 Windows 验证环境为 `timepix-local`；本地数据路径为 `D:\Project\Timepix\Data\Alpha_100`、`E:\C1Analysis\Proton_C`、`E:\C1Analysis\Proton_C_7`。
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

已观察到的结论：`resnet18_no_maxpool + conv1_kernel_size=2 + conv1_stride=1 + conv1_padding=0 + dropout=0.3` 在 A1 中表现最好。这里 no-maxpool 和 conv1 stem 是结构结论；A2 后续将 dropout 纳入训练超参搜索，因此最终 base 使用 A2 搜索得到的 `dropout=0.1`，不要把 `dropout=0.1` 写成 A1 结构结论。

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

阶段 3：oracle 控制诊断。

脚本：

```text
scripts/evaluate_oracle_complementarity.py
```

该阶段不训练新模型，而是重新加载 checkpoint，在 validation/test 上做确定性推理。关键决策是：A4b-3a 的纯 ToT 多 seed oracle control 使用 `a2_best_3seed`，因为它是当前已完成的 `Alpha_100 + ToT + resnet18_no_maxpool + A2 best` 三 seed基准组；A4b-3b 先使用 `a2_best_3seed` 的 seed42 ToT 与 `a4b_toa_transform_seed42` 的 `relative_minmax/no mask` candidate 做验证集/测试集复查。由于 `a2_best_3seed` 的历史配置仍写着 `Alpha` 和 `/root/autodl-tmp/Alpha`，服务器重放时要传 `--data-root /root/autodl-tmp/Alpha_100`，并用 `Alpha_100_ToT` split 复制出旧名称 `Alpha_ToT` 作为兼容别名。

当前结果：ToT-vs-ToT seed control 的 oracle gain 很小，validation 为 +2.33% ± 0.15%，test 为 +2.55% ± 0.06%；30 deg oracle gain 也只有 validation +2.55% ± 0.80%、test +1.15% ± 0.80%。相对地，ToT vs `relative_minmax/no mask` 的 oracle gain 在 validation/test 分别为 +10.19% 和 +11.03%，30 deg oracle gain 分别为 +27.08% 和 +25.52%。这说明 A4b-2.5 的互补性远大于普通随机 seed 多样性，后续问题应转为 selector/gate 能否从 validation 可用信息中学会何时信 candidate。

阶段 4：selector fusion。

脚本：

```text
scripts/evaluate_selector_fusion.py
```

该阶段不训练新的 ResNet，而是冻结 ToT baseline 与 `relative_minmax/no mask` candidate。当前重新编号为：A4b-4a rule-based selector；A4b-4b train-logit selector；A4b-4c validation-CV selector。旧的未编号 A4b-4 初版结果作废，后续按 4a/4b/4c 重新运行。所有版本都由 validation 选择规则/阈值/是否启用 selector，test 只做最终报告；若 validation 不支持 selector，脚本可以选择 `primary_only` 退回 ToT baseline。

当前结果：A4b-4a rule-based selector 选择 `entropy_adv_0p03`，test accuracy 70.97%，相对 ToT +0.50%，MAE 5.905 deg，macro-F1 0.658，选择率 14.51%。A4b-4b train-logit selector 选择 `threshold=0.95`，test accuracy 71.17%，相对 ToT +0.70%，MAE 5.890 deg，macro-F1 0.654，选择率 6.96%。A4b-4c validation-CV selector 选择 `threshold=0.95`，test accuracy 70.38%，相对 ToT -0.10%，MAE 6.009 deg，macro-F1 0.644，选择率 1.39%。因此可写成：简单 rule 和 train-logit selector 有小幅真实增益，但更严格的 validation-CV selector 未能稳定超过 ToT；互补性存在，可靠学习切换仍未完全解决。

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
- `epochs=25`
- `early_stopping_patience=5`

备注：B1-1 初版使用 20 epoch，但部分组合停止时准确率仍在上升，因此训练预算调整为 25 epoch。实际执行中用 `from20` continuation 方式只继续了 4 组未早停 run；该结果不等价于原生 `T_max=25` 训练，但足以判断 B1-1 的 validation-selected 最佳组合是否变化。

B1-1 搜索：

```text
learning_rate = [1e-4, 3e-4, 1e-3]
batch_size    = [64, 128, 256]
```

结果：20 epoch 旧结果和 from20 中继 25 epoch 结果均选择：

```text
learning_rate = 3e-4
batch_size    = 128
weight_decay  = 1e-4
dropout       = 0.1
scheduler     = cosine
eta_min       = 1e-7
```

备注：A1 最佳观测包含 `dropout=0.3`，但 B1 的结构继承只继承 no-maxpool 和 conv1 2/1/0；`dropout=0.1` 是沿用 A2/B1 训练超参选择，不写成 A1 结构参数。B1-2 已固定 `learning_rate=3e-4` 和 `batch_size=128` 后搜索 `weight_decay = [0, 1e-5, 1e-4]`。

B1-2 配置已新增：

```text
configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml
```

该配置显式将 `training.learning_rate` grid 限定为 `[0.0003]`、`training.batch_size` grid 限定为 `[128]`，只让 `training.weight_decay` 展开为 `0`、`1e-5`、`1e-4` 三组，避免继承 B1-1 的 `learning_rate × batch_size` 网格。

B1-2 结果：`weight_decay=1e-4` 取得最高 `val_accuracy=93.84%`，`test_accuracy=93.97%`，`test_mae=0.574`，`test_f1=0.9563`。`weight_decay=0` 很接近但略低，`weight_decay=1e-5` 明显更差。因此当前 B1 最佳组合为 `learning_rate=3e-4`、`batch_size=128`、`weight_decay=1e-4`、`dropout=0.1`、`scheduler=cosine`、`eta_min=1e-7`。

B1-best patience=8 正式配置：

```text
configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml
```

该配置不继承 B1-2，因为 B1-2 带有 `weight_decay` 搜索 grid；B1-best 独立写出固定配置，只展开 `training.seed=[42,43,44]`，用于报告 mean ± std。原 `configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml` 使用 `early_stopping_patience=5`，仅保留为早停过激的历史诊断配置。patience=8 正式结果为 `Test Acc = 93.26 ± 1.64%`、`MAE = 0.640 ± 0.161`、`Macro-F1 = 0.952 ± 0.011`。

## A4b 当前后续安排

A4b-4 结果已经将问题从“是否存在互补性”推进到“能否可靠识别何时切换”：

- A4b-4a rule selector：`entropy_adv_0p03`，Test Acc 70.97%，相对 ToT +0.50%，MAE/F1 同时略有改善。
- A4b-4b train-logit selector：Test Acc 71.17%，相对 ToT +0.70%，但因使用 train split expert logits 训练 selector，只作为探索性对照。
- A4b-4c validation-CV selector：Test Acc 70.38%，未超过 ToT，是更严格但负面/中性的 learned-selector 结果。
- Oracle 仍为 81.51%，说明互补性存在但当前选择器远未充分利用。

最新编号和状态：

- A4b-4d 已完成：seed42 `entropy_adv_0p03` rule selector 的 switch rate 为 14.51%，接近 oracle switch rate 12.43%，但 switch precision 只有 47.95%，146 个 switched samples 中 beneficial 70、harmful 69、neutral 7。结论是瓶颈不是切换太少，而是 beneficial/harmful switches 难以用 entropy/confidence 区分；该规则帮助 30 deg 和 45 deg，但伤害 15 deg 和 60 deg。
- A4b-4e 已完成三 seed：validation-selected rule selector 均优于 ToT baseline，Test Acc 从 70.44%±0.15 提高到 71.44%±0.57，MAE 从 5.949 降到 5.835，Macro-F1 从 0.636 提到 0.645。但三个 seed 选中的 rule 不一致，且 oracle 仍为 79.75%±1.96，说明 rule selector 有稳定小幅收益但远未利用全部互补性。
- A4b-5 已完成三 seed：validation-selected gate 将 Test Acc 从 ToT primary 的 70.44±0.15% 提高到 72.17±1.72%，MAE 从 5.949±0.068 降到 5.661±0.320，Macro-F1 从 0.636±0.009 提高到 0.662±0.027。它是强 test reference；但 seed43 选中了 `train` fit，论文中需说明偏乐观风险，保守口径可重点报告 `class_aware_prob_val-cv`。
- A4b-6 已完成三 seed：validation-selected residual 将 Test Acc 提高到 71.44±1.01%，MAE 降到 5.800±0.077，Macro-F1 提高到 0.656±0.010。它优于 ToT baseline，并作为 validation-selected expert-level 后处理融合系统；但它不是端到端多模态架构。
- A4c 第一阶段已完成 9 个 run。三 seed汇总：`dual_stream_concat_aux` 为 Val Acc `69.90±1.34%`、Test Acc `72.10±1.35%`、Test MAE `5.631±0.333`、Macro-F1 `0.686±0.017`；`dual_stream_gmu_aux` 为 Val Acc `70.20±0.67%`、Test Acc `71.94±0.51%`、Test MAE `5.721±0.009`、Macro-F1 `0.691±0.009`；`toa_conditioned_film` 为 Val Acc `70.43±0.95%`、Test Acc `71.60±1.21%`、Test MAE `5.775±0.236`、Macro-F1 `0.678±0.021`。最终端到端架构选择 GMU 时不能使用 test 结果反选；应写成基于 validation 侧均衡指标：GMU Val Macro-F1 与 FiLM 接近，Val MAE 更好，且 gate 机制符合 A4b 的选择性辅助结论。test 只用于最终泛化说明。
- A4c-4 已完成 6 个 run。`freeze_experts=true` 为 Val Acc `70.50±1.00%`、Test Acc `71.84±1.88%`、Test MAE `5.905±0.332`、Macro-F1 `0.660±0.015`、Candidate Gate `65.11±24.63%`；`freeze_experts=false` 为 Val Acc `68.80±1.36%`、Test Acc `70.15±2.34%`、Test MAE `6.123±0.554`、Macro-F1 `0.643±0.049`、Candidate Gate `56.53±12.19%`。注意 `gate_candidate` 是 candidate expert 权重，不是单独 ToA 通道权重。结论：`freeze=true` 可作为 warm-start gate 对照，`freeze=false` 不稳定；A4c-4 不是当前最佳融合方案。
- A4b-7：ToA-only relative controls。
- A4b-8：ToT image + ToA scalar physical features。
- GMU 已确定为论文主推端到端多模态架构；FiLM 和 warm-start expert gate 作为对照；MMTM 继续选做但当前不优先扩展。

## 当前可讲的论文故事线

建议主线：

1. Timepix/Timepix3 探测器输出稀疏像素矩阵，轨迹形态和 ToT/ToA 分布携带入射极角信息。
2. 先在 alpha ToT 单模态上确定适合稀疏探测器矩阵的 ResNet18 结构。
3. 再搜索训练超参数，固定统一训练预算，避免后续消融和模型对比混入调参因素。
4. 在统一训练设置下比较多种 CNN/现代视觉主干和 ViT-Tiny；当前 A3 记录支持 `resnet18_no_maxpool` 作为主干。
5. 比较 ToT、ToA、ToT+ToA；当前 A4 记录显示 ToT 单模态最好，ToT+ToA 没有提升。A4b 进一步说明，相对 ToA 表达和 fixed late logit fusion 仍不足以稳定超过 ToT，但 ToT 与 `relative_minmax/no mask` candidate 具有明显 oracle 互补性。后续 selector/gate 实验中，A4b-5 sample-wise gated late fusion 已获得三 seed 正收益，是当前最有希望写入论文主线的选择性融合结果；A4b-6 residual fusion 可作为对照。
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
