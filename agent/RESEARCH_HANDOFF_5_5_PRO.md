# Timepix 极角识别课题研究交接文档

本文档用于向研究型模型交接本课题的研究背景、实验体系、当前结论和后续论文写作任务。旧版交接文档已归档为 `agent/RESEARCH_HANDOFF_5_5_PRO.old.md`；旧文档保留历史阶段性记录，但其中部分“正在进行/后续安排”已不再代表当前状态。

## 一、课题概述

本课题研究基于 Timepix/Timepix3 像素探测器响应矩阵的带电粒子入射极角识别。输入数据是粒子事件在局部像素平面上的稀疏矩阵，主要模态包括：

- `ToT`：Time-over-Threshold，近似反映能量沉积、电荷量或信号强度相关信息。
- `ToA`：Time-of-Arrival，反映到达时间结构或相对时间分布。

当前训练主线包括两类数据：

| 数据集 | 任务 | 说明 |
| --- | --- | --- |
| `Alpha_100` | Alpha 粒子 4 类极角分类 | 支持 ToT 与 ToA，是 A 系列主线。 |
| `Proton_C_7` | Proton/C 七分类角度识别 | 只支持 ToT，是 B 系列主线。 |

研究核心问题不是简单追求一个最高 accuracy，而是系统回答：

1. Timepix 稀疏矩阵适合什么 CNN 结构？
2. ToT、ToA 及其相对时间表达分别提供什么信息？
3. ToA 是否适合直接 early fusion，还是更适合作为选择性辅助模态？
4. 物理标量特征是否能补充 CNN 表征？
5. 角度类别具有有序性，普通 CE one-hot 是否忽略了物理角度误差？

## 二、建议阅读顺序

必读：

```text
agent/RESEARCH_HANDOFF_5_5_PRO.md
agent/EXPERIMENT_LOG.md
agent/PHYSICS_CONTEXT.md
configs/README.md
agent/FILE_MAP.md
```

需要工程细节时再读：

```text
agent/CODE_CONTEXT.md
agent/ARCHITECTURE.md
agent/NEW_SYSTEM_GUIDE.md
agent/SERVER_TRAINING.md
```

历史计划文档只用于理解方案来源，不作为当前状态依据：

```text
agent/A4B_IMPLEMENTATION_PLAN.md
agent/A4B_SELECTOR_FUSION_PLAN.md
agent/EXPERIMENT_LOG.old.md
agent/EXPERIMENT_LOG.pre_rebuild.old.md
agent/RESEARCH_HANDOFF_5_5_PRO.old.md
configs/README.old.md
```

## 三、全局实验规范

### 数据与 split

当前正式 Alpha 主线使用 `Alpha_100`，不再使用 `Alpha_50` 作为正式结果线。`Alpha_50` 只保留为历史/对照数据集。

Alpha ToT split：

```text
outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json
```

Alpha ToT-ToA paired split：

```text
outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

paired split 是从 ToT split 复制得到的，因为 `Alpha_100` 中 ToT 与 ToA 样本一一对应，manifest 中保存的是 normalized sample key。这样 A1/A2/A3/A4/A4b/A4c/A5/A6 可以在同一数据划分逻辑上展开。

Proton/C 训练统一使用：

```text
Proton_C_7
```

全量 `Proton_C` 只用于论文数据分析和近垂直分辨极限分析。

### 模型选择规则

- 模型、超参数、特征组、loss、gate 或融合策略只能依据 validation 指标选择。
- `test` 只用于最终泛化报告，不能反向选择模型。
- 单 seed 结果是 screening / diagnostic；正式结论优先使用 `training.seed=[42,43,44]` 的 mean ± std。

该原则在 A4c GMU、A5d main_5feat 和 B3b loss 选择中尤其重要。

## 四、Alpha 主线结论

### A1/A2/A3：主干与训练基线

A1 结论：

```text
model             = resnet18_no_maxpool
conv1_kernel_size = 2
conv1_stride      = 1
conv1_padding     = 0
```

A1 中 `dropout=0.3` 是结构网格内的观测结果；后续正式 base 的 dropout 由 A2 超参搜索确定。

A2 best：

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

A3 主干对比支持 `resnet18_no_maxpool` 作为当前 Alpha 主干。ViT-Tiny 表现较弱，说明当前小样本、稀疏探测器矩阵更依赖 CNN 的局部归纳偏置。

### A4：模态基础对比

A4 直接比较 ToT、ToA 和 raw/log1p ToT+ToA：

| Modality | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ToT | 69.53% | 70.48% | 5.96 deg | 0.646 |
| ToT + ToA raw/log1p | 64.04% | 65.90% | 6.92 deg | 0.553 |
| ToA | 59.34% | 60.14% | 8.81 deg | 0.477 |

结论：ToT 是强主模态；raw ToA 不能无条件作为第二图像通道 early fusion；ToA 单模态尤其不利于 30 deg 类别。

### A4b：relative ToA 与选择性融合

A4b 的证据链：

1. relative ToA 表达比 raw/log1p ToA early fusion 更合理，但仍不超过 ToT。
2. fixed late logit fusion 在 validation 上选择 `alpha_toa=0`，说明全局固定 ToA 权重不可靠。
3. oracle diagnostics 显示 ToT 与 `relative_minmax, no mask` candidate 存在显著互补性，尤其 30 deg。
4. rule selector 能稳定小幅利用互补性，但离 oracle 上限很远。
5. sample-wise gated late fusion 与 residual gated fusion 进一步验证了选择性利用 candidate 的合理性。

A4b-5 结果：

| 方法 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ToT primary | 69.03±0.46% | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| A4b-5 validation-selected gate | 71.40±0.59% | 72.17±1.72% | 5.661±0.320 | 0.662±0.027 |

A4b-6 结果：

| 方法 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| A4b-6 validation-selected residual | 71.76±0.72% | 71.44±1.01% | 5.800±0.077 | 0.656±0.010 |

解释：

- A4b-6 是 validation-selected expert-level 后处理融合系统。
- A4b-5 是强 test reference，但包含 train-fit gate 的偏乐观风险，应谨慎表述。

### A4c：端到端双模态架构

A4c 重新训练 ToT/relative-ToA 双分支模型，研究 feature-level fusion 是否更好。

结果：

| 模型 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `dual_stream_concat_aux` | 69.90±1.34% | 72.10±1.35% | 5.631±0.333 | 0.686±0.017 |
| `dual_stream_gmu_aux` | 70.20±0.67% | 71.94±0.51% | 5.721±0.009 | 0.691±0.009 |
| `toa_conditioned_film` | 70.43±0.95% | 71.60±1.21% | 5.775±0.236 | 0.678±0.021 |

最终端到端多模态架构：`A4c-2 dual_stream_gmu_aux`。

选择理由不能写成“test 最好”。正确口径是：

- GMU 的 validation accuracy 不弱。
- GMU 的 Val Macro-F1 与 FiLM 几乎持平，差异远小于 std。
- GMU 的 Val MAE 在 A4c 端到端模型中最好。
- GMU 的 gate 机制与 A4b 得出的物理结论一致：ToT 是主模态，relative ToA 是选择性辅助模态。

测试集结果可用于最终泛化说明，但不能用于反选 GMU。

### A5：物理/手工标量特征

A5 不再属于 A4b，它是独立的物理标量特征融合阶段。主问题是：低维物理标量能否补充 ToT CNN，并提供更可解释的判别依据。

核心设计：

- 图像输入：ToT only。
- scalar source：ToT + ToA。
- ToA 只作为标量来源，不作为图像通道输入。

A5a 结论：手工特征有独立判别力，但弱于 CNN；Geometry 最重要，其次是 ToT，ToA 标量有辅助信号但不是主导。

A5b 结论：simple concat 没有证明稳定提升。

A5c 结论：gated 比 concat 更适合低维标量。

A5d 三 seed：

| 设置 | Val Acc | Val MAE | Val Macro-F1 | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 ToT baseline | 69.03±0.46% | 6.424±0.127 | 0.622±0.007 | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| `main_5feat` | 70.23±0.30% | 6.209±0.083 | 0.651±0.008 | 70.18±0.95% | 5.875±0.231 | 0.646±0.005 |
| `toa_only_diag` | 70.70±0.31% | 6.129±0.040 | 0.650±0.005 | 70.05±0.21% | 5.959±0.009 | 0.641±0.003 |

严格按 validation accuracy，A5d 内部最佳是 `toa_only_diag`。`main_5feat` 的 Test MAE/F1 更好，但不能用 test 反选。A5 的论文口径应是：物理标量有解释性与有限辅助作用，但未稳定提升总体 Test Acc。

### A6：Alpha 有序角度损失

当前状态：A6a/A6b 已完成，A6c 不推进。

A6 设计对齐 B3：固定 `Alpha_100 + ToT + resnet18_no_maxpool + A2 best`，只比较 loss / label strategy。

A6a 包含：

```text
Gaussian soft CE: sigma = 5.0, 7.5, 10.0
CE+ExpectedMAE: lambda = 0.02, 0.05, 0.10
CE+EMD: lambda = 0.02, 0.05, 0.10
```

CE one-hot baseline 不重跑，直接复用 A2-best seed42 与 three-seed baseline。

A6a 结果没有复现 Proton B3b 那样的强 loss 改进。按 validation selection，主候选是 `CE+EMD lambda=0.02`：Val Acc 与 A2 CE baseline 持平，Val MAE 和 Val Macro-F1 略好，但这个收益只是 tie-break 级别。A6b 随后只验证 `CE+EMD lambda=0.02` 三 seed，结果显示该优势不稳定且弱于 A2 CE baseline：A2 CE baseline 的 Val Acc/MAE/Macro-F1 为 69.03±0.46% / 6.424±0.127 / 0.622±0.007，A6b 为 68.33±1.15% / 6.618±0.424 / 0.609±0.034。逐类看，A6b 没有改善 30 deg，反而主要拉低 60 deg。结论：Alpha-ToT 后续继续使用 A2 CE one-hot，不采用 A6 的有序损失，A6c 不迁移到 `A4c-2 dual_stream_gmu_aux`。

### A7：最终多模态架构的手工物理特征确认

A7 已配置，目标是回答最后一个组件问题：

```text
在最终端到端多模态架构 GMU_aux 上，
A5 选出的五维物理标量 main_5feat 是否还能提供额外补充？
```

固定：

- Model: `A4c-2 dual_stream_gmu_aux`
- Input: `ToT + relative_minmax ToA, no mask`
- Loss: `CE one-hot`
- Training config: A2 best
- Handcrafted fusion: `gated`
- Seeds: 42/43/44

对照关系：

- `A7-0`：复用 A4c GMU，不重跑。
- `A7-1`：运行 `GMU_aux + main_5feat gated`。

`main_5feat`：

```text
active_pixel_count
bbox_fill_ratio
ToT_density
ToA_span
ToA_major_axis_corr_abs
```

关键决策：不跑 `GMU + CE+EMD`、不跑 `GMU + CE+EMD + handcrafted`、不跑 `toa_only_diag`、不新增 feature group 或架构。A7 只用 validation 判断 `main_5feat` 是否进入最终模型；test 只用于最终泛化说明。

## 五、Proton_C_7 主线结论

### B1：训练超参数与 baseline

B1 固定 Alpha A1 的结构适配结论，搜索 Proton_C_7 的训练超参数。

B1-best patience=8：

```text
model             = resnet18_no_maxpool
conv1_kernel_size = 2
conv1_stride      = 1
conv1_padding     = 0
learning_rate     = 3e-4
batch_size        = 128
weight_decay      = 1e-4
dropout           = 0.1
scheduler         = cosine
eta_min           = 1e-7
epochs            = 25
early_stopping_patience = 8
```

结果：

| 指标 | Mean ± Std |
| --- | ---: |
| Val Acc | 92.94±1.81% |
| Test Acc | 93.26±1.64% |
| Test MAE | 0.640±0.161 |
| Test Macro-F1 | 0.952±0.011 |

旧 patience=5 版本只作为 early stopping 过激诊断。

### B2：手工特征验证

B2 固定 B1-best，只验证 ToT-only low-redundancy handcrafted features。

结论：

- `geometry_lowcorr` 只有极小正收益。
- `ToT_density` 在 concat 下明显破坏结果。
- gated 可以抑制坏特征的负面影响，但没有证明手工特征能显著提升 Proton_C_7。
- B2c three-seed 不优先推进。

### B3：有序角度损失

B3 固定 B1-best，不启用手工特征，只比较 loss / label strategy。

B3b 结果：

| 设置 | Val Acc | Val MAE | Val F1 | Test Acc | Test MAE | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1-best CE onehot | 92.94±1.81% | 0.676±0.184 | 0.949±0.012 | 93.26±1.64% | 0.640±0.161 | 0.952±0.011 |
| B3b-main CE+ExpectedMAE lambda=0.05 | 94.20±0.05% | 0.550±0.002 | 0.958±0.000 | 94.38±0.08% | 0.532±0.010 | 0.960±0.001 |
| B3b-optional CE+EMD lambda=0.05 | 94.20±0.16% | 0.545±0.015 | 0.957±0.001 | 94.35±0.09% | 0.532±0.011 | 0.959±0.001 |

结论：`CE+ExpectedMAE lambda=0.05` 是 Proton_C_7 当前推荐损失函数；`CE+EMD lambda=0.05` 是强有序损失对照。

## 六、论文主线建议

### Alpha 叙事

1. Timepix 稀疏矩阵不是自然图像，ResNet18 需要结构适配。
2. ToT 是强主模态；raw ToA 无条件 early fusion 会破坏性能。
3. relative ToA 虽然自身不强，但与 ToT 存在 oracle 互补性，尤其对 30 deg。
4. A4b 证明选择性后处理融合可以利用一部分互补性。
5. A4c 证明端到端双流 GMU 架构能从 feature level 利用 ToA 辅助信息，提升类别均衡性。
6. A5 显示物理标量有解释性辅助价值，但不是稳定 accuracy gain 的主线。
7. A6b 显示 `CE+EMD lambda=0.02` 在 Alpha-ToT 上不稳定且弱于 A2 CE baseline；Alpha 后续继续使用 CE one-hot。
8. A7 只做最终组件确认：在 GMU + CE one-hot 上验证 A5 `main_5feat` gated 是否仍有补充价值。

### Proton_C_7 叙事

1. B1 得到高性能 ToT-only baseline。
2. B2 显示手工标量主要与 CNN 表征冗余。
3. B3 显示角度有序性损失显著改善 Proton_C_7 的相邻大角度混淆与训练稳定性。

## 七、当前待协助问题

建议 5.5 Pro 重点协助：

1. 检索 Timepix/Timepix3 ToT/ToA 在粒子识别、轨迹重建、角度估计中的相关文献。
2. 提炼 ToT 与 ToA 的物理互补性，解释为什么 ToA 不适合 raw early fusion，却适合作为选择性辅助信息。
3. 帮助构建论文方法章节：结构适配、模态融合、物理标量、角度有序损失。
4. 帮助撰写 A4c GMU 的理论动机与实验解释，特别强调不能用 test 反选模型。
5. 帮助解释 A6 负结果：为什么 Proton_C_7 上有效的有序损失没有迁移到 Alpha-ToT，并说明 A6c 不推进的合理性。
6. 帮助解释 A7 结果：如果 `main_5feat` 有效，说明低维物理摘要能补充 GMU；如果无效，说明 GMU 图像分支已经吸收大部分可表达信息。
7. 帮助组织 Proton_C_7 的 B3 正结果，突出 expected-angle MAE auxiliary loss 的物理意义。

## 八、重要注意事项

- 不要把 `Alpha_50` 写成当前正式训练主线。
- 不要把 `Proton_C` 写成训练主线；训练使用 `Proton_C_7`。
- 不要用 test 指标解释模型选择过程。
- 不要把 A4b-6 和 A4c GMU 混成单一 winner：A4b-6 是 expert-level 后处理系统，A4c GMU 是 final end-to-end multimodal architecture。
- 不要把 A5d `main_5feat` 写成 validation-selected best；A5d validation-best 是 `toa_only_diag`。
- A6b 已完成且为负结果；不要把 A6 的有序损失迁移到 GMU，多模态主线继续使用 A4c GMU + CE one-hot。
- A7 只验证 `GMU + CE one-hot + main_5feat gated`，不扩展 `toa_only_diag`、loss 或新架构。

## 九、建议给 5.5 Pro 的初始提示

```text
请阅读 agent/RESEARCH_HANDOFF_5_5_PRO.md、agent/EXPERIMENT_LOG.md、
agent/PHYSICS_CONTEXT.md、configs/README.md 和 agent/FILE_MAP.md。

本课题研究基于 Timepix/Timepix3 探测器 ToT/ToA 像素矩阵的带电粒子入射极角识别。
当前正式数据主线为 Alpha_100 和 Proton_C_7。Alpha 主线已完成 A1-A6 和 A4/A4b/A4c
多模态融合分析，A6b 证明 Alpha 有序损失分支不采用；A7 正在验证最终 GMU 多模态架构
是否还需要 A5 的 main_5feat 物理标量。Proton_C_7 主线已完成 B1-B3，
B3b 证明 CE+ExpectedMAE lambda=0.05 是当前推荐损失。

请协助进行文献调研、论文结构设计、创新点提炼和实验结果解释。请特别注意：
模型选择只依据 validation，test 只用于最终报告；A4c GMU 是基于 validation 侧均衡指标
与物理解释选择的端到端多模态架构，而不是由 test 反选得到。
```
