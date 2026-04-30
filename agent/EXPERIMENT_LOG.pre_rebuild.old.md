# Timepix 实验日志与决策索引

本文档是当前项目的权威实验日志，用于记录实验编号、阶段目的、完成状态、关键结果和方法学决策。旧版长日志已归档为 `agent/EXPERIMENT_LOG.old.md`；旧日志保留全部历史讨论和命令细节，但不再作为当前状态判断依据。

本文档采用中文记录实验目的、流程和决策；配置名、命令、模型名、指标名等专业标识保留英文。服务器命令默认按 Linux 环境书写，本地检查命令才使用 Windows 路径。

## 一、记录原则

1. 所有正式实验均遵循 validation-driven model selection。`test` 指标只用于最终泛化报告，不用于选择模型、超参数、融合策略、loss 或特征组。
2. 单 seed 实验只作为 screening / diagnostic，正式论文结论优先使用 three-seed mean ± std。
3. 对比实验应固定 split、模型结构、训练预算和评价指标，只改变当前阶段研究的问题变量。
4. 每次新增实验配置或改变实验决策，必须同步记录训练命令和汇总命令。命令索引集中维护在 `configs/README.md`。
5. 旧配置、旧命名和废弃方案保留可追溯性，但论文写作时按本文档的当前编号和解释口径使用。

## 二、当前权威状态

截至 2026-04-30：

| 主线 | 当前状态 | 结论摘要 |
| --- | --- | --- |
| Alpha A1-A3 | 已完成 | `resnet18_no_maxpool` 与 `conv1_kernel_size=2, stride=1, padding=0` 是当前 Alpha-ToT 主结构；A3 支持 ResNet18 no-maxpool 作为主干。 |
| Alpha A4/A4b/A4c | 阶段性完成 | ToT 是最强单模态；raw ToA early fusion 失败；relative ToA 有互补性；A4b-6 是 validation-selected expert-level 后处理融合系统；A4c-2 `dual_stream_gmu_aux` 是论文主推端到端多模态架构。 |
| Alpha A5 | 已完成 | 低维物理标量具有解释性和辅助信号，`gated` 融合优于 simple concat，但未稳定提升 Test Acc。 |
| Alpha A6 | A6a 正在运行 | A6 是 Alpha 版 B3，固定 Alpha-ToT baseline，仅筛选 angle-ordinal loss / label strategy。CE one-hot baseline 复用 A2-best，不重跑。 |
| Proton B1-B3 | 已完成 | B1-best patience=8 是 Proton_C_7 baseline；B2 手工特征线收口；B3b 证明 `CE+ExpectedMAE lambda=0.05` 是当前 Proton_C_7 推荐损失。 |
| 数据分析 D1-D3 | 已实现 | 数据分析链路独立于训练主线，默认分析 `Alpha_100` 与全量 `Proton_C`，不把训练子集 `Proton_C_7` 当作全量分析对象。 |

## 三、数据、划分与评价约定

### 3.1 数据集命名

| 名称 | 用途 | 说明 |
| --- | --- | --- |
| `Alpha_100` | 当前 Alpha 正式训练主线 | 100 x 100，支持 `ToT` 与 `ToA`。 |
| `Alpha_50` | 历史/对照数据集 | 曾短暂使用，但效果与故事线不如 `Alpha_100`，不作为当前正式主线。 |
| `Proton_C_7` | Proton/C 七分类训练主线 | 只支持 `ToT`，用于 B 系列训练实验。 |
| `Proton_C` | 论文数据分析 | 全量 Proton/C 数据集，用于数据统计和近垂直分辨极限分析，不作为训练入口。 |

本地 Windows 数据路径：

```text
Alpha_100  -> D:\Project\Timepix\Data\Alpha_100
Proton_C   -> E:\C1Analysis\Proton_C
Proton_C_7 -> E:\C1Analysis\Proton_C_7
```

### 3.2 Split 决策

Alpha ToT 历史 split 已恢复并规范命名为：

```text
outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json
```

Alpha paired ToT-ToA split 规范命名为：

```text
outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

该 paired split 从历史 ToT split 复制得到，不重新随机生成。原因是 `Alpha_100` 中 ToT 与 ToA 文件一一对应，split manifest 保存的是去掉 ToT/ToA 标记后的 normalized sample key，因此同一划分内容可用于 ToT、ToA 和 ToT+ToA。

Proton_C_7 正式 split：

```text
outputs/splits/Proton_C_7_ToT_seed42_0.8_0.1_0.1.json
```

### 3.3 随机性

- `split.seed` 控制数据划分。
- `training.seed` 控制模型初始化、DataLoader shuffle 和训练随机性。
- 正式 three-seed 认证固定 `split.seed=42`，展开 `training.seed=[42,43,44]`。

## 四、Alpha 主线

### A1：ResNet18 结构适配

目的：确定 ResNet18 如何适配 Timepix 稀疏探测器矩阵。

固定条件：`Alpha_100`、ToT、classification、`cross_entropy`、`onehot`、无 handcrafted features。

比较因素：

- `resnet18_original`
- `resnet18_maxpool`
- `resnet18_no_maxpool`
- `conv1_kernel_size = 2 / 3 / 5`
- `conv1_stride = 1 / 2`
- `dropout = 0 / 0.1 / 0.3`

关键结论：

- 结构层面选择 `resnet18_no_maxpool`。
- stem 选择 `conv1_kernel_size=2`、`conv1_stride=1`、`conv1_padding=0`。
- A1 中 `dropout=0.3` 是该结构网格内的观测结果；正式后续 base 的 dropout 由 A2/B1 训练超参搜索决定，不能把 `dropout=0.1` 写成 A1 结构结论。

### A2：Alpha-ToT 训练超参数搜索

目的：在固定 A1 结构后搜索训练超参数，为后续 Alpha 消融实验提供统一训练配置。

A2 best base：

```text
model                 = resnet18_no_maxpool
conv1_kernel_size     = 2
conv1_stride          = 1
conv1_padding         = 0
learning_rate         = 4.3878e-05
weight_decay          = 4.7324e-04
batch_size            = 32
eta_min               = 1.6433e-07
dropout               = 0.1
scheduler             = cosine
epochs                = 25
early_stopping_patience = 8
mixed_precision       = true
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

### A2-best：三 seed 基线认证

目的：认证 A2 best 在训练随机性下的稳定性，并作为后续 ToT baseline 和 A4b seed-control 的基准池。

配置：

```text
configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml
```

该结果在后续 A4b-3a/b、A4b-4/5/6 中作为 primary ToT expert 使用。

### A3：主干模型对比

目的：在统一 Alpha-ToT 条件下比较不同主干，确定后续实验主干。

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

关键结论：

- A3 支持 `resnet18_no_maxpool` 作为当前主干。
- ViT-Tiny 在稀疏小样本 Timepix 矩阵上表现较弱，符合其缺少 CNN 局部归纳偏置的预期。
- A3 进一步巩固了后续 A4/A5/A6 固定 ResNet18 no-maxpool 的合理性。

### A4：ToT / ToA / ToT+ToA 模态基础对比

目的：验证 Alpha 数据集中 ToT 与 ToA 对极角识别的贡献。

固定条件：`Alpha_100`、`resnet18_no_maxpool`、A2 best、无 handcrafted features、paired split。

结果摘要：

| Modality | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ToT | 69.53% | 70.48% | 5.96 deg | 0.646 |
| ToT + ToA raw/log1p | 64.04% | 65.90% | 6.92 deg | 0.553 |
| ToA | 59.34% | 60.14% | 8.81 deg | 0.477 |

结论：ToT 是最强单模态；raw/log1p ToA 直接作为图像通道 early fusion 会降低性能；ToA 单模态尤其不利于 30 deg 类别。

### A4b：ToA 选择性辅助融合

A4b 的核心问题：在 ToA 单模态弱、raw early fusion 失败的情况下，relative ToA 是否可以作为选择性辅助信号补充 ToT。

#### A4b-1：relative ToA early fusion

比较 `relative_minmax`、`relative_centered`、`relative_rank` 以及 `add_hit_mask`。

结论：

- relative ToA 明显优于 raw/log1p ToA early fusion。
- 所有 early fusion 变体仍未超过 ToT baseline。
- `relative_minmax, no mask` 虽然整体 Test Acc 不高，但 30 deg F1 与后续 oracle 互补性最值得关注。

#### A4b-2：fixed late logit fusion

固定权重 `alpha_toa` 在 validation 上选择，结果选择 `alpha_toa=0`。结论是全局固定 ToA 权重不可靠，不能用 test 上个别非零 alpha 的小幅提升反向选择模型。

#### A4b-2.5 / A4b-3：oracle complementarity

关键结果：

- ToT-vs-ToT seed control 的 oracle gain 较小：test 约 +2.55%，30 deg 约 +1.15%。
- ToT vs `relative_minmax, no mask` 的 oracle gain 明显更大：test +11.03%，30 deg +25.52%。

解释：`relative_minmax, no mask` candidate 的互补性不能简单归因于随机 seed 多样性，说明 ToA relative-time 信息确有局部补充价值，尤其在 30 deg 类别。

#### A4b-4：hard selector

包括 rule selector、train-logit selector 和 validation-CV selector。

结论：

- rule selector 三 seed 稳定小幅提升，但规则形式不稳定。
- train-logit selector 有探索性收益，但可能受 train split expert confidence 过度自信影响。
- validation-CV selector 更严格但未稳定超过 ToT。
- A4b-4d switch diagnostics 表明瓶颈不是切换不足，而是 beneficial/harmful switch 难以通过 entropy/confidence 区分。

#### A4b-5：sample-wise gated late fusion

定位：frozen-expert decision-level gated fusion，不训练新 ResNet。

三 seed 主结果：

| 方法 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ToT primary | 69.03±0.46% | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| Candidate only | 68.03±1.08% | 68.75±1.52% | 6.546±0.313 | 0.626±0.016 |
| A4b-5 validation-selected gate | 71.40±0.59% | 72.17±1.72% | 5.661±0.320 | 0.662±0.027 |

注意：seed43 选中 `train` fit，因此完整 validation-selected 表含偏乐观风险。论文中更保守时应单独报告 `val-cv` 系列。

#### A4b-6：residual gated fusion

定位：validation-selected expert-level residual post-processing system。

公式：

```text
logits_final = logits_tot + residual_weight * (logits_candidate - logits_tot)
```

三 seed 主结果：

| 方法 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ToT primary | 69.03±0.46% | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| A4b-6 validation-selected residual | 71.76±0.72% | 71.44±1.01% | 5.800±0.077 | 0.656±0.010 |

结论：A4b-6 是当前性能导向的 frozen-expert 后处理融合系统；它不是端到端多模态网络架构。

### A4c：端到端双模态融合

A4c 与 A4b 的区别：A4b-5/6 是 frozen-expert 后处理；A4c 重新训练 ToT/ToA 图像分支，让模型在 feature level 学习融合。

固定输入：`ToT + relative_minmax ToA, no mask`。

第一阶段模型：

| 编号 | model.name | 定位 |
| --- | --- | --- |
| A4c-1 | `dual_stream_concat_aux` | 双流 feature concat + auxiliary heads。 |
| A4c-2 | `dual_stream_gmu_aux` | 双流 GMU gate + auxiliary heads。 |
| A4c-3 | `toa_conditioned_film` | ToA-conditioned FiLM modulation。 |

三 seed 结果：

| 模型 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `dual_stream_concat_aux` | 69.90±1.34% | 72.10±1.35% | 5.631±0.333 | 0.686±0.017 |
| `dual_stream_gmu_aux` | 70.20±0.67% | 71.94±0.51% | 5.721±0.009 | 0.691±0.009 |
| `toa_conditioned_film` | 70.43±0.95% | 71.60±1.21% | 5.775±0.236 | 0.678±0.021 |

最终端到端多模态架构决策：

- 不能用 test 结果选择 GMU。
- 选择 `dual_stream_gmu_aux` 的依据是 validation 侧均衡指标、Val MAE、稳定性与结构解释。
- GMU Val Macro-F1 与 FiLM 接近，差异小于标准差量级；GMU Val MAE 在 A4c 端到端模型中最好。
- GMU gate 与 A4b 得出的物理结论一致：ToT 是主模态，relative ToA 是选择性辅助模态。

论文口径：

```text
A4b-6 residual gated fusion 是性能导向的 expert-level 后处理融合系统；
A4c-2 dual_stream_gmu_aux 是论文主推的 final end-to-end ToT/ToA multimodal architecture。
```

#### A4c-4：warm-started expert gate

结果：

| 设置 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| `freeze_experts=true` | 70.50±1.00% | 71.84±1.88% | 5.905±0.332 | 0.660±0.015 |
| `freeze_experts=false` | 68.80±1.36% | 70.15±2.34% | 6.123±0.554 | 0.643±0.049 |

结论：冻结 expert 的 warm-start gate 有一定效果，但不超过 A4b-5 或 A4c-1/2；解冻 expert 不稳定。A4c-4 作为对照，不作为最终架构。

### A5：物理/手工标量特征融合

A5 的问题：低维物理标量能否补充 ToT CNN 图像特征，并提供可解释的角度判别依据。

关键固定：

- image input = ToT only
- scalar feature source = ToT + ToA
- 不让 ToA 图像通道进入 A5 主模型
- 不参考 `timepix/analysis/` 的特征实现，训练链路在 `timepix/data/features.py` 中独立实现。

#### A5a：handcrafted feature screening

使用 `RandomForest`、one-vs-rest `LogisticRegression` 和 validation permutation importance 进行筛选。结论：

- 手工特征本身有角度判别力，但弱于 CNN。
- Geometry 最重要，其次是 ToT；ToA 标量有补充但不是主导。
- 相关性风险较高，因此不将 12 维候选全部塞入 CNN。

#### A5b：CNN + low-redundancy concat pilot

四组：

```text
geometry_lowcorr:
  active_pixel_count
  bbox_fill_ratio

geometry_plus_tot_lowcorr:
  active_pixel_count
  bbox_fill_ratio
  ToT_density

toa_lowcorr_diagnostic:
  ToA_span
  ToA_major_axis_corr_abs

geometry_plus_tot_plus_toa_lowcorr:
  active_pixel_count
  bbox_fill_ratio
  ToT_density
  ToA_span
  ToA_major_axis_corr_abs
```

结论：simple concat 没有证明能稳定提升 Alpha ToT CNN。

#### A5c：gated diagnostic

镜像 A5b 的四组特征，只把 fusion mode 从 `concat` 改为 `gated`。结论：

- gated 比 simple concat 更适合这批低维特征。
- seed42 上五维 `geometry_plus_tot_plus_toa_lowcorr_gated` 同时改善 Test Acc、MAE 和 Macro-F1，但属于小幅正结果。

#### A5d：three-seed verification

两组：

| 组别 | 特征 | 定位 |
| --- | --- | --- |
| `main_5feat` | `active_pixel_count; bbox_fill_ratio; ToT_density; ToA_span; ToA_major_axis_corr_abs` | 物理上完整的低冗余五维组。 |
| `toa_only_diag` | `ToA_span; ToA_major_axis_corr_abs` | A5c validation-best ToA-only 标量诊断组。 |

三 seed 结果：

| 设置 | Val Acc | Val MAE | Val Macro-F1 | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 ToT baseline | 69.03±0.46% | 6.424±0.127 | 0.622±0.007 | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| `main_5feat` | 70.23±0.30% | 6.209±0.083 | 0.651±0.008 | 70.18±0.95% | 5.875±0.231 | 0.646±0.005 |
| `toa_only_diag` | 70.70±0.31% | 6.129±0.040 | 0.650±0.005 | 70.05±0.21% | 5.959±0.009 | 0.641±0.003 |

结论：

- 严格按 validation accuracy，A5d 内部最佳是 `toa_only_diag`。
- `main_5feat` 的 Test MAE/Test Macro-F1 更好，但不能用 test 反选模型。
- 两组均未超过 A2 baseline 的 Test Acc。A5 应表述为解释性辅助消融，而非稳定准确率提升方法。

### A6：Alpha 角度有序性损失与标签策略

A6 是 Alpha 版 B3，固定最干净的 Alpha-ToT baseline，仅比较 loss / label strategy。

当前状态：A6a 正在运行。

固定：

- `Alpha_100`
- ToT only
- `resnet18_no_maxpool`
- A2 best training config
- handcrafted disabled
- fusion none
- split: `Alpha_100_ToT_seed42_0.8_0.1_0.1.json`

A6a matrix：

```text
CE onehot baseline: reuse A2-best, not rerun
Gaussian soft CE: sigma = 5.0, 7.5, 10.0
CE+ExpectedMAE: lambda = 0.02, 0.05, 0.10
CE+EMD: lambda = 0.02, 0.05, 0.10
```

决策：

- 不做 pure EMD，因为可能使输出分布过宽并损害 exact classification。
- 暂不做 hybrid regression head，因为 `ce_expected_mae` 已能表达连续角度误差辅助监督。
- A6b 等 A6a 结果后，只对 validation-selected best 和可选 second-best 做三 seed。
- A6c 仅在 A6b 证明 best loss 有价值后，将其迁移到 `A4c-2 dual_stream_gmu_aux`。

## 五、Proton_C_7 主线

### B1：Proton_C_7 训练超参数搜索

固定结构继承 Alpha A1 的结构结论：

```text
model             = resnet18_no_maxpool
conv1_kernel_size = 2
conv1_stride      = 1
conv1_padding     = 0
```

#### B1-1：learning rate × batch size

搜索：

```text
learning_rate = [1e-4, 3e-4, 1e-3]
batch_size    = [64, 128, 256]
```

结论：validation-selected best 为 `learning_rate=3e-4`、`batch_size=128`。

20 epoch 旧结果和 from20 continuation 到 25 epoch 的结果均未改变该选择。from20 结果只是 epoch-budget rescue，不等价于从头 `T_max=25` 的原生训练。

#### B1-2：weight decay

固定 `learning_rate=3e-4`、`batch_size=128`，搜索：

```text
weight_decay = [0, 1e-5, 1e-4]
```

结论：`weight_decay=1e-4` 最优。

#### B1-best：patience=8 three-seed baseline

正式配置：

```text
configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml
```

正式训练超参：

```text
learning_rate = 3e-4
batch_size    = 128
weight_decay  = 1e-4
dropout       = 0.1
scheduler     = cosine
eta_min       = 1e-7
epochs        = 25
early_stopping_patience = 8
mixed_precision = true
```

旧 `patience=5` 版本只作为 early stopping 过激诊断，不作为最终 baseline。

正式结果：

| 指标 | Mean ± Std |
| --- | ---: |
| Val Acc | 92.94±1.81% |
| Test Acc | 93.26±1.64% |
| Test MAE | 0.640±0.161 |
| Test P90 | 0.0±0.0 |
| Test Macro-F1 | 0.952±0.011 |

### B2：Proton_C_7 手工特征低成本验证

B2 不再做主干或结构迁移验证，固定 B1-best。由于 Proton_C_7 只有 ToT，B2 只允许 ToT-only handcrafted features。

#### B2a：concat

结果：

- `geometry_lowcorr` 相对 B1-best seed42 只有极小提升：Test Acc 94.09% -> 94.26%。
- `geometry + ToT_density` 明显变差：Test Acc 91.63%。

#### B2b：gated

结果：

- gated 能抑制 `ToT_density` 在 concat 下的负面影响。
- gated 没有证明手工特征显著提升 Proton_C_7。

决策：B2c three-seed 不优先推进。B2 论文口径是：Proton_C_7 的 ToT 图像信息已被 CNN 较充分利用，手工标量主要与 CNN 表征冗余；gated 的价值更多是稳定化，而非显著增益。

### B3：Proton_C_7 有序角度损失

B3 固定 B1-best 主模型，不启用 handcrafted features，只比较 loss / label strategy。

#### B3a：seed42 screening

比较：

- Gaussian soft CE
- `CE+ExpectedMAE`
- `CE+EMD`

结论：`CE+ExpectedMAE lambda=0.05` 是 validation-selected best；`CE+EMD lambda=0.05` 非常接近，作为 ordered-loss optional 对照。Gaussian soft label 不继续推进。

#### B3b：three-seed verification

结果：

| 设置 | Val Acc | Val MAE | Val F1 | Test Acc | Test MAE | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1-best CE onehot | 92.94±1.81% | 0.676±0.184 | 0.949±0.012 | 93.26±1.64% | 0.640±0.161 | 0.952±0.011 |
| B3b-main CE+ExpectedMAE lambda=0.05 | 94.20±0.05% | 0.550±0.002 | 0.958±0.000 | 94.38±0.08% | 0.532±0.010 | 0.960±0.001 |
| B3b-optional CE+EMD lambda=0.05 | 94.20±0.16% | 0.545±0.015 | 0.957±0.001 | 94.35±0.09% | 0.532±0.011 | 0.959±0.001 |

结论：

- `CE+ExpectedMAE lambda=0.05` 是 Proton_C_7 当前推荐损失函数。
- `CE+EMD lambda=0.05` 是强 ordered-loss 对照，Val MAE 和 far-error diagnostics 略优，但 primary validation metrics 不替代 B3b-main。
- B3b 是强正结果，说明角度有序性损失比继续扩展 B2 手工特征更适合 Proton_C_7 的剩余相邻大角度错误。

### B4：Proton_C_7 最终模型确认

当前状态：待定。

建议定位：汇总 B1-best、B2 诊断、B3b best loss，形成 Proton_C_7 最终推荐设置。

## 六、数据分析链路

数据分析链路独立于训练系统，主要服务论文数据描述与近垂直角度分辨极限讨论。

| 编号 | 阶段 | 当前状态 | 说明 |
| --- | --- | --- | --- |
| D1 | Dataset analysis | 已实现 | 分析 `Alpha_100` 与全量 `Proton_C` 的样本数量、事件特征、代表样本和分布图。 |
| D2 | Resolution limit analysis | 已实现 | 分析全量 `Proton_C` 近垂直角度 80-90 deg 的 ToT 可分性。 |
| D3 | Combined analysis report | 已实现 | 合并 D1/D2 报告。 |

注意：数据分析脚本默认使用全量 `Proton_C`，不要改为训练子集 `Proton_C_7`。

## 七、文档整理记录

2026-04-30 进行实验文档交叉核对与结构化整理。核心决策如下：

- `agent/EXPERIMENT_LOG.md` 重新整理为当前权威实验日志，按 A/B/D 系列记录阶段目的、完成状态、关键结论与待办事项。
- `agent/RESEARCH_HANDOFF_5_5_PRO.md` 重新整理为交给 5.5 Pro 的研究交接文档，避免继续引用 A4/A5 早期状态。
- `configs/README.md` 重新定位为配置与命令索引，不再承担完整实验日志角色。
- `agent/README.md`、`agent/CODE_CONTEXT.md`、`agent/NEW_SYSTEM_GUIDE.md`、`agent/SERVER_TRAINING.md` 因旧版存在编码损坏与过时状态，已分别归档为 `.old.md` 并重写当前版。
- `agent/A4B_IMPLEMENTATION_PLAN.md` 与 `agent/A4B_SELECTOR_FUSION_PLAN.md` 保留为历史规划文档，页首已标注当前权威状态应以本日志和 `RESEARCH_HANDOFF_5_5_PRO.md` 为准。
- `agent/FILE_MAP.md` 已更新当前文档与归档文档的角色边界。

后续原则：如果某个文档需要大改，先归档为 `.old.md`，再重写当前版；若仅作小修，则需先完整阅读原文后再修改。

## 八、当前待办

1. 等待 A6a 结果。
2. 根据 A6a validation 结果撰写 A6b three-seed 配置；CE baseline 复用 A2-best，不重跑。
3. 评估是否需要适配 A6c：将 A6b best loss 迁移到 `A4c-2 dual_stream_gmu_aux`。
4. 等另一个窗口整理完整结果表后，补充本文档中仍缺失或需精确替换的数据表。
5. 后续论文写作中，避免使用 test 指标反向选择模型，尤其是 A4c GMU 与 A5d main_5feat 的表述。
