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

## 实验编号与阶段总览

本节作为后续实验命名的总索引。原则是：一级编号回答一个论文问题，子编号只表示该问题下的方法阶段；已生成的历史配置和输出目录不强制改名，避免破坏可追溯性，但论文、日志和新配置应按本节编号口径解释。

编号规则：

- `A*`：`Alpha_100` 主线训练实验。
- `B*`：`Proton_C_7` 训练主线实验。
- `D*` 或 `Analysis*`：论文数据分析、分辨极限分析，不混入训练实验编号。
- 带 `_seed42` 的配置表示快速单 seed 配置；不带 `_seed42` 的正式对比配置默认优先三 seed 或完整 grid。
- 历史 `A4b-2.5` 保留为已经发生的互补性初步诊断编号；后续不再新增半号编号。

Alpha 主线阶段：

| 编号 | 阶段目的 | 当前状态 | 关键说明 |
| --- | --- | --- | --- |
| `A1` | ResNet18 结构适配 | 已完成 | 确定 `resnet18_no_maxpool`、`conv1_kernel_size=2`、`conv1_stride=1` 是 Alpha-ToT 的最佳结构方向；A1 中 `dropout=0.3` 表现好，但最终 dropout 交给 A2 训练超参搜索决定。 |
| `A2` | Alpha-ToT 训练超参数搜索 | 已完成 | 固定 A1 结构后搜索 `learning_rate`、`weight_decay`、`batch_size`、`eta_min`、`dropout`，得到 A2 best base。 |
| `A2-best` | A2 最佳配置三 seed 认证 | 已完成 | 作为后续 ToT baseline 和 A4b seed-control 的基准池。 |
| `A3` | 主干模型对比 | 已完成 | 比较 ShallowCNN、ShallowResNet、ResNet18、DenseNet121、EfficientNet-B0、ConvNeXt-Tiny、ViT-Tiny；支持 `resnet18_no_maxpool` 作为当前主干。 |
| `A4` | 模态基础对比 | 已完成 | 比较 ToT、ToA、raw/log1p ToT+ToA；结论是 ToT 单模态最好，raw ToA 直接加入会降低效果。 |
| `A4b` | ToA-assisted decision-level selective fusion | 已完成主体 | 研究 early fusion 失败后，`relative_minmax/no mask` candidate 是否可作为选择性辅助 expert。 |
| `A4c` | End-to-end full bimodal fusion | 第一批与 A4c-4 已完成 | `A4c-1/2/3` 结果接近 A4b-5 的 Test Acc，并明显提升 Macro-F1；`A4c-4 freeze=true` 可作 warm-start gate 对照，但不是最佳融合方案。 |
| `A5` | 物理/手工特征融合 | 方案已定，暂未实现 | 聚焦 ToT image + selected ToT/ToA scalar physical features；先做 A5a 随机森林/置换重要性筛选，再做少量 CNN 融合验证，避免全特征大网格。 |
| `A6` | 损失函数与标签策略 | 待定 | 建议比较 CE one-hot、Gaussian soft label、ordinal/EMD-style loss、regression/hybrid 等。 |
| `A7` | 最终模型确认 | 待定 | 汇总最优结构、训练配置、融合策略、loss/feature 设置，做最终三 seed 或更多 seed 认证。 |

A4b 子阶段命名：

| 编号 | 阶段目的 | 当前状态 | 关键结论 |
| --- | --- | --- | --- |
| `A4b-1` | ToA 表达方式对比，仍属于 early fusion | 已完成 | relative ToA 明显好于 raw/log1p ToA，但所有 early fusion 变体仍未超过 ToT baseline。 |
| `A4b-2` | fixed late logit fusion | 已完成 | validation 选择 `alpha_toa=0`，说明全局固定 ToA 权重不可靠。 |
| `A4b-2.5` | 预测互补性初步诊断 | 已完成 | 发现 ToT 与 `relative_minmax/no mask` candidate 存在 oracle 互补性。 |
| `A4b-3a` | ToT-vs-ToT seed oracle control | 已完成 | 普通 ToT 多 seed oracle gain 较小，不能解释 ToT/candidate 的高互补性。 |
| `A4b-3b` | ToT vs `relative_minmax/no mask` validation/test oracle check | 已完成 | 证明互补性在 validation/test 均稳定存在，尤其 30 deg 类别。 |
| `A4b-4` | hard selector 系列 | 已完成 | rule selector 小幅有效，learned selector 不够稳定。 |
| `A4b-4d` | selector switch diagnostics | 已完成 | 主要瓶颈不是切换不足，而是 beneficial/harmful switch 难以区分，switch precision 不够。 |
| `A4b-4e` | rule selector 三 seed 验证 | 已完成 | validation-selected rule 三 seed 稳定小幅提升，但 rule 形式不稳定且远低于 oracle。 |
| `A4b-5` | sample-wise gated late fusion | 已完成 | 当前 A4b 最强结果；frozen-expert gate 比 hard rule 更好。 |
| `A4b-6` | residual gated fusion | 已完成 | 优于 ToT baseline，但正式 validation-selected 结果不如 A4b-5；作为 residual 对照。 |

A4c 计划命名：

| 编号 | 名称 | 阶段目的 | 当前安排 |
| --- | --- | --- | --- |
| `A4c-1` | `dual_stream_concat_aux` | 完整双流 feature fusion baseline | 已完成三 seed；Test Acc/MAE 在 A4c 第一阶段最好。 |
| `A4c-2` | `dual_stream_gmu_aux` | feature-level sample-wise gated fusion | 已完成三 seed；Macro-F1 最好，gate 诊断支持 ToA 作为辅助模态。 |
| `A4c-3` | `toa_conditioned_film` | ToA 作为辅助条件调制 ToT 特征 | 已完成三 seed；结果正向但弱于 concat/GMU。 |
| `A4c-4` | `warm_started_expert_gate` | A4b-5 frozen expert gate 的端到端 warm-start 版本 | 已完成三 seed × 两种 freeze 设置；`freeze_experts=true` 有效但不超过 A4b-5/A4c-1/2，`freeze_experts=false` 不稳定。 |
| `A4c-5` | `mmtm_lite` | 中间层跨模态通道重标定 | 选做；仅在 A4c-1/2/3/4 后仍有时间和算力时考虑。 |

A4c 固定决策：

- Dataset: `Alpha_100`
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Inputs: `ToT + relative_minmax ToA, no mask`
- Backbone base: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Training config: A2 best
- Loss/label: `cross_entropy` + `onehot`
- AMP: enabled
- Seeds: `42`, `43`, `44` for formal comparison；implementation smoke test 可以先跑 `seed42`
- Model selection: 只使用 validation；test 只用于最终报告。

A4c 自由度收敛决策：

- `ToA` 不再使用 raw/log1p，也不再扩展更多 mask/transform 网格；`relative_minmax, no mask` 来自 A4b 互补性结果。
- `A4c-1/2/3` 从头训练，不混入 checkpoint initialization 变量。
- `A4c-4` 单独作为 warm-start expert gate，加载 ToT baseline 和 candidate checkpoint；当前最小受控实现比较 `freeze_experts=true/false`，暂不加入更复杂的多阶段 schedule。
- `A4c-1/2` 使用 auxiliary heads，建议初始权重为 `tot_aux=0.3`、`toa_aux=0.1`；`A4c-3` 中 ToA 是条件调制，不强制 ToA auxiliary head。
- `A4c-2` 的 gate bias 初始化偏向 ToT；`A4c-3` 的 FiLM 最后一层 zero-init，使初始调制接近 identity。
- `A4c-5 mmtm_lite` 暂缓，避免完整双模态部分膨胀成新大网格；A4c-4 已完成后，是否推进 MMTM 需要结合剩余时间和论文主线再决定。

A5 计划命名：

| 编号 | 名称 | 阶段目的 | 当前安排 |
| --- | --- | --- | --- |
| `A5a` | handcrafted feature screening | 不训练 CNN，先用传统模型和置换重要性筛选物理标量特征 | 方案已定，暂未实现；只用 train/validation，test 不参与筛选。 |
| `A5b` | CNN + selected feature group ablation | 用少量 seed42 CNN run 检查不同精选特征组是否补充 ToT CNN | 方案已定，暂未实现；统一使用 concat，避免同时搜索融合方式。 |
| `A5c` | handcrafted fusion mode comparison | 只对 A5b 最好的 1 个特征组比较 handcrafted-only、concat、gated | 方案已定，暂未实现；不扩大特征集合。 |
| `A5d` | best handcrafted fusion 3-seed verification | 对 A5c 选出的最佳 1-2 个设置做 `training.seed=42/43/44` 认证 | 方案已定，暂未实现；报告 mean ± std。 |

A5 固定决策：

- A5 不再作为 A4b 子阶段，不命名为 A4b-8；A5 是独立的物理/手工标量特征融合阶段。
- A5 的核心问题是：低维物理标量特征是否能补充 CNN 图像特征，并提供更可解释的角度判别依据。
- 主线固定为 `image input = ToT only`，避免与 A4/A4c 的 ToT+ToA 图像融合问题混在一起。
- scalar feature source 可以读取 `ToT + ToA`，但图像分支只输入 ToT。
- 后续实现需要解耦 `dataset.modalities` 与 `handcrafted_features.source_modalities`；否则写入 `modalities: [ToT, ToA]` 会让模型看到 ToA 图像通道，导致 A5 与 A4c 混淆。
- 不参考 `timepix/analysis/` 数据分析链路里的既有特征实现；那条链路可能存在定义或实现偏差。A5 训练链路应重新实现、单独验证合理保留组的特征。
- A5 先不做 25 维大特征池，也不做逐特征开关网格；第一版候选特征压缩为 12 维。
- RandomForest / LogisticRegression / permutation importance 只作为 A5a 筛选与诊断，不作为 CNN 融合最终结论。
- 特征选择只允许使用 train/validation；test set 只用于最终报告。

A5 第一版候选特征池：

```text
Geometry:
  active_pixel_count
  bbox_long
  bbox_short
  bbox_fill_ratio
  pca_major_axis
  pca_minor_axis

ToT:
  total_ToT
  ToT_density

ToA:
  ToA_span
  ToA_p90_minus_p10

Axis interaction:
  ToA_major_axis_slope_abs
  ToA_major_axis_corr_abs
```

A5 特征暂缓/不作为第一版主特征：

```text
bbox_area
pca_eccentricity
ToA_std
ToA_iqr
mean_ToT_nonzero
std_ToT_nonzero
p90_ToT_nonzero
max_ToT_fraction
top10_ToT_fraction
ToT_ToA_corr_abs
ToT_axis_asymmetry_abs
ToA_axis_asymmetry_abs
raw bbox_width / bbox_height
PCA_angle
raw ToA sum / mean / max / min
```

A5 分组递进方案：

1. `A5a`：提取 12 维候选特征，训练 handcrafted-only `RandomForest` / `LogisticRegression`，使用 validation permutation importance 和 group permutation importance 筛选特征；test 不参与筛选。
2. `A5b`：基于 A5a 选出的 6-8 个特征，跑少量 seed42 CNN concat 消融：
   - `A5b-1`: CNN + selected Geometry
   - `A5b-2`: CNN + selected Geometry + ToT
   - `A5b-3`: CNN + selected ToA/Axis
   - `A5b-4`: CNN + selected all
3. `A5c`：只拿 A5b 最好的 1 个特征组比较融合方式：
   - handcrafted-only MLP
   - CNN + handcrafted concat
   - CNN + handcrafted gated
4. `A5d`：只对 A5c 最终选出的最佳 1-2 个设置做三 seed 认证。

预期 CNN run 数量控制：

- A5b: 约 4 个 seed42 run。
- A5c: 约 1-2 个 seed42 run。
- A5d: 约 3-6 个正式三 seed run，取决于最终保留 1 个还是 2 个设置。
- A5a 为轻量传统模型/特征诊断，不计入主要深度训练预算。

Proton/C 主线阶段：

| 编号 | 阶段目的 | 当前状态 | 关键说明 |
| --- | --- | --- | --- |
| `B1-1` | Proton_C_7 第一轮训练超参搜索：`learning_rate × batch_size` | 已完成 | 20 epoch 旧结果和 from20 中继 25 epoch 结果均选择 `learning_rate=3e-4`、`batch_size=128`。 |
| `B1-2` | Proton_C_7 第二轮训练超参搜索：`weight_decay` | 已完成 | 固定 B1-1 最佳 `learning_rate=3e-4`、`batch_size=128`，搜索 `weight_decay = [0, 1e-5, 1e-4]`；最终仍选择 `weight_decay=1e-4`。 |
| `B1-best` | Proton_C_7 最佳训练配置三 seed 认证 | patience=8 配置已撰写，待重跑 | 固定 B1-2 最佳组合 `learning_rate=3e-4`、`batch_size=128`、`weight_decay=1e-4`；原 `early_stopping_patience=5` 对 seed43/44 过激，改为 8 后重跑 `training.seed=42/43/44`。 |
| `B2` | Proton_C_7 主干/结构迁移验证 | 待定 | 如有需要，验证 Alpha 最佳结构是否仍适合 Proton_C_7。 |
| `B3` | Proton_C_7 损失/近角度分类策略 | 待定 | 可与 A6 对齐，特别关注角度有序性。 |
| `B4` | Proton_C_7 最终模型确认 | 待定 | 最终报告用。 |

数据分析链路：

| 编号 | 阶段目的 | 当前状态 | 关键说明 |
| --- | --- | --- | --- |
| `D1` | Dataset analysis | 已实现 | 分析 `Alpha_100` 与全量 `Proton_C`，输出论文数据集统计、图表和报告。 |
| `D2` | Resolution limit analysis | 已实现 | 分析全量 `Proton_C` 近垂直角度分辨极限，不使用训练专用 `Proton_C_7`。 |
| `D3` | Combined analysis report | 已实现 | 合并 `D1/D2` 报告。 |

当前推进顺序决策：

1. A4b 已形成完整闭环：A4/A4b-1/A4b-3/A4b-4/A4b-5/A4b-6。
2. A4b-5 继续作为当前多模态主结果；A4c 作为完整端到端双模态补充验证组。
3. A4c 第一批 `A4c-1/2/3` 已完成；第二批 `A4c-4 warm_started_expert_gate` 也已完成；`A4c-5 mmtm_lite` 仍为选做，是否推进需要结合时间和论文主线再决定。
4. B1 继续按 `Proton_C_7` 主线收尾，不和 A4c 混编号。
5. A5 方案已定但暂不实现；下一步如推进，应先做 A5a 特征筛选/诊断，再决定少量 A5b/A5c/A5d CNN 融合实验。

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

当前结果记录（用户汇报）：

B1-1 固定设置：

- Dataset: `Proton_C_7`
- Modality: `ToT`
- Model: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss/label: `cross_entropy` + `onehot`
- `dropout=0.1`
- `weight_decay=1e-4`
- Scheduler: `cosine`
- `eta_min=1e-7`
- 搜索项：`learning_rate × batch_size`

20 epoch 旧结果：

| learning_rate | batch_size | best epoch | early stop | Val Acc | Test Acc | Test MAE | Test F1 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1e-4 | 64 | 19 | 否 | 93.64% | 93.81% | 0.586 | 0.955 |
| 1e-4 | 128 | 20 | 否 | 93.34% | 93.64% | 0.601 | 0.954 |
| 1e-4 | 256 | 20 | 否 | 93.04% | 93.32% | 0.629 | 0.952 |
| 3e-4 | 64 | 6 | 是 | 89.79% | 89.91% | 0.963 | 0.930 |
| 3e-4 | 128 | 20 | 否 | **93.94%** | **93.97%** | **0.566** | **0.957** |
| 3e-4 | 256 | 3 | 是 | 89.90% | 89.64% | 0.972 | 0.925 |
| 1e-3 | 64 | 2 | 是 | 86.20% | 86.77% | 1.242 | 0.904 |
| 1e-3 | 128 | 5 | 是 | 80.94% | 81.04% | 1.634 | 0.840 |
| 1e-3 | 256 | 3 | 是 | 88.53% | 88.38% | 1.110 | 0.916 |

20 epoch 旧结果中，按 `Val Acc` 选择的最佳组合为：

```text
learning_rate = 3e-4
batch_size    = 128
```

该组合同时在 Test Acc、Test MAE 和 Test F1 上也是最优。

from20 中继到 25 epoch 的结果：

- 只有 4 组未早停 run 被继续训练，其余早停组合跳过。
- 这些结果是 `from20` continuation，不能视作从一开始使用 `CosineAnnealingLR(T_max=25)` 的原生 25 epoch 结果。

| learning_rate | batch_size | best epoch | early stop | Val Acc | Test Acc | Test MAE | Test F1 |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1e-4 | 64 | 23 | 否 | 93.78% | **94.11%** | **0.558** | **0.957** |
| 1e-4 | 128 | 22 | 否 | 93.39% | 93.79% | 0.586 | 0.955 |
| 1e-4 | 256 | 22 | 否 | 93.11% | 93.44% | 0.618 | 0.953 |
| 3e-4 | 128 | 20 | 是 | **93.94%** | 93.97% | 0.566 | 0.956 |

20 epoch 与 from20 中继 25 epoch 的变化：

| learning_rate | batch_size | Val Acc 变化 | Test Acc 变化 | MAE 变化 | F1 变化 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-4 | 64 | +0.14 percentage points | +0.30 percentage points | -0.028 | +0.0019 |
| 1e-4 | 128 | +0.05 percentage points | +0.15 percentage points | -0.015 | +0.0007 |
| 1e-4 | 256 | +0.07 percentage points | +0.12 percentage points | -0.011 | +0.0006 |
| 3e-4 | 128 | 0 | 0 | 0 | 0 |

最佳超参数决策：

- 如果严格按实验规范使用 validation 指标选择超参数，20 epoch 旧结果和 from20 中继 25 epoch 结果的最佳组合不变。
- B1-1 选择：

```text
learning_rate = 3e-4
batch_size    = 128
weight_decay  = 1e-4
dropout       = 0.1
scheduler     = cosine
eta_min       = 1e-7
```

- from20 中继结果中 `1e-4, batch_size=64` 的 Test Acc/MAE 略好，但 Val Acc 低于 `3e-4, batch_size=128`，且该结果来自补救式 continuation。因此不改变 B1-1 的 validation-selected 结论。
- B1-2 将固定 `learning_rate=3e-4`、`batch_size=128`，继续搜索 `weight_decay = [0, 1e-5, 1e-4]`。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_lr_batch_ep25 --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_runs.csv
```

### B1-2 weight decay 搜索

配置文件：

```text
configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml
```

实验目的：

- 在 B1-1 已确定 `learning_rate=3e-4`、`batch_size=128` 的基础上，只搜索 `weight_decay`。
- 得到后续 Proton_C_7 消融和对比实验的默认训练配置。

固定设置：

- Dataset: `Proton_C_7`
- Modality: `ToT`
- Task: classification
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss/label: `cross_entropy` + `onehot`
- Handcrafted: disabled
- Fusion: `none`
- `learning_rate=3e-4`
- `batch_size=128`
- `dropout=0.1`
- Scheduler: `cosine`
- `eta_min=1e-7`
- `epochs=25`
- `early_stopping_patience=5`
- AMP: enabled
- Split: `outputs/splits/Proton_C_7_ToT_seed42_0.8_0.1_0.1.json`

搜索项：

```yaml
grid:
  training.learning_rate:
    - 0.0003
  training.batch_size:
    - 128
  training.weight_decay:
    - 0.0
    - 0.00001
    - 0.0001
```

决策备注：

- B1-2 配置继承 B1-1，因此需要在 `grid` 中显式把 `training.learning_rate` 限定为 `[0.0003]`、`training.batch_size` 限定为 `[128]`，避免继承 B1-1 的 `learning_rate × batch_size` 网格。
- B1-2 是单 seed 超参搜索，选择标准仍为 `val_accuracy`；test 指标只作最终报告和辅助记录。
- B1-2 不做 `aggregate_seeds.py`，因为它不是多 seed 验证；运行后用 `scripts/summarize.py` 输出 3 组结果即可。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_weight_decay_ep25 --out outputs/b1_proton_c7_resnet18_tot_weight_decay_ep25_runs.csv
```

本地验证：

- `python scripts\run_grid.py --config configs\experiments\b1_proton_c7_resnet18_tot_weight_decay.yaml --dry-run`
- dry-run 通过，规划 3 个 run：
  - `weight_decay=0.0`
  - `weight_decay=1e-5`
  - `weight_decay=1e-4`

当前结果记录（用户汇报）：

B1-2 固定设置：

- Dataset: `Proton_C_7`
- Modality: `ToT`
- Model: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss/label: `cross_entropy` + `onehot`
- `learning_rate=3e-4`
- `batch_size=128`
- `dropout=0.1`
- `epochs=25`
- `early_stopping_patience=5`
- 搜索项：`weight_decay = [0, 1e-5, 1e-4]`
- 选择标准：`val_accuracy`

B1-2 `weight_decay` 搜索结果：

| `weight_decay` | Best epoch | Early stop | Val Acc | Test Acc | Test MAE | Test F1 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 13 | 是 | 93.57% | 93.90% | 0.578 | 0.9561 |
| 1e-5 | 7 | 是 | 92.56% | 92.42% | 0.715 | 0.9446 |
| 1e-4 | 17 | 是 | 93.84% | 93.97% | 0.574 | 0.9563 |

相对 B1-2 最佳组的差异：

| `weight_decay` | Δ Val Acc | Δ Test Acc | Δ Test MAE | Δ Test F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | -0.27 percentage points | -0.07 percentage points | +0.0039 | -0.0002 |
| 1e-5 | -1.28 percentage points | -1.55 percentage points | +0.1408 | -0.0117 |
| 1e-4 | 0 | 0 | 0 | 0 |

B1-1 与 B1-2 最佳对照：

| 来源 | LR | Batch | WD | Best epoch | Val Acc | Test Acc | Test MAE | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B1-1 old 20epoch | 3e-4 | 128 | 1e-4 | 20 | 93.94% | 93.97% | 0.566 | 0.9565 |
| B1-1 ep25_from20 | 3e-4 | 128 | 1e-4 | 20 | 93.94% | 93.97% | 0.566 | 0.9565 |
| B1-2 fresh ep25 | 3e-4 | 128 | 1e-4 | 17 | 93.84% | 93.97% | 0.574 | 0.9563 |

B1-2 最佳组 per-class test 指标：

| 角度 | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 10 deg | 1.0000 | 1.0000 | 1.0000 |
| 20 deg | 0.9987 | 0.9987 | 0.9987 |
| 30 deg | 0.9955 | 0.9964 | 0.9959 |
| 45 deg | 0.9342 | 0.9683 | 0.9509 |
| 50 deg | 0.9277 | 0.9290 | 0.9283 |
| 60 deg | 0.9085 | 0.8743 | 0.8911 |
| 70 deg | 0.9236 | 0.9349 | 0.9292 |

结论与决策：

- B1-2 没有改变 B1 的最佳训练超参数。
- `weight_decay=1e-4` 仍是当前 B1 最佳选择；`weight_decay=0` 很接近，但 validation、test 和 MAE 均略低；`weight_decay=1e-5` 明显更差，不建议继续使用。
- 当前 B1 最佳组合固定为：

```text
learning_rate = 3e-4
batch_size    = 128
weight_decay  = 1e-4
dropout       = 0.1
scheduler     = cosine
eta_min       = 1e-7
```

- 下一步进入 `B1-best`：固定上述组合，做 `training.seed=42/43/44` 三 seed 认证，并报告 mean ± std。

### B1-best 三 seed 认证

配置文件：

```text
configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml
```

历史诊断配置：

```text
configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml
```

实验目的：

- 固定 B1-1/B1-2 得到的 `Proton_C_7` 默认训练超参数，验证该组合在不同训练随机种子下的稳定性。
- 该实验不是新的超参搜索；只改变 `training.seed`，`split.seed` 与 split manifest 保持为 42，确保不同 seed 使用同一数据划分。

固定设置：

- Dataset: `Proton_C_7`
- Modality: `ToT`
- Model: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss/label: `cross_entropy` + `onehot`
- `learning_rate=3e-4`
- `batch_size=128`
- `weight_decay=1e-4`
- `dropout=0.1`
- `scheduler=cosine`
- `eta_min=1e-7`
- `epochs=25`
- `early_stopping_patience=8`
- AMP: enabled
- Split: `outputs/splits/Proton_C_7_ToT_seed42_0.8_0.1_0.1.json`

网格：

```yaml
grid:
  training.seed:
    - 42
    - 43
    - 44
```

决策备注：

- B1-best 配置不继承 B1-2 配置文件，因为 B1-2 自身包含 `training.weight_decay` 搜索 grid；直接继承会因深度合并而把旧搜索项带入 B1-best。
- 因此 B1-best 使用独立 YAML 显式写出固定配置，只保留 `training.seed` 一个 grid 维度。
- 2026-04-30 追加决策：原 `early_stopping_patience=5` 的 B1-best 运行表现不稳定。seed42 证明模型后期可以从低谷恢复到 93%+，但 seed43/44 在 epoch 10 左右被截断，可能尚未等到后期恢复。因此将正式 B1-best 重跑配置改为 `early_stopping_patience=8`。
- 为避免覆盖或混淆已运行的 patience=5 结果，新增独立配置、独立 `experiment_group` 和独立汇总文件：`b1_proton_c7_resnet18_tot_best_patience8_3seed`。原 patience=5 结果仅作为“早停过激”的诊断记录，不作为 B1-best 最终三 seed 认证。
- 运行后使用 `scripts/summarize.py` 输出逐 run 汇总，再使用 `scripts/aggregate_seeds.py` 计算 mean ± std。

服务器 `tmux` 持久化运行：

```bash
cd ~/Timepix
tmux new -s b1_best_p8
```

进入 `tmux` 后一次性运行完整链路：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_best_patience8_3seed --out outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv --out outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_mean_std.csv
```

`tmux` 使用：

```bash
# 断开但保持运行
Ctrl+b 然后按 d

# 重新进入
tmux attach -t b1_best_p8
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

## A4b selector 结果后的计划更新

记录日期：2026-04-29。

背景：
- A4b-4a `rule selector`、A4b-4b `train-logit selector` 和 A4b-4c `validation-CV selector` 已经有结果。
- A4b-4a 和 A4b-4b 在 test set 上相对 ToT 有小幅真实提升，但 A4b-4c 未超过 ToT。
- `oracle` 结果仍明显高于所有实际 selector，因此当前瓶颈不再是证明互补性存在，而是解释并提高 switch 的可靠性。

当前解释：
- A4b-4a 是当前最干净的正结果，因为规则由 validation set 选择，test set 只做最终报告。
- A4b-4b 作为探索性对照，因为它使用 train split 上的 expert 输出训练 selector，而冻结 expert 在 train split 上可能过度自信。
- A4b-4c 是更严格的 learned-selector 结果，目前属于负面或中性结果。
- 因此 A4b 不应直接跳到复杂的 end-to-end fusion。下一步先做 switch diagnostics，再做低成本的 soft-gate 和 residual 变体。

更新后的 A4b 编号：
- A4b-4d：对 A4b-4a 选出的规则 `entropy_adv_0p03` 做 switch diagnostics。不训练新模型，报告 switch precision、switch recall、harmful/neutral switch rate、per-class switch 行为和 score distribution。
- A4b-4e：对 A4b-4a 的小幅正结果做可选三 seed 确认。只补跑关键 candidate `relative_minmax/no mask` 的 seed43 和 seed44，ToT baseline 复用已有 `a2_best_3seed`。
- A4b-5：基于 A4b-4a 的 entropy advantage 信号做 `entropy soft gate`，阈值和 slope 只在 validation set 上选择。
- A4b-6：做 constrained residual interpolation，使用 validation-selected `beta` grid 控制 candidate 对 ToT logits 的修正幅度。
- A4b-7：做紧凑的 ToA-only relative controls。
- A4b-8：做 ToT image plus ToA scalar physical features。
- A4c：end-to-end full bimodal fusion 已从 A4b 中分离；不再新增 A4b-9。A4c 第一批为 `dual_stream_concat_aux`、`dual_stream_gmu_aux`、`toa_conditioned_film`，`warm_started_expert_gate` 放入 A4c 第二批。

暂缓内容：
- A4b 内部不再新增 GMU、FiLM、MMTM、ordinary feature concat 或更大的 mask/transform grid；其中 GMU/FiLM/feature concat 已迁移到 A4c，MMTM 作为 A4c 选做项。

### A4b-4d switch diagnostics 实现

实现脚本：

```text
scripts/analyze_selector_switches.py
```

目的：
- 复现 A4b-4a 在 validation set 上选出的固定规则 `entropy_adv_0p03`。
- 解释 A4b-4a 的小幅收益是否受限于 switch precision 偏低、switch recall 偏低、harmful switch、遗漏 30 deg beneficial samples，或 selector score distribution 高度重叠。
- 这是一个不训练新模型的诊断实验，不重新选择 rule 或 threshold。

当前 selector 实现中的规则定义：

```text
entropy_adv_0p03 switches to candidate when:
  primary prediction and candidate prediction disagree
  candidate_entropy <= primary_entropy - 0.03
```

服务器命令：

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

输出文件：
- summary CSV：总体 switch precision/recall、harmful/neutral switch rate、最终指标和 oracle 指标。
- by-class CSV：按真实角度类别统计同类诊断指标，重点关注 30 deg。
- samples CSV：逐样本记录 predictions、errors、selected flag、oracle-beneficial flag 和 switch outcome。
- distribution CSV：记录 selected-beneficial、selected-harmful、selected-neutral、missed-beneficial 和 no-benefit 分组的 score distribution。

本地验证：
- `python scripts\analyze_selector_switches.py --help`
- `python -m py_compile scripts\analyze_selector_switches.py`
- 使用小型 synthetic helper 数据完成 switch precision/recall 计算 smoke test。

当前结果记录（用户汇报）：

A4b-4d 解释的是 seed42 的规则选择器 `entropy_adv_0p03` 为什么离 oracle 仍然很远。

测试集整体：

| 指标 | 数值 |
| --- | ---: |
| ToT baseline acc | 70.48% |
| selector acc | 70.97% |
| oracle acc | 81.51% |
| selector gain | +0.50 percentage points |
| selector switch rate | 14.51% |
| oracle switch rate | 12.43% |
| beneficial switches | 70 |
| harmful switches | 69 |
| neutral switches | 7 |
| missed beneficial | 55 |
| switch precision | 47.95% |
| switch recall | 56.00% |

按类别统计：

| class | ToT acc | selector acc | oracle acc | switch precision | missed beneficial |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 deg | 82.05% | 78.06% | 87.18% | 34.04% | 7 |
| 30 deg | 29.66% | 37.24% | 55.17% | 59.26% | 24 |
| 45 deg | 75.42% | 79.38% | 86.72% | 67.44% | 17 |
| 60 deg | 71.15% | 67.31% | 81.41% | 31.03% | 7 |

结论：

- A4b-4d 的核心解释是：`entropy_adv_0p03` 不是切换太少，而是切换质量不够高。
- 测试集切换 146 个样本，其中 beneficial 70、harmful 69、neutral 7，几乎一半有益、一半有害，因此整体只能获得 +0.50 percentage points 的小幅收益。
- selector 主要帮助 30 deg 和 45 deg，但明显伤害 15 deg 和 60 deg。
- 30 deg 虽然改善最大，但仍漏掉 24 个 oracle-beneficial 样本，因此远低于 oracle。
- beneficial switch 和 harmful switch 的 entropy/confidence 分布高度重叠，说明 candidate 更自信不等于 candidate 更正确；这正是 rule selector 难以接近 oracle 的主要瓶颈。

## B1-1 epoch-20 中继恢复方案

记录日期：2026-04-29。

问题：
- B1-1 误用了旧的 20 epoch 配置运行。
- 当前计划中的 B1-1 训练预算已经调整为 25 epochs，`early_stopping_patience=5`。
- 从头重跑完整 9 组 grid 成本较高。

决策：
- 如果旧 run 中存在 `last_checkpoint.pth`，可以从该 checkpoint 继续训练，并将 `training.epochs` 改为 25。
- 这种方式可作为 epoch budget 补救，但不等同于从头训练的 25 epoch 结果，因为 cosine scheduler 在恢复前已经按照 20 epoch 的 schedule 运行过。结果必须标记为 `from20` continuation，不能伪装成原生 `T_max=25` 训练。
- 已经在 epoch 20 之前触发 early stopping 的 run 通常应跳过，因为即使把 `max_epochs` 增加到 25，已经因 patience 停止的 run 也不会继续训练。
- 保留旧 20 epoch group 不动，将恢复后的 run 复制到新 group，避免旧结果和补救结果混在一起。

实现支持：

```text
scripts/extend_runs.py
```

推荐服务器 dry-run：

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

推荐服务器正式执行：

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

汇总命令：

```bash
python scripts/summarize.py \
  --group b1_proton_c7_resnet18_tot_lr_batch_ep25_from20 \
  --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_from20_runs.csv
```

中继结果结论：

- 4 组未早停 run 被继续到 25 epoch，其余早停组合跳过。
- continuation 改善了 `1e-4` 系列的 test 指标，但按 validation 选择，最佳组合仍为 `learning_rate=3e-4`、`batch_size=128`。
- 因此 B1-1 结论不受中继影响；B1-2 固定 `3e-4 + batch_size 128` 后搜索 `weight_decay`。

本地验证：
- `python scripts\extend_runs.py --help`
- `python -m py_compile scripts\extend_runs.py`
- 使用本地 B1 outputs 运行 `D:\Program\Anaconda\envs\timepix-local\python.exe scripts\extend_runs.py ... --dry-run`。

## A4b-4e 三 seed selector 确认

记录日期：2026-04-29。

目的：
- A4b-4a 在 seed42 上得到小幅正结果，但 test accuracy 提升只有约 +0.50%。
- 如果要把它作为正式方法而不只是诊断结果，需要做三 seed 确认。
- 主要成本在 candidate expert；ToT 的 seed42/43/44 已经存在于 `a2_best_3seed`。

决策：
- 不重训 seed42 candidate，直接复用已有 `a4b_toa_transform_seed42` 中的 `relative_minmax/no mask` run。
- 只补训关键 candidate `ToT + relative_minmax ToA, no mask` 的 seed43 和 seed44。
- 随后对 seed42/43/44 统一评估 oracle complementarity 和 A4b-4a rule selector。

新增文件：

```text
configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml
scripts/aggregate_selector_fusion.py
```

candidate 训练：

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

三 seed oracle 确认：

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

三 seed rule-selector 确认：

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

selector 聚合命令：

```bash
python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_4e_rule_selector_seed42_summary.csv \
    outputs/a4b_4e_rule_selector_seed43_summary.csv \
    outputs/a4b_4e_rule_selector_seed44_summary.csv \
  --out outputs/a4b_4e_rule_selector_mean_std.csv
```

解释口径：
- 如果 validation-selected rule selector 相对 `primary_only` 提高了 mean test accuracy、MAE 和 macro-F1，A4b-4a 可以作为小幅但较稳定的 selector baseline 汇报。
- 如果 mean improvement 消失或方差较高，A4b-4a 仍只作为 seed42 诊断结果；更强的结论仍是 oracle-level complementarity，而不是可靠可部署的 fusion gain。

当前三 seed 结果记录（用户汇报）：

A4b-4e 说明 validation-selected rule selector 在三 seed 上都有稳定小幅收益，但仍不是强结论，也没有接近 oracle。

三 seed 汇总：

| method | Test Acc | Test MAE | Macro-F1 | selection rate |
| --- | ---: | ---: | ---: | ---: |
| ToT baseline | 70.44% ± 0.15 | 5.949 ± 0.068 | 0.636 ± 0.009 | 0 |
| candidate only | 68.75% ± 1.52 | 6.546 ± 0.313 | 0.626 ± 0.016 | 100% |
| validation-selected rule | 71.44% ± 0.57 | 5.835 ± 0.121 | 0.645 ± 0.013 | 9.51% ± 5.40 |
| oracle | 79.75% ± 1.96 | 4.061 ± 0.438 | 0.748 ± 0.036 | 9.94% ± 2.44 |

逐 seed 结果：

| seed | selected rule | ToT Acc | selector Acc | gain |
| ---: | --- | ---: | ---: | ---: |
| 42 | `entropy_adv_0p03` | 70.48% | 70.97% | +0.50 percentage points |
| 43 | `entropy_adv_0p0` | 70.58% | 72.07% | +1.49 percentage points |
| 44 | `conf_adv_0p15` | 70.28% | 71.27% | +0.99 percentage points |

解释口径更新：

- validation-selected rule selector 在三 seed 上均优于 ToT baseline，平均 Test Acc 提升约 +0.99 percentage points，同时 MAE 和 Macro-F1 也略有改善。
- 三个 seed 选中的具体 rule 不一致，说明简单规则形式本身不够稳定。
- candidate-only 明显不如 ToT，说明不能直接用 relative ToA early-fusion 模型替代 ToT。
- oracle 仍达到 79.75%，比 validation-selected rule selector 高约 8.31 percentage points，说明互补信息存在，但当前 rule selector 只能利用其中一小部分。
- 如果固定使用 seed42 的原始 A4b-4a 规则 `entropy_adv_0p03`，三 seed Test Acc 约为 71.74%，平均比 ToT 高约 +1.29 percentage points。但这只作为额外诊断，不应优先于 validation-selected rule，因为正式流程应尊重每个 seed 的 validation selection。

## A4b-5 sample-wise gated late fusion

记录日期：2026-04-29。

决策：
- A4b-5 和 A4b-6 不再作为“是否继续”的前置判断，而是作为正式的 selective-fusion 对比系列。
- soft/constrained variants 定位为诊断性 ablation，learned variants 在同一个脚本中一起比较。
- 先实现 A4b-5，因为它是 A4b-4 的直接扩展：用 sample-wise gate 替代 hard selection 或 global alpha。

实现脚本：

```text
scripts/evaluate_gated_late_fusion.py
```

固定约束：
- Dataset/split：`Alpha_100`，paired split 为 `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`。
- Primary expert：来自 `a2_best_3seed` 的 ToT baseline。
- Candidate expert：`ToT + relative_minmax ToA, no mask`。
- Primary/candidate ResNet experts 全部冻结，只训练或校准 gate。
- Test set 不用于选择 gate type、threshold、slope、fit mode 或 regularization。

已实现的 A4b-5 变体：
- A4b-5a: entropy soft gate, probability fusion.
- A4b-5b: learned scalar gate, probability fusion.
- A4b-5c: learned scalar gate, logit fusion.
- A4b-5d: class-aware probability gate.
- A4b-5e: conservative scalar probability gate, initialized toward ToT and penalized for high mean gate.

Gate features：
- ToT logits、candidate logits、logit differences。
- ToT probabilities、candidate probabilities、probability differences。
- 每个 expert 的 top1 confidence、top1-top2 margin 和 entropy。
- Disagreement flag 和 predicted angle difference。
- ToT-predicts-30 flag 和 candidate-predicts-30 flag。

Gate fitting：
- `train`：探索性/偏乐观参考，因为 expert 在 train split 上的输出可能过度自信。
- `val-cv`：更严格的设置，用 validation cross-fitting 得到 validation 指标，并用完整 validation fit 生成 test 报告。
- A4b-5a 使用 validation-grid selection 选择 entropy threshold 和 sigmoid slope。

seed42 命令：

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

A4b-4e candidate seeds 可用后的三 seed 命令：

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

输出内容：
- Summary CSV 包含 ToT baseline、candidate-only、A4b-4a rule、所有 A4b-5 variants 和 oracle。
- Per-class CSV 包含 baselines、rule、selected A4b-5 variant 和 oracle。
- Summary rows 包含 validation/test Acc、MAE、P90、macro-F1、mean gate、high-gate rate、true-30 mean gate、beneficial high-gate count 和 harmful high-gate count。

本地验证：
- `python scripts\evaluate_gated_late_fusion.py --help`
- `python -m py_compile scripts\evaluate_gated_late_fusion.py`
- 使用 `D:\Program\Anaconda\envs\timepix-local\python.exe` 完成合成 logits 的 smoke test。

当前三 seed 结果记录（用户汇报）：

原始结果文件：

```text
outputs/a4b_5_gated_late_fusion_mean_std.csv
outputs/a4b_5_gated_late_fusion_seed42_summary.csv
outputs/a4b_5_gated_late_fusion_seed42_by_class.csv
outputs/a4b_5_gated_late_fusion_seed43_summary.csv
outputs/a4b_5_gated_late_fusion_seed43_by_class.csv
outputs/a4b_5_gated_late_fusion_seed44_summary.csv
outputs/a4b_5_gated_late_fusion_seed44_by_class.csv
```

A4b-5 三 seed 主汇总：

| 方法 | Val Acc | Test Acc | Test MAE | Test F1 |
| --- | ---: | ---: | ---: | ---: |
| ToT primary | 69.03±0.46% | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| Candidate only | 68.03±1.08% | 68.75±1.52% | 6.546±0.313 | 0.626±0.016 |
| A4b-5 validation-selected gate | 71.40±0.59% | **72.17±1.72%** | **5.661±0.320** | **0.662±0.027** |
| Oracle | 77.42±2.16% | 79.75±1.96% | 4.061±0.438 | 0.748±0.036 |

相对变化：

- A4b-5 相比 ToT：Test Acc +1.72 percentage points，MAE -0.288 deg，Macro-F1 +0.026。
- A4b-5 相比 A4b-4e rule selector：Test Acc +0.73 percentage points，MAE -0.174 deg，Macro-F1 +0.017。

每个 seed 的 validation-selected gate：

| Seed | 选中策略 | Fit | Test Acc | vs ToT | Test MAE | Test F1 | Gate Mean | High Gate |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `a4b5d_class_aware_prob_val-cv` | val-cv | 71.07% | +0.60 | 5.815 | 0.655 | 0.499 | 49.01% |
| 43 | `a4b5d_class_aware_prob_train` | train | 74.16% | +3.58 | 5.293 | 0.692 | 0.861 | 87.67% |
| 44 | `a4b5d_class_aware_prob_val-cv` | val-cv | 71.27% | +0.99 | 5.875 | 0.639 | 0.622 | 63.02% |

关键备注：

- seed43 由 validation 选中了 `train` fit。根据前述决策，train-fit 结果作为探索性/偏乐观参考；论文正式表述应说明这一点。
- 更保守的论文口径可单独报告 `val-cv only` 结果，尤其是 `a4b5d_class_aware_prob_val-cv`。
- 按 test 诊断排序，部分 entropy soft gate 策略可达到约 72.23±0.80% Test Acc，但这些策略不是 validation-selected，不能作为正式模型选择依据。

主要 gate 变体三 seed 均值：

| Strategy | Fit | Test Acc | Test MAE | Test F1 |
| --- | --- | ---: | ---: | ---: |
| `a4b5d_class_aware_prob_val-cv` | val-cv | 71.80±1.09% | 5.711±0.234 | 0.657±0.019 |
| `a4b5c_learned_scalar_logit_val-cv` | val-cv | 71.70±0.77% | 5.815±0.179 | 0.658±0.014 |
| `a4b5e_conservative_scalar_prob_val-cv` | val-cv | 71.84±0.64% | 5.845±0.142 | 0.662±0.013 |
| `a4b5b_learned_scalar_prob_val-cv` | val-cv | 71.77±0.53% | 5.840±0.158 | 0.661±0.012 |
| `a4b5d_class_aware_prob_train` | train | 71.97±2.82% | 5.765±0.612 | 0.669±0.021 |
| fixed `rule_entropy_adv_0p03` | rule | 71.74±0.68% | 5.780±0.145 | 0.647±0.010 |

selected gate 按角度的三 seed test 均值：

| 角度 | ToT Acc | Candidate Acc | Selected Gate Acc | Oracle Acc |
| ---: | ---: | ---: | ---: | ---: |
| 15 deg | 83.19±1.14% | 76.92±7.87% | 81.48±2.75% | 88.70±1.74% |
| 30 deg | 25.29±3.92% | 32.18±13.80% | 30.80±8.62% | 41.61±12.83% |
| 45 deg | 77.02±1.56% | 79.94±5.98% | 79.47±4.34% | 87.76±2.30% |
| 60 deg | 68.80±2.06% | 58.97±4.00% | 73.08±3.85% | 76.92±4.00% |

阶段性判断：

- A4b-5 比 A4b-4 的 hard selector 更强，尤其 MAE 和 Macro-F1 改善更明显。
- 当前更保守、可写入论文主线的 A4b-5 代表结果建议使用 `class_aware_prob_val-cv`；包含 train-fit 的 validation-selected 总表可作为完整实验结果，但需标注 seed43 的偏乐观风险。

## A4b-6 residual gated fusion

记录日期：2026-04-29。

决策：
- A4b-6 实现为 frozen-expert post-processing 实验，而不是新的 ResNet 训练任务。
- ToT 仍是 primary expert；`relative_minmax/no mask` candidate 只负责将 logits 从 ToT 向 candidate 方向做部分修正。
- Validation set 用于选择 residual beta/gate variants；test set 只在选择完成后报告一次。

实现脚本：

```text
scripts/evaluate_residual_gated_fusion.py
```

核心公式：

```text
logits_final = logits_tot + residual_weight * (logits_candidate - logits_tot)
```

已实现的 A4b-6 变体：
- A4b-6a: scalar beta residual, validation beta grid.
- A4b-6b: per-class beta residual, validation class-wise beta grid.
- A4b-6c: learned sample gate plus scalar beta, train-fit and val-CV.
- A4b-6d: learned sample gate plus per-class beta, train-fit and val-CV.
- A4b-6e: conservative residual, including entropy-constrained residual grids and ToT-biased learned scalar residual.

seed42 命令：

```bash
cd /root/Timepix

python scripts/evaluate_residual_gated_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_6_residual_gated_fusion_seed42.json \
  --output-summary outputs/a4b_6_residual_gated_fusion_seed42_summary.csv \
  --output-by-class outputs/a4b_6_residual_gated_fusion_seed42_by_class.csv
```

A4b-4e candidates 可用后的三 seed 命令：

```bash
for seed in 42 43 44; do
  python scripts/evaluate_residual_gated_fusion.py \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_6_residual_gated_fusion_seed${seed}.json" \
    --output-summary "outputs/a4b_6_residual_gated_fusion_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_6_residual_gated_fusion_seed${seed}_by_class.csv"
done
```

residual 结果聚合命令：

```bash
python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_6_residual_gated_fusion_seed42_summary.csv \
    outputs/a4b_6_residual_gated_fusion_seed43_summary.csv \
    outputs/a4b_6_residual_gated_fusion_seed44_summary.csv \
  --out outputs/a4b_6_residual_gated_fusion_mean_std.csv
```

输出内容：
- Summary CSV 包含 ToT baseline、candidate-only、A4b-4a rule、所有 A4b-6 variants 和 oracle。
- Per-class CSV 包含 baselines、rule、selected A4b-6 variant 和 oracle。
- Summary rows 包含 validation/test Acc、MAE、P90、macro-F1、residual-weight mean/high-rate、true-30 residual weight、beneficial high-residual count 和 harmful high-residual count。

本地验证：
- `python scripts\evaluate_residual_gated_fusion.py --help`
- `python -m py_compile scripts\evaluate_residual_gated_fusion.py`
- 使用 `D:\Program\Anaconda\envs\timepix-local\python.exe` 完成合成 logits 的 smoke test。

当前三 seed 结果记录（用户汇报）：

原始结果文件：

```text
outputs/a4b_6_residual_gated_fusion_mean_std.csv
outputs/a4b_6_residual_gated_fusion_seed42_summary.csv
outputs/a4b_6_residual_gated_fusion_seed42_by_class.csv
outputs/a4b_6_residual_gated_fusion_seed43_summary.csv
outputs/a4b_6_residual_gated_fusion_seed43_by_class.csv
outputs/a4b_6_residual_gated_fusion_seed44_summary.csv
outputs/a4b_6_residual_gated_fusion_seed44_by_class.csv
```

A4b-6 三 seed 主汇总：

| 方法 | Val Acc | Test Acc | Test MAE | Test F1 | Test residual/high rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| ToT primary | 69.03±0.46% | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 | 0 |
| Candidate only | 68.03±1.08% | 68.75±1.52% | 6.546±0.313 | 0.626±0.016 | 1.000 |
| A4b-6 validation-selected residual | **71.76±0.72%** | **71.44±1.01%** | **5.800±0.077** | **0.656±0.010** | 0.377 |
| Oracle | 77.42±2.16% | 79.75±1.96% | 4.061±0.438 | 0.748±0.036 | 0.099 |

每个 seed 的 validation-selected residual：

| Seed | Selected strategy | 类型 | 参数 | Val Acc | Test Acc | Test MAE | Test F1 | Test residual mean |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | `a4b6b_per_class_beta_grid` | per-class beta | `0.5;1;0.2;0.3` | 72.23% | 70.28% | 5.890 | 0.661 | 0.431 |
| 43 | `a4b6b_per_class_beta_grid` | per-class beta | `1;1;0.05;0.5` | 72.13% | 72.07% | 5.755 | 0.663 | 0.574 |
| 44 | `a4b6e_entropy_residual_t0p1_k5_b0p5` | entropy residual | `t=0.1,k=5,beta=0.5` | 70.93% | 71.97% | 5.755 | 0.645 | 0.213 |

主要候选策略三 seed 聚合：

| Strategy | Test Acc | Test MAE | Test F1 | Residual mean |
| --- | ---: | ---: | ---: | ---: |
| `a4b6e_entropy_residual_tm0p05_k5_b0p5` | **72.60±0.77%** | **5.567** | 0.655 | 0.296 |
| `a4b6e_entropy_residual_tm0p05_k20_b0p3` | 72.43±0.15% | 5.577 | 0.654 | 0.187 |
| `a4b6a_scalar_beta_b0p3` | 72.40±0.47% | 5.572 | 0.650 | 0.300 |
| `a4b6e_entropy_residual_t0p1_k5_b0p5` | 72.33±0.55% | 5.621 | 0.652 | 0.231 |
| `a4b6a_scalar_beta_b0p5` | 72.30±1.47% | 5.676 | 0.652 | 0.500 |
| official selected residual | 71.44±1.01% | 5.800 | 0.656 | 0.377 |

注意：上表前几名是按 test 诊断排序，不能用于正式模型选择；正式结果仍应以 validation-selected residual 为准。

selected residual 按角度的 test 均值：

| 角度 | ToT Acc | Candidate Acc | Selected Residual Acc | Oracle Acc |
| ---: | ---: | ---: | ---: | ---: |
| 15 deg | 83.19±1.14% | 76.92±7.87% | 81.77±5.41% | 88.70±1.74% |
| 30 deg | 25.29±3.92% | 32.18±13.80% | 32.41±8.97% | 41.61±12.83% |
| 45 deg | 77.02±1.56% | 79.94±5.98% | 78.44±2.40% | 87.76±2.30% |
| 60 deg | 68.80±2.06% | 58.97±4.00% | 68.59±1.70% | 76.92±4.00% |

与前面方法的数值对比：

| 对比 | Test Acc 变化 | MAE 变化 | F1 变化 |
| --- | ---: | ---: | ---: |
| A4b-6 vs ToT | +0.99 percentage points | -0.149 deg | +0.020 |
| A4b-6 vs A4b-4e rule | +0.00 percentage points | -0.035 deg | +0.012 |
| A4b-6 vs A4b-5 gate | -0.73 percentage points | +0.139 deg | -0.006 |

阶段性判断：

- A4b-6 确实优于 ToT baseline，但正式 validation-selected 结果不如 A4b-5。
- A4b-6 的价值主要体现在作为 residual fusion 对照实验：相比 A4b-4e rule，MAE 和 Macro-F1 略好，但整体 Test Acc 没有进一步提高。
- 当前 A4b 系列最佳选择性融合结果仍是 A4b-5 gate；A4b-6 更适合作为“以 ToT 为主、candidate 只做受限修正”的对照方案。

## A4c end-to-end full bimodal fusion

记录日期：2026-04-29。

实验定位：

- A4c 是 A4b 之后的完整端到端双模态补充验证组。
- A4b-5/6 是 frozen-expert / frozen-logit 后处理融合，不重新训练 ResNet expert。
- A4c 重新训练 ToT/ToA 图像分支，让 feature-level gate 或条件调制模块直接访问中间图像特征。
- A4c 不替代 A4b-5 当前主结果；它用于回答完整端到端双模态模型能否进一步超过 A4b-5 frozen-expert gate。

固定设置：

- Dataset: `Alpha_100`
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Inputs: `ToT + relative_minmax ToA, no mask`
- Backbone base: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Training config: A2 best
- Loss/label: `cross_entropy` + `onehot`
- Handcrafted: disabled
- AMP: enabled
- Formal seeds: `42`, `43`, `44`
- Model selection: validation only; test only final report。

新增代码：

```text
timepix/models/dual_stream.py
```

新增模型：

| 编号 | `model.name` | 说明 |
| --- | --- | --- |
| `A4c-1` | `dual_stream_concat_aux` | ToT/relative-ToA 两个 ResNet18 encoder，高层 feature concat，并带 ToT/ToA auxiliary heads。 |
| `A4c-2` | `dual_stream_gmu_aux` | ToT/relative-ToA 两个 ResNet18 encoder，使用 GMU-style feature gate；gate bias 初始化偏向 ToT。 |
| `A4c-3` | `toa_conditioned_film` | ToT 为主分支，relative ToA encoder 生成 FiLM `gamma/beta`，在 ToT `layer3` 后调制；FiLM 最后一层 zero-init。 |

训练框架适配：

- `ModelOutput` 新增 `aux_logits` 和 `diagnostics` 字段。
- `train_one_epoch()` / `evaluate()` 支持 `model.aux_loss`，当模型返回 auxiliary logits 时计算：

```text
loss = CE(main) + weight_tot * CE(aux_tot) + weight_toa * CE(aux_toa)
```

- A4c 当前默认：

```yaml
model:
  aux_loss:
    enabled: true
    weight_tot: 0.3
    weight_toa: 0.1
```

- `metrics.json` 会记录 `validation_diagnostics` 和 `test_diagnostics`。
- `scripts/summarize.py` 与 `scripts/aggregate_seeds.py` 已加入 gate/FiLM 诊断均值列，例如 `test_gate_tot_mean`、`test_gate_toa_mean`、`test_film_gamma_abs_mean`、`test_film_beta_abs_mean`。

新增配置：

```text
configs/experiments/a4c_end_to_end_bimodal_fusion.yaml
configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml
```

seed42 快速验证命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml --continue-on-error
python scripts/summarize.py --group a4c_end_to_end_bimodal_fusion_seed42 --out outputs/a4c_end_to_end_bimodal_fusion_seed42_runs.csv
```

正式三 seed 命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion.yaml --continue-on-error
python scripts/summarize.py --group a4c_end_to_end_bimodal_fusion --out outputs/a4c_end_to_end_bimodal_fusion_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4c_end_to_end_bimodal_fusion_runs.csv --out outputs/a4c_end_to_end_bimodal_fusion_mean_std.csv
```

本地验证：

```powershell
python -m py_compile timepix\models\base.py timepix\models\dual_stream.py timepix\models\registry.py timepix\training\trainer.py timepix\training\runner.py timepix\config_validation.py scripts\summarize.py scripts\aggregate_seeds.py
python scripts\run_grid.py --config configs\experiments\a4c_end_to_end_bimodal_fusion_seed42.yaml --dry-run
& 'D:\Program\Anaconda\envs\timepix-local\python.exe' -  # 已用 synthetic batch 验证三种 A4c 模型 forward/train/evaluate 路径
```

当前实现验证结果：

- `python -m py_compile` 通过。
- `a4c_end_to_end_bimodal_fusion_seed42.yaml --dry-run` 规划 3 个 run。
- `a4c_end_to_end_bimodal_fusion.yaml --dry-run` 规划 9 个 run。
- 使用本地 `timepix-local` 环境和 synthetic `2 x 100 x 100` batch 验证：
  - `dual_stream_concat_aux` 可完成 train/evaluate；
  - `dual_stream_gmu_aux` 可输出 `gate_tot`、`gate_toa` diagnostics；
  - `toa_conditioned_film` 可输出 `film_gamma_abs`、`film_beta_abs` diagnostics。

### A4c 第一阶段结果记录

记录日期：2026-04-29。

用户汇报：A4c 第一阶段对应 `A4c-1/2/3`，不包括第二阶段 `A4c-4 warm_started_expert_gate`。当前完成 9 个 run，即 3 个模型 × 3 个 seed。

固定设置：

- Dataset: `Alpha_100`
- Inputs: `ToT + relative_minmax ToA, no mask`
- Model family: `resnet18_no_maxpool` style stem
- Training config: A2 best
- Loss/label: `cross_entropy` + `onehot`
- Seeds: `42/43/44`

A4c 三 seed 汇总：

| 模型 | Val Acc | Test Acc | Test MAE | Test P90 | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dual_stream_concat_aux` | 69.90±1.34% | **72.10±1.35%** | **5.631±0.333** | 15.000±0.000 | 0.686±0.017 |
| `dual_stream_gmu_aux` | 70.20±0.67% | 71.94±0.51% | 5.721±0.009 | 15.000±0.000 | **0.691±0.009** |
| `toa_conditioned_film` | **70.43±0.95%** | 71.60±1.21% | 5.775±0.236 | 15.000±0.000 | 0.678±0.021 |

逐 seed 数据：

| 模型 | Seed | Best/Stop | Val Acc | Test Acc | Test MAE | Test F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| concat | 42 | 4/12 | 69.33% | 72.27% | 5.502 | 0.685 |
| concat | 43 | 5/13 | 71.43% | **73.36%** | **5.383** | 0.703 |
| concat | 44 | 6/14 | 68.93% | 70.68% | 6.009 | 0.669 |
| GMU | 42 | 5/13 | 69.63% | 71.37% | 5.726 | 0.680 |
| GMU | 43 | 3/11 | 70.03% | 72.07% | 5.726 | 0.697 |
| GMU | 44 | 6/14 | 70.93% | 72.37% | 5.711 | 0.696 |
| FiLM | 42 | 8/16 | 69.53% | 70.28% | 5.979 | 0.663 |
| FiLM | 43 | 9/17 | 71.43% | 71.87% | 5.830 | 0.669 |
| FiLM | 44 | 5/13 | 70.33% | 72.66% | 5.517 | 0.703 |

与已有主结果对比：

| 方法 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| ToT primary / A2-best | 69.03±0.46% | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| Candidate ToT+relative-ToA | 68.03±1.08% | 68.75±1.52% | 6.546±0.313 | 0.626±0.016 |
| A4b-4e rule selector | 70.70±0.47% | 71.44±0.57% | 5.835±0.121 | 0.645±0.013 |
| A4b-5 gated late fusion | 71.40±0.59% | **72.17±1.72%** | 5.661±0.320 | 0.662±0.027 |
| A4b-6 residual fusion | **71.76±0.72%** | 71.44±1.01% | 5.800±0.077 | 0.656±0.010 |
| A4c concat | 69.90±1.34% | 72.10±1.35% | **5.631±0.333** | 0.686±0.017 |
| A4c GMU | 70.20±0.67% | 71.94±0.51% | 5.721±0.009 | **0.691±0.009** |
| A4c FiLM | 70.43±0.95% | 71.60±1.21% | 5.775±0.236 | 0.678±0.021 |

阶段性判断：

- A4c 第一阶段不是失败结果。它没有在 Test Acc 上明确超过 A4b-5，但已经达到同一水平：A4b-5 为 `72.17±1.72%`，A4c concat 为 `72.10±1.35%`，考虑标准差后基本不可区分。
- A4c 更突出的价值是 Macro-F1：三个 A4c 模型均明显高于 ToT primary 和 A4b-5，其中 `dual_stream_gmu_aux` 达到 `0.691±0.009`。
- 这说明端到端双模态模型的主要贡献可能不是刷新总体 accuracy，而是改善类别均衡性和错误结构。
- 按每类表现，ToT primary 对 `30 deg` 类别 recall 约 `25.3%`；A4c GMU 将 `30 deg` recall 提高到 `57.0±6.9%`，代价是 `15 deg` recall 从约 `83.2%` 降到 `70.7%`。
- GMU gate 诊断合理：test 上平均约 `77.6%` 偏向 ToT、`22.4%` 使用 ToA 分支，说明模型没有被 ToA 带偏，而是将 ToA 作为辅助模态使用。

论文口径：

- A4c 第一阶段可表述为：端到端双模态融合没有显著刷新最高 accuracy，但提升了 Macro-F1 和 `30 deg` 困难类别识别能力，说明 ToA 的价值更体现在类别互补和误差结构修正上。
- A4c 内部如果按 Test Acc/MAE 选代表模型，可选 `dual_stream_concat_aux`；如果按机制解释、Macro-F1 和 gate 诊断，优先讲 `dual_stream_gmu_aux`。
- 下一步曾计划运行 `A4c-4 warm_started_expert_gate`，因为它结合 A4b-5 的 frozen expert 稳定性与 A4c 的端到端训练思想；该实验现已完成，结果见下一节。

### A4c-4 warm-started expert gate 实现与结果

记录日期：2026-04-29。

阶段目的：

- A4c-4 是 A4c 第二批实验，不扩展为新的大网格。
- 它把 A4b-5 已证明有互补性的两个 expert 纳入训练框架：`ToT primary expert` 与 `ToT + relative_minmax ToA, no mask candidate expert`。
- 目标是检验：继承已有 expert 权重后，feature/logit gate 是否能比从头训练的完整双模态模型更稳定地利用 candidate。

关键决策：

- A4c-5 `mmtm_lite` 继续保持选做；A4c-4 已完成后，是否推进 MMTM 需要结合剩余时间和论文主线再决定。
- A4c-4 只比较两个受控变体：`freeze_experts=true` 与 `freeze_experts=false`。
- `freeze_experts=true`：冻结 primary/candidate，只训练 gate，定位为最保守的 warm-start gate。
- `freeze_experts=false`：加载 checkpoint 后允许 primary/candidate 与 gate 一起 fine-tune，检验端到端微调是否有效。
- gate 初始化偏向 primary：`init_bias_to_candidate=-2.0`，使初始 `gate_candidate≈0.12`，避免一开始破坏 ToT 主模态。
- 这里的 `gate_candidate` 表示 candidate expert 的融合权重，不是单独 ToA 通道权重；candidate expert 的输入是 `ToT + relative_minmax ToA, no mask`。
- checkpoint 不写死时间戳路径；runner 根据 `outputs/experiments/<group>/*/metadata.json` 自动按 `training.seed`、`modalities`、`model.name`、`toa_transform` 和 `add_hit_mask` 找到对应 `best_model.pth`。
- 服务器持久化方式统一使用 `tmux`，不再优先写 `nohup` 命令。

新增模型：

```text
model.name: warm_started_expert_gate
```

结构：

```text
Primary branch: ToT -> ResNet18 no-maxpool -> logits_primary, features_primary
Candidate branch: ToT + relative_minmax ToA -> ResNet18 no-maxpool -> logits_candidate, features_candidate
Gate input: features_primary, features_candidate, logits_primary, logits_candidate
g = sigmoid(MLP(...))
logits_final = (1 - g) * logits_primary + g * logits_candidate
```

新增/更新代码：

```text
timepix/models/dual_stream.py
timepix/models/registry.py
timepix/training/runner.py
timepix/config_validation.py
scripts/summarize.py
scripts/aggregate_seeds.py
```

新增配置：

```text
configs/experiments/a4c_warm_started_expert_gate.yaml
configs/experiments/a4c_warm_started_expert_gate_seed42.yaml
```

checkpoint 搜索来源：

```yaml
primary_search:
  group: a2_best_3seed
  model: resnet18_no_maxpool
  modalities: [ToT]

candidate_search:
  groups:
    - a4b_toa_transform_seed42
    - a4b_4e_relative_minmax_no_mask_seed43_44
  model: resnet18_no_maxpool
  modalities: [ToT, ToA]
  data:
    toa_transform: relative_minmax
    add_hit_mask: false
```

seed42 快速验证命令：

```bash
cd ~/Timepix
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate_seed42.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4c_warm_started_expert_gate_seed42 --out outputs/a4c_warm_started_expert_gate_seed42_runs.csv
```

正式三 seed 命令：

```bash
cd ~/Timepix
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4c_warm_started_expert_gate --out outputs/a4c_warm_started_expert_gate_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4c_warm_started_expert_gate_runs.csv --out outputs/a4c_warm_started_expert_gate_mean_std.csv
```

服务器 `tmux` 持久化命令：

```bash
cd ~/Timepix
tmux new -s a4c_warm_gate
```

进入 `tmux` 后运行：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4c_warm_started_expert_gate --out outputs/a4c_warm_started_expert_gate_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a4c_warm_started_expert_gate_runs.csv --out outputs/a4c_warm_started_expert_gate_mean_std.csv
```

`tmux` 使用：

```bash
# 断开但保持运行
Ctrl+b 然后按 d

# 重新进入
tmux attach -t a4c_warm_gate

# 查看会话
tmux ls
```

本地验证：

```powershell
python -m py_compile timepix\models\dual_stream.py timepix\models\registry.py timepix\training\runner.py timepix\config_validation.py scripts\summarize.py scripts\aggregate_seeds.py
python scripts\run_grid.py --config configs\experiments\a4c_warm_started_expert_gate_seed42.yaml --dry-run
python scripts\run_grid.py --config configs\experiments\a4c_warm_started_expert_gate.yaml --dry-run
```

验证结果：

- `a4c_warm_started_expert_gate_seed42.yaml --dry-run` 规划 2 个 run。
- `a4c_warm_started_expert_gate.yaml --dry-run` 规划 6 个 run。
- 使用本地 `timepix-local` 环境验证 seed42/seed43/seed44 checkpoint 自动搜索与加载通过。
- 使用 synthetic `2 x 100 x 100` batch 验证 forward/backward 通过。
- `freeze_experts=true` 时 primary/candidate 保持 `eval()` 且不产生梯度，只有 gate 可训练。

结果记录（用户汇报）：

- A4c-4 已完成 6 个 run，均有 `metrics.json`、`predictions.csv`、`confusion_matrix.csv`。
- 实验设置为 `warm_started_expert_gate`：加载 ToT primary expert 和 `ToT + relative_minmax ToA` candidate expert。
- 对比两种策略：
  - `freeze_experts=true`：冻结两个 expert，只训练 gate。
  - `freeze_experts=false`：加载 expert 后继续端到端微调 expert + gate。

A4c-4 单次结果：

| Freeze | Seed | Best/Stop | Val Acc | Test Acc | Test MAE | Test F1 | Candidate Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| true | 42 | 9/17 | 69.73% | 69.78% | 6.262 | 0.660 | 86.11% |
| true | 43 | 9/17 | 71.63% | 73.46% | 5.606 | 0.675 | 71.20% |
| true | 44 | 4/12 | 70.13% | 72.27% | 5.845 | 0.645 | 38.00% |
| false | 42 | 1/9 | 67.23% | 67.69% | 6.740 | 0.591 | 68.61% |
| false | 43 | 2/10 | 69.73% | 70.38% | 5.964 | 0.650 | 56.75% |
| false | 44 | 3/11 | 69.43% | 72.37% | 5.666 | 0.687 | 44.24% |

A4c-4 三 seed 汇总：

| 设置 | Val Acc | Test Acc | Test MAE | Test F1 | Candidate Gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `freeze_experts=true` | 70.50±1.00% | 71.84±1.88% | 5.905±0.332 | 0.660±0.015 | 65.11±24.63% |
| `freeze_experts=false` | 68.80±1.36% | 70.15±2.34% | 6.123±0.554 | 0.643±0.049 | 56.53±12.19% |

和主线结果对比：

| 方法 | Test Acc | Test MAE | Test F1 |
| --- | ---: | ---: | ---: |
| ToT primary / A2-best | 70.44±0.15% | 5.949±0.068 | 0.636±0.009 |
| Candidate expert | 68.75±1.52% | 6.546±0.313 | 0.626±0.016 |
| A4b-5 gated late fusion | 72.17±1.72% | 5.661±0.320 | 0.662±0.027 |
| A4c concat | 72.10±1.35% | 5.631±0.333 | 0.686±0.017 |
| A4c GMU | 71.94±0.51% | 5.721±0.009 | 0.691±0.009 |
| A4c-4 `freeze=true` | 71.84±1.88% | 5.905±0.332 | 0.660±0.015 |
| A4c-4 `freeze=false` | 70.15±2.34% | 6.123±0.554 | 0.643±0.049 |

阶段性判断：

- `freeze_experts=true` 比 ToT primary 有一定提升，但没有超过 A4b-5 gated late fusion，也没有超过 A4c 第一阶段的 concat/GMU。
- `freeze_experts=false` 明显不稳定，平均结果接近或低于 ToT primary，说明端到端微调已训练 expert 容易破坏原有决策边界。
- `freeze=true` 的 `candidate gate` 均值较高且方差很大，说明 gate 倾向使用 candidate expert，但不同 seed 的使用比例不稳定；它不能解释为“模型大量使用 ToA 通道”，因为 candidate expert 本身是 ToT+relative-ToA 模型。
- A4c-4 的论文定位应作为 warm-start expert gate 对照：它支持“冻结强 expert 后做受控融合比端到端微调更稳”，但当前不是 A4c/A4b 的最佳结果。
