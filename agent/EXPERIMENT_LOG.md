# Timepix 实验日志与决策索引

本文档是 Timepix 极角识别项目的实验决策主日志。它根据旧版日志 `agent/EXPERIMENT_LOG.old.md`、当前配置目录 `configs/experiments/`、训练脚本和已同步实验结论重新整理，目标是让后续论文写作和 5.5 Pro 交接不再依赖零散对话记录。

本文档记录实验目的、固定配置、实验矩阵、配置文件、运行命令、关键结果和阶段决策。详细数值表、逐 run CSV、混淆矩阵和图表以 `outputs/` 中的原始结果为准；本文档只保存论文分析所需的权威摘要和决策链路。

## 一、全局实验规范

### 1.1 数据集命名

| 名称 | 用途 | 当前状态 |
| --- | --- | --- |
| `Alpha_100` | Alpha 主线正式数据集，100 x 100 输入，历史 A1/A2/A3/A4 结果均基于该数据故事线 | Alpha 后续实验默认使用 |
| `Alpha_50` | 曾短暂用于重跑 A3，但效果和叙事不稳定 | 不作为后续正式主线 |
| `Proton_C` | 质子/C 全量数据，用于论文数据分析链路 | 不作为训练主数据集 |
| `Proton_C_7` | 质子/C 七分类训练数据，用于 B 系列训练实验；数据分析中从全量 `Proton_C` 过滤 `10,20,30,45,50,60,70` 得到 | Proton 后续训练默认使用 |

关键决策：

- Alpha 主线统一回到 `Alpha_100`。早期讨论过 `Alpha_50`，但为了保持 A1/A2 历史结果、A3/A4/A4b/A4c 叙事一致，后续正式训练全部使用 `Alpha_100`。
- `Proton_C_7` 只代表七分类训练子集；数据分析链路仍可使用 `Proton_C` 作为全量分析对象。
- 本地 Windows 数据路径和服务器 Linux 数据路径不同。文档中的训练命令默认是服务器 Linux 命令，不应直接照搬到 Windows PowerShell。

### 1.2 Split 决策

| Split 文件 | 适用实验 | 决策说明 |
| --- | --- | --- |
| `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json` | Alpha_100 + ToT 单模态 | 从历史 A1/A2 使用的 ToT split 恢复，保持历史结论可追溯 |
| `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json` | Alpha_100 + ToT/ToA paired 实验 | 由 ToT split 复制得到；因为 ToT 与 ToA 文件一一对应，归一化 sample key 相同 |
| `outputs/splits/Proton_C_7_ToT_seed42_0.8_0.1_0.1.json` | Proton_C_7 + ToT | B 系列固定复用 |

关键决策：

- 对比实验必须复用固定 split。除非明确记录新 split，否则不得让程序静默重新随机划分。
- `Alpha_100_ToT-ToA` split 的内容应与历史 ToT split 一致，只是文件名标注 paired 用途。
- test set 不用于选择模型、阈值、feature group、loss 或 fusion strategy。

### 1.3 模型选择和汇报规则

- 主选择指标默认是 `Val Acc`。
- 若 `Val Acc` 接近，则按 `Val MAE` 更低、`Val Macro-F1` 更高作为 tie-break。
- 对物理有序角度任务，必须同时汇报 `Accuracy`、`MAE`、`P90`、`Macro-F1`、per-class recall/F1 和 confusion matrix。
- 三 seed 实验报告 `mean ± std`，不得用单个最高 seed 作为正式结论。
- test 指标只用于最终泛化说明，不得反向决定实验配置。

### 1.4 训练命令和汇总规范

服务器推荐使用 `tmux` 保持训练不中断：

```bash
tmux new -s <session_name>
cd ~/Timepix
python scripts/run_grid.py --config configs/experiments/<config>.yaml --skip-existing --continue-on-error
```

标准汇总命令：

```bash
python scripts/summarize.py --group <experiment_group> --out outputs/<name>_runs.csv
python scripts/aggregate_seeds.py --summary outputs/<name>_runs.csv --out outputs/<name>_mean_std.csv
```

单 seed 筛选实验通常只需要 `summarize.py`。三 seed 或多 seed 实验必须同时运行 `aggregate_seeds.py`。

## 二、实验编号总览

| 编号 | 阶段目的 | 状态 | 当前结论 |
| --- | --- | --- | --- |
| A1 | Alpha ResNet18 结构适配 | 已完成 | `resnet18_no_maxpool`、`conv1=2/1/0`、`dropout=0.3` 在 A1 中表现最佳；原始 ResNet18 作为 baseline |
| A2 | Alpha-ToT 训练超参数搜索 | 已完成 | A2 best 为 `lr=4.3878e-05`、`wd=4.7324e-04`、`batch=32`、`eta_min=1.6433e-07`、`dropout=0.1` |
| A2-best | Alpha-ToT 三 seed 基线 | 已完成 | 后续 Alpha ToT baseline 复用 A2 best |
| A3 | 主干模型对比 | 已完成 | `resnet18_no_maxpool` 支持作为主要 CNN 主干；ViT-Tiny 明显不适合当前小样本稀疏矩阵 |
| A4 | ToT/ToA/ToT+ToA 模态基础对比 | 已完成 | ToT 单模态最好；raw ToT+ToA 下降；ToA 单模态弱 |
| A4b | ToA 选择性辅助融合 | 已完成 | frozen expert gated late fusion 与 residual fusion 可利用相对 ToA candidate 的互补性 |
| A4c | 端到端 ToT/ToA 双模态架构 | 已完成 | GMU_aux 是论文主推端到端多模态架构；concat_aux 和 FiLM 是重要对照 |
| A5 | 物理/手工标量特征融合 | 已完成 | 手工特征有解释性补充，但三 seed 下未稳定提高 Alpha test accuracy |
| A6 | Alpha 有序角度损失 | 已完成 | A6b 三 seed 证明 `CE + EMD λ=0.02` 不稳定且弱于 A2 CE baseline；Alpha-ToT 继续采用 A2 CE one-hot |
| A7 | Alpha 最终多模态组件确认 | 已完成 | `main_5feat` 未带来稳定 validation 收益；Alpha 最终端到端多模态主模型保持 `dual_stream_gmu_aux + CE one-hot + no handcrafted` |
| B1 | Proton_C_7 训练超参数搜索 | 已完成 | B1-best 使用 `lr=3e-4`、`batch=128`、`wd=1e-4`、`patience=8` |
| B2 | Proton_C_7 手工特征验证 | B2c 已配置 | B2c 只验证 `geometry_lowcorr` 的 concat/gated 三 seed，不再加入不稳定的 `ToT_density` |
| B3 | Proton_C_7 有序角度损失 | 已完成 | `CE + ExpectedMAE λ=0.05` 是 Proton_C_7 当前推荐 loss |
| B4 | Proton_C_7 最终模型确认 | 已形成建议 | 最终建议为 B1-best 结构与训练超参 + `CE + ExpectedMAE λ=0.05`；无需新增训练组 |

## 三、Alpha 主线

### A1：ResNet18 结构适配实验

状态：已完成。

实验目的：在 Alpha_100 的 ToT 单模态任务上，确定 ResNet18 在 Timepix 稀疏矩阵上的合适输入 stem 和 dropout 设置，为后续 A2/A3/A4 系列提供结构基线。

固定配置：

- Dataset: `Alpha_100`
- Modality: `ToT`
- Task: classification
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Fusion: none
- Seed: 42

实验矩阵：

| 组别 | 变量 |
| --- | --- |
| baseline | `resnet18_original`，保留第一层 maxpool |
| A1 grid | `resnet18_no_maxpool`，搜索 `conv1_kernel_size=[2,3,5]`、`conv1_stride=[1,2]`、`dropout=[0,0.1,0.3]` |

配置文件：

- `configs/experiments/a1_resnet18_original_baseline.yaml`
- `configs/experiments/a1_structure_adaptation.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a1_resnet18_original_baseline.yaml --skip-existing --continue-on-error
python scripts/run_grid.py --config configs/experiments/a1_structure_adaptation.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a1_structure_adaptation --out outputs/a1_structure_adaptation_runs.csv
```

关键结果与决策：

- A1 最佳结构使用 `resnet18_no_maxpool`。
- A1 最佳 stem 为 `conv1_kernel_size=2`、`conv1_stride=1`、`conv1_padding=0`。
- A1 中最佳 dropout 为 `0.3`，但 A2 训练超参数搜索后续将 dropout 重新固定为 A2 best 的 `0.1`。
- `resnet18_original` 单独作为 baseline，不参与后续网格搜索。

### AMP 对比实验

状态：已完成。

实验目的：验证 CUDA AMP mixed precision 是否能在不明显损失精度的情况下提升训练效率，并确定后续训练是否默认启用。

配置文件：

- `configs/experiments/compare_mixed_precision.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group compare_mixed_precision --out outputs/compare_mixed_precision_runs.csv
```

关键决策：

- AMP 对最终精度无明显负面影响，后续正式训练可以启用 `mixed_precision: true`。
- 曾出现 AMP 开启后训练变慢的怀疑，后续排查认为性能变化不应归因于超参数搜索逻辑。AMP 仍作为默认可用训练选项保留。

### A2：Alpha-ToT 训练超参数搜索

状态：已完成。

实验目的：在 A1 已确定的 ResNet18 no-maxpool 结构基础上，搜索 Alpha-ToT 后续消融和对比实验统一使用的训练超参数。

固定配置：

- Dataset: `Alpha_100`
- Modality: `ToT`
- Model: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`、`conv1_stride=1`、`conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Scheduler: `cosine`
- Handcrafted: disabled

搜索范围：

| 参数 | 范围 |
| --- | --- |
| `training.learning_rate` | log float `[3e-5, 3e-3]` |
| `training.weight_decay` | log float `[1e-6, 1e-3]` |
| `training.batch_size` | `[32,64,128,256]` |
| `training.eta_min` | log float `[1e-7,1e-5]` |
| `model.dropout` | `[0.0,0.1,0.2]` |

配置文件：

- `configs/experiments/alpha_tot_a2_best_base.yaml` 保存 A2 best base 的固定设置。

关键结果：

- 最佳 trial: trial 12
- Val Acc: `0.6953`
- Test Acc: `0.7048`
- Val MAE: `6.279°`
- Test MAE: `5.964°`
- Test Macro-F1: `0.6461`
- Best epoch: 24
- Best hyperparameters: `learning_rate=4.3878e-05`、`weight_decay=4.7324e-04`、`batch_size=32`、`eta_min=1.6433e-07`、`dropout=0.1`、`scheduler=cosine`

决策：

- `alpha_tot_a2_best_base.yaml` 是 Alpha 后续 A3/A4/A4b/A4c/A5/A6 的训练配置来源。
- A2 是搜索实验，不作为最终正式训练结论；最终泛化表现用 A2-best three-seed 认证。

### A2-best：Alpha-ToT 三 seed 基线认证

状态：已完成。

实验目的：用 A2 最佳训练超参数对 Alpha-ToT ResNet18 no-maxpool baseline 做 three-seed 认证，作为后续 A4b/A4c/A5/A6 的主要比较基线。

配置文件：

- `configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a2_best_3seed --out outputs/a2_best_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a2_best_3seed_runs.csv --out outputs/a2_best_3seed_mean_std.csv
```

关键结果：

- Test Acc: `70.44 ± 0.15%`
- Test MAE: `5.949 ± 0.068°`
- Test Macro-F1: `0.636 ± 0.009`

决策：

- 该结果是 Alpha ToT 单模态主要 baseline。
- 后续 A4b frozen expert、A4c、A5、A6 都以该结构和训练配置为主参照。

### A3：主干模型对比

状态：已完成。

实验目的：在 Alpha-ToT 单模态任务上比较不同 CNN/Transformer 主干对 Timepix 稀疏矩阵的适应性。

固定配置：

- Dataset: `Alpha_100`
- Modality: `ToT`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled
- Training config: A2 best
- Epochs: 25

模型矩阵：

| 编号 | 模型 |
| --- | --- |
| A3-1 | `shallow_cnn` |
| A3-2 | `shallow_resnet` |
| A3-3 | `resnet18_no_maxpool` |
| A3-4 | `densenet121` |
| A3-5 | `efficientnet_b0` |
| A3-6 | `convnext_tiny` |
| A3-7 | `vit_tiny` |

配置文件：

- `configs/experiments/a3_backbone_comparison_seed42.yaml`
- `configs/experiments/a3_backbone_comparison.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a3_backbone_comparison --out outputs/a3_backbone_comparison_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a3_backbone_comparison_runs.csv --out outputs/a3_backbone_comparison_mean_std.csv
```

关键结果与决策：

- 单 seed 结果支持 `resnet18_no_maxpool` 作为当前最佳主干，Test Acc `70.48%`，MAE `5.96°`，Macro-F1 `0.646`。
- `convnext_tiny`、`shallow_resnet`、`densenet121` 构成第二梯队。
- `vit_tiny` 明显不适合当前小样本稀疏矩阵，说明 CNN 局部归纳偏置仍有优势。
- 后续正式主干继续使用 `resnet18_no_maxpool`；A4c 中的 GMU/FiLM/concat 也以 ResNet18 no-maxpool branch 为基础。

### A4：ToT / ToA / ToT+ToA 模态基础对比

状态：已完成。

实验目的：验证 Alpha 数据集中 ToT、ToA、ToT+ToA 对极角识别的贡献。

固定配置：

- Dataset: `Alpha_100`
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Model: `resnet18_no_maxpool`
- Stem: `2/1/0`
- Training config: A2 best
- Loss: `cross_entropy`
- Label: `onehot`

实验组：

| 组别 | 输入 |
| --- | --- |
| A4-1 | ToT |
| A4-2 | ToA |
| A4-3 | ToT + ToA raw/log1p |

配置文件：

- `configs/experiments/a4_modality_comparison_seed42.yaml`
- `configs/experiments/a4_modality_comparison.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4_modality_comparison --out outputs/a4_modality_comparison_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4_modality_comparison_runs.csv --out outputs/a4_modality_comparison_mean_std.csv
```

关键结果：

- ToT baseline: Test Acc `70.48%`，MAE `5.96°`，Macro-F1 `0.646`
- ToT+ToA raw/log1p: Test Acc `65.90%`，MAE `6.92°`，Macro-F1 `0.553`
- ToA only: Test Acc `60.14%`，MAE `8.81°`，Macro-F1 `0.477`

决策：

- ToT 是 Alpha 当前强主模态。
- raw ToA 不能通过 early channel concat 无条件加入。
- A4b 转向相对 ToA 表达、oracle 互补性和选择性融合。

### A4b：ToA 选择性辅助融合

状态：已完成。

总体目的：在 A4 发现 raw ToT+ToA 下降后，验证 ToA 是否以相对时间表达、frozen expert、selector/gate/residual 等方式作为弱辅助信息使用。

固定原则：

- Dataset: `Alpha_100`
- Paired split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Primary expert: A2-best ToT ResNet18 no-maxpool
- Candidate expert: ToT + `relative_minmax` ToA, no mask
- Test set 只用于最终报告。

#### A4b-1：relative ToA early fusion

实验目的：先排除 raw ToA 表达方式错误的可能，比较不同 relative ToA transform 和 hit mask 设置。

候选：

- `relative_minmax`
- `relative_centered`
- `relative_rank`
- `add_hit_mask=false/true`

配置文件：

- `configs/experiments/a4b_toa_transform_seed42.yaml`
- `configs/experiments/a4b_toa_transform.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4b_toa_transform --out outputs/a4b_toa_transform_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4b_toa_transform_runs.csv --out outputs/a4b_toa_transform_mean_std.csv
```

关键结果：

- 相对 ToA 明显优于 raw/log1p ToA early fusion，但仍未超过 ToT 单模态。
- `relative_minmax, no mask` 自身 Test Acc 不高，但 30° 类别表现和后续 oracle 互补性突出。

决策：

- 不继续扩展更多 early fusion transform/mask 网格。
- `relative_minmax, no mask` 作为后续 A4b frozen candidate。

#### A4b-2：fixed late logit fusion

实验目的：验证 ToT expert 与 ToA/candidate expert 是否能通过全局固定 alpha 融合。

公式：

```text
logits_final = (1 - alpha) * logits_tot + alpha * logits_candidate
```

脚本：

- `scripts/evaluate_logit_fusion.py`

服务器命令：

```bash
python scripts/evaluate_logit_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform \
  --seed 42 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --alphas 0,0.05,0.10,0.20,0.30,0.50 \
  --output-csv outputs/a4b_2_late_logit_fusion.csv \
  --output-json outputs/a4b_2_late_logit_fusion.json
```

关键结果：

- Validation 选择 `alpha=0.00`，即完全不用 ToA/candidate。
- 部分 alpha 在 test 上有小幅提升，但因未被 validation 选择，不作为正式模型。

决策：

- 固定全局 late fusion 不符合 “ToA 只对部分样本有用” 的现象。
- 后续转向 oracle、selector 和 sample-wise gate。

#### A4b-2.5：prediction complementarity 诊断

实验目的：不训练新模型，利用已有 predictions/logits 判断 ToT baseline 与 candidate 是否存在错误互补性。

脚本：

- `scripts/analyze_prediction_complementarity.py`
- `scripts/evaluate_oracle_complementarity.py`

核心指标：

- ToT 正确、candidate 错误
- ToT 错误、candidate 正确或误差更小
- per-class overlap，尤其 30°
- oracle accuracy 和 oracle MAE

决策：

- Oracle 显示 ToT 与 `relative_minmax,no mask` candidate 互补性很高。
- Candidate 自身不如 ToT，但在部分 ToT 错误样本上给出更接近真实角度的判断。

#### A4b-3a/b：oracle 控制诊断

实验目的：排除 A4b-2.5 的互补性只是随机 seed 差异或 test 特例。

实验组：

| 编号 | 对比 | 目的 |
| --- | --- | --- |
| A4b-3a | ToT-vs-ToT seed control | 判断普通 seed 多样性 oracle gain |
| A4b-3b | ToT vs `relative_minmax,no mask` | 验证 val/test 上 candidate 互补性稳定 |

服务器命令：

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-seed-control \
  --tot-group a2_best_3seed \
  --splits val,test \
  --seeds 42 43 44 \
  --output-summary outputs/a4b_3a_tot_seed_control_summary.csv \
  --output-by-class outputs/a4b_3a_tot_seed_control_by_class.csv \
  --output-json outputs/a4b_3a_tot_seed_control.json

python scripts/evaluate_oracle_complementarity.py \
  --mode tot-vs-candidate \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --splits val,test \
  --seeds 42 \
  --output-summary outputs/a4b_3b_tot_vs_relative_summary.csv \
  --output-by-class outputs/a4b_3b_tot_vs_relative_by_class.csv \
  --output-json outputs/a4b_3b_tot_vs_relative.json
```

关键结果：

- ToT-vs-ToT seed control 的 oracle gain 约 `+2.5 pp`，30° gain 很小。
- ToT vs `relative_minmax,no mask` 的 oracle gain 在 val/test 上约 `+10~11 pp`，30° oracle gain 约 `+25 pp`。

决策：

- Candidate 互补性不能简单解释为 seed 随机性。
- 后续核心问题变为：能否学习 “何时相信 candidate”。

#### A4b-4：hard selector

实验目的：在 frozen ToT expert 与 frozen candidate expert 之间做硬选择，验证 oracle 互补性是否可由简单规则或轻量 selector 利用。

实验组：

| 编号 | 方法 | 定位 |
| --- | --- | --- |
| A4b-4a | validation-selected rule selector | 正式小规模正结果 |
| A4b-4b | train-fit selector | 探索性，可能受 train logits 过度自信影响 |
| A4b-4c | validation-CV selector | 更严格 learned selector，对照负结果 |

服务器命令：

```bash
python scripts/evaluate_selector_fusion.py \
  --selector-mode rule \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --seed 42 \
  --output-summary outputs/a4b_4a_rule_selector_summary.csv \
  --output-by-class outputs/a4b_4a_rule_selector_by_class.csv \
  --output-json outputs/a4b_4a_rule_selector.json

python scripts/evaluate_selector_fusion.py \
  --selector-mode trained --selector-fit train \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --seed 42 \
  --output-summary outputs/a4b_4b_train_selector_summary.csv \
  --output-by-class outputs/a4b_4b_train_selector_by_class.csv \
  --output-json outputs/a4b_4b_train_selector.json

python scripts/evaluate_selector_fusion.py \
  --selector-mode trained --selector-fit val-cv \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --seed 42 \
  --output-summary outputs/a4b_4c_valcv_selector_summary.csv \
  --output-by-class outputs/a4b_4c_valcv_selector_by_class.csv \
  --output-json outputs/a4b_4c_valcv_selector.json
```

关键结果：

- A4b-4a rule: Test Acc `70.97%`，相对 ToT `+0.50 pp`，MAE 和 Macro-F1 同时改善。
- A4b-4b train selector: Test Acc `71.17%`，但仅作为探索性结果。
- A4b-4c val-CV selector: 未超过 ToT，切换率过低。

决策：

- Rule selector 证明互补性可被部分利用。
- Learned selector 仍不稳定，不能作为主结论。

#### A4b-4d：switch diagnostics

实验目的：解释 seed42 rule selector 为什么离 oracle 仍然很远。

脚本：

- `scripts/analyze_selector_switches.py`

服务器命令：

```bash
python scripts/analyze_selector_switches.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --seed 42 \
  --rule entropy_adv_0p03 \
  --splits val,test \
  --output-summary outputs/a4b_4d_switch_summary.csv \
  --output-by-class outputs/a4b_4d_switch_by_class.csv \
  --output-samples outputs/a4b_4d_switch_samples.csv \
  --output-distribution outputs/a4b_4d_switch_distribution.csv \
  --output-json outputs/a4b_4d_switch.json
```

关键结果：

- Test switch rate `14.51%`，oracle switch rate `12.43%`。
- selector 切换 146 个样本，其中 beneficial 70、harmful 69、neutral 7。
- switch precision `47.95%`，recall `56.00%`。
- 主要帮助 30° 和 45°，伤害 15° 和 60°。

决策：

- 瓶颈不是切换太少，而是 beneficial/harmful switch 难以用 entropy/confidence 稳定区分。

#### A4b-4e：three-seed rule selector

实验目的：验证 A4b-4a 的 rule selector 是否在 42/43/44 三 seed 上稳定。

配置文件：

- `configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml --skip-existing --continue-on-error
python scripts/evaluate_selector_fusion.py \
  --selector-mode rule \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform,a4b_4e_relative_minmax_no_mask_seed43_44 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-summary outputs/a4b_4e_rule_selector_summary.csv \
  --output-by-class outputs/a4b_4e_rule_selector_by_class.csv \
  --output-json outputs/a4b_4e_rule_selector.json
python scripts/aggregate_selector_fusion.py --inputs outputs/a4b_4e_rule_selector_summary.csv --out outputs/a4b_4e_rule_selector_mean_std.csv
```

关键结果：

- Validation-selected rule 三 seed 均优于 ToT baseline。
- Mean Test Acc 从 `70.44%` 提升到 `71.44%`，MAE 从 `5.949°` 降到 `5.835°`。

决策：

- Rule selector 是稳定小幅正结果，但 rule 选择不稳定且远低于 oracle。
- 继续做 A4b-5 sample-wise gated late fusion 和 A4b-6 residual fusion。

#### A4b-5：sample-wise gated late fusion

实验目的：用样本级 soft gate 替代 hard selector 和 fixed alpha。

公式：

```text
p_final = (1 - g) * p_tot + g * p_candidate
```

变体：

| 编号 | 方法 |
| --- | --- |
| A4b-5a | entropy soft gate |
| A4b-5b | learned scalar gate, probability fusion |
| A4b-5c | learned scalar gate, logit fusion |
| A4b-5d | class-aware probability gate |
| A4b-5e | conservative scalar gate |

脚本：

- `scripts/evaluate_gated_late_fusion.py`

服务器命令：

```bash
python scripts/evaluate_gated_late_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform,a4b_4e_relative_minmax_no_mask_seed43_44 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --fit-modes train,val-cv \
  --output-summary outputs/a4b_5_gated_late_fusion_summary.csv \
  --output-by-class outputs/a4b_5_gated_late_fusion_by_class.csv \
  --output-json outputs/a4b_5_gated_late_fusion.json
python scripts/aggregate_selector_fusion.py --inputs outputs/a4b_5_gated_late_fusion_summary.csv --out outputs/a4b_5_gated_late_fusion_mean_std.csv
```

关键结果：

- Validation-selected gate three-seed Test Acc `72.17 ± 1.72%`，MAE `5.661 ± 0.320°`，Macro-F1 `0.662 ± 0.027`。
- 相比 ToT baseline，Test Acc `+1.72 pp`，MAE `-0.288°`，F1 `+0.026`。
- seed43 选中 train-fit 版本，存在偏乐观风险；论文中应同时说明 val-CV conservative 口径。

决策：

- A4b-5 是 frozen expert 选择性融合中最强的整体结果。
- 作为 performance-oriented expert-level fusion 重要结果保留。

#### A4b-6：residual gated fusion

实验目的：以 ToT 为主，让 candidate 只作为 residual correction。

公式：

```text
logits_final = logits_tot + residual_weight * (logits_candidate - logits_tot)
```

变体：

| 编号 | 方法 |
| --- | --- |
| A4b-6a | scalar beta residual |
| A4b-6b | per-class beta residual |
| A4b-6c | learned gate + scalar beta |
| A4b-6d | learned gate + per-class beta |
| A4b-6e | conservative / entropy residual |

脚本：

- `scripts/evaluate_residual_gated_fusion.py`

服务器命令：

```bash
python scripts/evaluate_residual_gated_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform,a4b_4e_relative_minmax_no_mask_seed43_44 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --fit-modes train,val-cv \
  --output-summary outputs/a4b_6_residual_gated_fusion_summary.csv \
  --output-by-class outputs/a4b_6_residual_gated_fusion_by_class.csv \
  --output-json outputs/a4b_6_residual_gated_fusion.json
python scripts/aggregate_selector_fusion.py --inputs outputs/a4b_6_residual_gated_fusion_summary.csv --out outputs/a4b_6_residual_gated_fusion_mean_std.csv
```

关键结果：

- Validation-selected residual Test Acc `71.44 ± 1.01%`，MAE `5.800 ± 0.077°`，Macro-F1 `0.656 ± 0.010`。
- 优于 ToT baseline，但不如 A4b-5 gated late fusion。
- A4b-6 的 validation accuracy 较强，可作为 expert-level residual fusion 对照。

决策：

- A4b-6 证明 candidate 作为 ToT 修正项有效，但不作为最终端到端架构。

#### A4b 命名收口

- A4b-7 ToA-only relative controls：讨论过，但未作为正式主线推进。
- A4b-8 ToA scalar physical features：拆分为独立 A5，不再作为 A4b 子阶段。
- A4b-9 end-to-end full bimodal fusion：拆分为 A4c。

### A4c：端到端 ToT/ToA 双模态架构

状态：已完成。

实验目的：在 A4b frozen expert 后处理取得正结果后，验证完整端到端双模态模型能否直接从 ToT/relative ToA 图像特征中学习选择性融合。

固定配置：

- Dataset: `Alpha_100`
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Input: ToT + `relative_minmax` ToA, no mask
- Backbone: ResNet18 no-maxpool branch
- Training config: A2 best
- Seeds: 42/43/44

#### A4c-1/2/3：第一阶段端到端双模态

实验矩阵：

| 编号 | 模型 | 目的 |
| --- | --- | --- |
| A4c-1 | `dual_stream_concat_aux` | 完整双流 feature concat baseline |
| A4c-2 | `dual_stream_gmu_aux` | feature-level sample-wise gate |
| A4c-3 | `toa_conditioned_film` | ToA-conditioned FiLM 调制 ToT |

配置文件：

- `configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml`
- `configs/experiments/a4c_end_to_end_bimodal_fusion.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4c_end_to_end_bimodal_fusion --out outputs/a4c_end_to_end_bimodal_fusion_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4c_end_to_end_bimodal_fusion_runs.csv --out outputs/a4c_end_to_end_bimodal_fusion_mean_std.csv
```

关键结果：

| 模型 | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: |
| `dual_stream_concat_aux` | `72.10 ± 1.35%` | `5.631 ± 0.333°` | `0.686 ± 0.017` |
| `dual_stream_gmu_aux` | `71.94 ± 0.51%` | `5.721 ± 0.009°` | `0.691 ± 0.009` |
| `toa_conditioned_film` | `71.60 ± 1.21%` | `5.775 ± 0.236°` | `0.678 ± 0.021` |

决策：

- A4c 不是单纯刷新 accuracy，而是显著提高 Macro-F1 和 30° 困难类别表现。
- `dual_stream_gmu_aux` 的 validation Macro-F1 与 FiLM 接近，Val MAE 更优且物理解释更一致，因此作为论文主推端到端多模态架构。
- `dual_stream_concat_aux` 是强 baseline，但缺少显式选择机制，不作为主推架构。

#### A4c-4：warm-started expert gate

实验目的：把 A4b-5 frozen expert gate 改造成可训练端到端 expert gate，比较冻结专家和解冻微调。

实验组：

| 组别 | 设置 |
| --- | --- |
| freeze=true | 加载 ToT primary expert 与 candidate expert，冻结 expert，只训练 gate |
| freeze=false | 加载 expert 后继续端到端微调 expert + gate |

注意：`gate_candidate` 是 candidate expert 权重，不是单独 ToA 通道权重。

配置文件：

- `configs/experiments/a4c_warm_started_expert_gate_seed42.yaml`
- `configs/experiments/a4c_warm_started_expert_gate.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4c_warm_started_expert_gate --out outputs/a4c_warm_started_expert_gate_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4c_warm_started_expert_gate_runs.csv --out outputs/a4c_warm_started_expert_gate_mean_std.csv
```

关键结果：

- `freeze=true`: Test Acc `71.84 ± 1.88%`，MAE `5.905 ± 0.332°`，Macro-F1 `0.660 ± 0.015`
- `freeze=false`: Test Acc `70.15 ± 2.34%`，MAE `6.123 ± 0.554°`，Macro-F1 `0.643 ± 0.049`

决策：

- freeze=true 可以保持较好 accuracy，但 Macro-F1 和 30° 改善不如 GMU。
- freeze=false 不稳定，说明 expert gate 端到端解冻会破坏已有 expert 边界。

#### A4c-5：MMTM-lite

状态：未推进。

决策：

- MMTM-lite 曾作为可选中间层跨模态通道重标定模型讨论。
- A4c-1/2/3/4 已经足够支撑端到端双模态结构结论，继续引入 MMTM 会扩大工程复杂度和论文解释成本。
- 当前多模态架构探索收束，不优先继续新增架构。

#### A4 final multimodal architecture decision

论文表述建议：

- Performance-oriented expert-level fusion: A4b-5/A4b-6 作为 frozen expert 后处理系统，用于说明选择性 decision-level fusion 可提升 accuracy/MAE。
- End-to-end multimodal architecture: A4c-2 `dual_stream_gmu_aux` 作为主推架构。

选择 GMU 的依据不能使用 test 反选。正式口径为：

- GMU 的 validation Macro-F1 与 FiLM 几乎相同，差异远小于标准差。
- GMU 的 validation MAE 更好，稳定性更好。
- GMU 的结构与 A4b 得出的物理结论一致：ToT 是主信息源，ToA 是弱辅助时间结构，应由 gate 选择性融合。

### A5：物理/手工标量特征融合

状态：已完成。

总体目的：验证低维物理标量特征是否能补充 ToT CNN 图像特征，并提供更可解释的角度判别依据。A5 独立于 A4，不再命名为 A4b-8。

固定原则：

- Dataset: `Alpha_100`
- Image input: ToT only
- Scalar feature source: ToT + ToA
- Split: `Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Model: `resnet18_no_maxpool`
- Training config: A2 best

#### A5a：handcrafted feature screening

实验目的：不训练 CNN，先用传统模型筛选低冗余手工特征，避免 25 维大特征池造成冗余和不可解释。

配置文件：

- `configs/experiments/a5a_alpha_handcrafted_screening.yaml`

服务器命令：

```bash
python scripts/screen_handcrafted_features.py \
  --config configs/experiments/a5a_alpha_handcrafted_screening.yaml \
  --out-dir outputs/a5a_alpha_handcrafted_screening \
  --name a5a_alpha_handcrafted_screening \
  --seed 42
```

关键实现决策：

- 特征实现不参考独立数据分析链路，避免把潜在分析脚本误差带入训练主线。
- `LogisticRegression` 使用 OVR 兼容多分类，修复 `liblinear` 不支持原生 multiclass 的问题。
- A5a 不使用 test 参与特征选择。

关键结果：

- RandomForest Val Acc `63.94%`，Logistic OVR Val Acc `58.94%`。
- Geometry 组最重要，其次是 ToT 能量/密度，ToA 标量和 axis interaction 有补充价值但非主导。
- 相关性风险明显，因此进入 CNN 融合的只保留低冗余代表特征。

#### A5b：CNN + low-redundancy concat pilot

实验目的：用 simple concat 验证低冗余手工特征能否补充 ToT CNN。

配置文件：

- `configs/experiments/a5b_alpha_handcrafted_group_ablation.yaml`
- `configs/experiments/a5b_alpha_handcrafted_group_ablation_TEMPLATE.yaml`

实验组：

| 编号 | 组名 | 特征 | Fusion |
| --- | --- | --- | --- |
| A5b-1 | `geometry_lowcorr` | `active_pixel_count`, `bbox_fill_ratio` | concat |
| A5b-2 | `geometry_plus_tot_lowcorr` | A5b-1 + `ToT_density` | concat |
| A5b-3 | `toa_lowcorr_diagnostic` | `ToA_span`, `ToA_major_axis_corr_abs` | concat |
| A5b-4 | `geometry_plus_tot_plus_toa_lowcorr` | A5b-2 + A5b-3 | concat |

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a5b_alpha_handcrafted_group_ablation.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a5b_alpha_handcrafted_group_ablation --out outputs/a5b_alpha_handcrafted_group_ablation_runs.csv
```

关键结果：

- A5b 没有证明 simple concat 能稳定提升 ToT CNN。
- `geometry_lowcorr` Test Acc 接近 baseline，MAE 略降。
- `ToA_lowcorr` 提高 30° recall，但总体 test accuracy 下降。

决策：

- simple concat 不适合直接融合这批低维标量。
- A5c 保持相同特征组，只把 fusion 改为 gated。

#### A5c：gated diagnostic

实验目的：验证 gated fusion 是否比 simple concat 更适合低维手工特征。

配置文件：

- `configs/experiments/a5c_alpha_handcrafted_gated_seed42.yaml`
- `configs/experiments/a5c_alpha_handcrafted_fusion_mode_TEMPLATE.yaml`
- `configs/experiments/a5c_alpha_handcrafted_only_TEMPLATE.yaml`

实验组：

| 编号 | 组名 | 特征 | Fusion |
| --- | --- | --- | --- |
| A5c-1 | `geometry_lowcorr` | `active_pixel_count`, `bbox_fill_ratio` | gated |
| A5c-2 | `geometry_plus_tot_lowcorr` | A5c-1 + `ToT_density` | gated |
| A5c-3 | `toa_lowcorr_diagnostic` | `ToA_span`, `ToA_major_axis_corr_abs` | gated |
| A5c-4 | `geometry_plus_tot_plus_toa_lowcorr` | all 5 features | gated |

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a5c_alpha_handcrafted_gated_seed42.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a5c_alpha_handcrafted_gated_seed42 --out outputs/a5c_alpha_handcrafted_gated_seed42_runs.csv
```

关键结果：

- Gated 在四个特征组上均优于 A5b concat 的对应组。
- A5c-4 seed42 Test Acc `71.07%`，MAE `5.651°`，Macro-F1 `0.652`，是 A5 seed42 最均衡设置。

决策：

- 进入 A5d 的主组为五维 `geometry+ToT+ToA gated`。
- 同时保留 `toa_only_diag` 三 seed 诊断，因为它是 A5c seed42 validation accuracy 最高组。

#### A5d：three-seed verification

实验目的：对 A5c 的主要 gated 手工特征组做 three-seed 认证。

配置文件：

- `configs/experiments/a5d_alpha_handcrafted_gated_3seed.yaml`
- `configs/experiments/a5d_alpha_handcrafted_best_3seed_TEMPLATE.yaml`

实验组：

| 组别 | 特征 | 定位 |
| --- | --- | --- |
| `main_5feat` | `active_pixel_count`, `bbox_fill_ratio`, `ToT_density`, `ToA_span`, `ToA_major_axis_corr_abs` | 物理上最完整的低冗余组 |
| `toa_only_diag` | `ToA_span`, `ToA_major_axis_corr_abs` | ToA 标量诊断组 |

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a5d_alpha_handcrafted_gated_3seed.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a5d_alpha_handcrafted_gated_3seed --out outputs/a5d_alpha_handcrafted_gated_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a5d_alpha_handcrafted_gated_3seed_runs.csv --out outputs/a5d_alpha_handcrafted_gated_3seed_mean_std.csv
```

关键结果：

| 设置 | Val Acc | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| A2 ToT baseline | `69.03 ± 0.46%` | `70.44 ± 0.15%` | `5.949 ± 0.068°` | `0.636 ± 0.009` |
| `main_5feat` | `70.23 ± 0.30%` | `70.18 ± 0.95%` | `5.875 ± 0.231°` | `0.646 ± 0.005` |
| `toa_only_diag` | `70.70 ± 0.31%` | `70.05 ± 0.21%` | `5.959 ± 0.009°` | `0.641 ± 0.003` |

决策：

- 若严格按 validation accuracy，A5d 内部最佳是 `toa_only_diag`。
- `main_5feat` 的 test MAE 和 test Macro-F1 更好，但不能以 test 反选为最终模型。
- A5 的论文口径：手工标量有物理解释性和 MAE/F1 辅助价值，但未稳定提升 Alpha test accuracy。
- A6 不继续扩展 A5 特征组合，转向角度有序性损失。

### A6：Alpha 有序角度损失与标签策略

状态：A6a/A6b 已完成；A6c 不推进。

实验目的：在固定结构和输入设置下比较角度有序性 loss/label strategy，检查 CE one-hot 是否过于粗糙，是否能降低 MAE、改善 Macro-F1 和 30° 困难类别。

#### A6a：Alpha-ToT loss screening

固定配置：

- Dataset: `Alpha_100`
- Input: ToT only
- Model: `resnet18_no_maxpool`
- Training config: A2 best
- Handcrafted: disabled
- Fusion: none
- Seed: 42

实验矩阵：

| 编号 | Loss / Label |
| --- | --- |
| A6a-0 | CE one-hot baseline，复用 A2-best seed42，不重跑 |
| A6a-1 | Gaussian soft label `sigma=5` |
| A6a-2 | Gaussian soft label `sigma=7.5` |
| A6a-3 | Gaussian soft label `sigma=10` |
| A6a-4 | CE + ExpectedMAE `lambda=0.02` |
| A6a-5 | CE + ExpectedMAE `lambda=0.05` |
| A6a-6 | CE + ExpectedMAE `lambda=0.10` |
| A6a-7 | CE + EMD `lambda=0.02` |
| A6a-8 | CE + EMD `lambda=0.05` |
| A6a-9 | CE + EMD `lambda=0.10` |

不做项：

- 不做 pure EMD，因为它可能让输出分布变宽，损害 exact classification accuracy。
- 不做 large sigma Gaussian 和 focal loss 大网格，避免任务从分类主线漂移。

配置文件：

- `configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a6a_alpha_tot_ordinal_loss_seed42 --out outputs/a6a_alpha_tot_ordinal_loss_seed42_runs.csv
```

关键结果：

| 方法 | Val Acc | Val MAE | Val P90 | Val Macro-F1 | Test Acc | Test MAE | Test P90 | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 CE onehot baseline | **69.53%** | 6.279 | 15.0 | 0.630 | 70.48% | 5.964 | 15.0 | 0.646 |
| Gaussian sigma=5 | 68.83% | 6.414 | 15.0 | 0.620 | 69.38% | 6.054 | 15.0 | 0.628 |
| Gaussian sigma=7.5 | 68.33% | 6.533 | 30.0 | 0.615 | 70.68% | 5.949 | 15.0 | 0.643 |
| Gaussian sigma=10 | 68.63% | 6.414 | 15.0 | 0.612 | 71.17% | **5.755** | 15.0 | 0.642 |
| CE+ExpectedMAE lambda=0.02 | 69.03% | 6.399 | 30.0 | 0.629 | 69.98% | 5.934 | 15.0 | **0.648** |
| CE+ExpectedMAE lambda=0.05 | 69.03% | 6.489 | 30.0 | 0.614 | 70.58% | 5.994 | 15.0 | 0.636 |
| CE+ExpectedMAE lambda=0.10 | 68.53% | 6.489 | 30.0 | 0.622 | 69.09% | 6.113 | 15.0 | 0.629 |
| CE+EMD lambda=0.02 | **69.53%** | **6.264** | 15.0 | **0.636** | 69.68% | 5.964 | 15.0 | 0.641 |
| CE+EMD lambda=0.05 | 69.13% | 6.444 | 30.0 | 0.605 | 70.38% | 5.964 | 15.0 | 0.621 |
| CE+EMD lambda=0.10 | 68.53% | 6.533 | 30.0 | 0.586 | 70.68% | 5.994 | 15.0 | 0.621 |

Validation 排名：

| 排名 | 方法 | Val Acc | Val MAE | Val Macro-F1 |
| ---: | --- | ---: | ---: | ---: |
| 1 | CE+EMD lambda=0.02 | 69.53% | **6.264** | **0.636** |
| baseline | A2 CE onehot | 69.53% | 6.279 | 0.630 |
| 2 | CE+ExpectedMAE lambda=0.02 | 69.03% | 6.399 | 0.629 |
| 3 | CE+ExpectedMAE lambda=0.05 | 69.03% | 6.489 | 0.614 |

类别表现摘要：

| Split | 方法 | 15 deg R/F1 | 30 deg R/F1 | 45 deg R/F1 | 60 deg R/F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| Val | A2 baseline | 0.777 / 0.732 | 0.243 / 0.332 | 0.790 / 0.766 | 0.716 / 0.689 |
| Val | CE+EMD lambda=0.02 | 0.757 / 0.729 | **0.278 / 0.372** | **0.801 / 0.763** | 0.703 / 0.679 |
| Val | CE+ExpectedMAE lambda=0.02 | 0.751 / 0.725 | 0.264 / 0.353 | 0.793 / 0.761 | 0.716 / 0.677 |
| Test | A2 baseline | 0.821 / 0.763 | 0.297 / 0.402 | 0.754 / 0.751 | 0.712 / 0.669 |
| Test | CE+EMD lambda=0.02 | 0.795 / 0.757 | 0.310 / 0.405 | 0.757 / 0.741 | 0.699 / 0.661 |
| Test | CE+ExpectedMAE lambda=0.02 | 0.778 / 0.753 | **0.331 / 0.423** | 0.774 / 0.748 | 0.699 / 0.667 |
| Test | Gaussian sigma=10 | **0.863 / 0.778** | 0.241 / 0.348 | 0.746 / 0.753 | **0.731 / 0.689** |

A6a 阶段判断：

- A6a 没有得到像 Proton B3b 那样强的 loss 改进；Alpha-ToT 的有序损失收益明显更弱。
- 按 validation selection 规则，`CE+EMD lambda=0.02` 是 A6a 的主候选：Val Acc 与 A2 baseline 持平，Val MAE 与 Val Macro-F1 更好。
- 该收益属于 tie-break 级别，而不是 accuracy 层面的明确提升。它在 test 上不优于 A2 baseline，因此不能把它写成稳定泛化收益。
- `CE+ExpectedMAE lambda=0.02` 仅作为 A6a 结果解释中的诊断点：它的 test Macro-F1 与 30 deg test recall/F1 更好，但 validation 不支持它作为主选择，因此不进入 A6b。
- Gaussian soft label 不进入 A6b。`sigma=10` 的 test accuracy 最高，但 validation 表现不支持选择它，不能用 test 反选。

#### A6b：three-seed verification

状态：已完成。

计划：

- 复用 A2-best CE one-hot three-seed，不重跑 CE baseline。
- A6b-main 候选：`CE+EMD lambda=0.02`，因为它是 A6a validation-selected best。
- 不运行 `CE+ExpectedMAE lambda=0.02`。原因是它不是 validation-selected 主候选，且当前 A6b 需要收窄实验规模，只验证 A6a validation-selected best。
- 不进入 A6b：Gaussian soft label。原因是 validation 不支持，且较大 sigma 有软化分类边界风险。
- 选择标准：Val Acc 为主，Val MAE 和 Val Macro-F1 为 tie-break。

配置文件：

```text
configs/experiments/a6b_alpha_tot_ce_emd_0p02_3seed.yaml
```

A6b-main 服务器命令：

```bash
tmux new -s a6b
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a6b_alpha_tot_ce_emd_0p02_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a6b_alpha_tot_ce_emd_0p02_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a6b_alpha_tot_ce_emd_0p02_3seed --out outputs/a6b_alpha_tot_ce_emd_0p02_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a6b_alpha_tot_ce_emd_0p02_3seed_runs.csv --out outputs/a6b_alpha_tot_ce_emd_0p02_3seed_mean_std.csv
```

逐 seed 结果：

| Seed | Best epoch | Stopped epoch | Early stop | Val Acc | Val MAE | Val Macro-F1 | Test Acc | Test MAE | Test Macro-F1 |
| ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 23 | 25 | 否 | 69.53% | 6.264 | 0.636 | 69.68% | 5.964 | 0.641 |
| 43 | 3 | 11 | 是 | 67.23% | 7.088 | 0.570 | 68.79% | 6.531 | 0.584 |
| 44 | 18 | 25 | 否 | 68.23% | 6.503 | 0.621 | 70.38% | 5.934 | 0.644 |

A2 vs A6b 三 seed 汇总：

| 方法 | Val Acc | Val MAE | Val P90 | Val Macro-F1 | Test Acc | Test MAE | Test P90 | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 CE baseline | **69.03±0.46%** | **6.424±0.127** | 25.0±8.66 | **0.622±0.007** | **70.44±0.15%** | **5.949±0.068** | **15.0±0.0** | **0.636±0.009** |
| A6b CE+EMD lambda=0.02 | 68.33±1.15% | 6.618±0.424 | 25.0±8.66 | 0.609±0.034 | 69.62±0.80% | 6.143±0.336 | 20.0±8.66 | 0.623±0.034 |

逐类 F1 对比：

| 方法 | Split | 15 deg F1 | 30 deg F1 | 45 deg F1 | 60 deg F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| A2 CE baseline | Val | 0.727 | 0.323 | 0.761 | 0.676 |
| A6b CE+EMD | Val | 0.725 | 0.316 | 0.757 | 0.637 |
| A2 CE baseline | Test | 0.762 | 0.357 | 0.757 | 0.669 |
| A6b CE+EMD | Test | 0.752 | 0.336 | 0.757 | 0.647 |

A6b 阶段判断：

- A6a 中 `CE+EMD lambda=0.02` 的 seed42 validation tie-break 优势不可靠；A6b seed43 明显崩了一次，导致三 seed 均值和方差均弱于 A2 CE baseline。
- 按 validation 选择规则，A2 CE baseline 全面优于 A6b：Val Acc 高 0.70 pp，Val MAE 低 0.195 deg，Val Macro-F1 高 0.013。
- A6b 没有解决 Alpha-ToT 最关键的 30 deg 难分类问题；30 deg F1 反而低于 A2 baseline。主要负面影响体现在 60 deg 类别下降，45 deg 基本持平。
- A6 最终结论：Alpha-ToT 不采用 `CE+EMD lambda=0.02` 或其他 A6a 候选有序损失；后续 Alpha ToT baseline 继续使用 A2 CE one-hot。

#### A6c：迁移到 GMU 多模态架构

状态：不推进。

计划：

- A6b 已证明有序损失在 Alpha-ToT baseline 上不稳定且弱于 CE one-hot，因此不把该 loss 迁移到 A4c-2 `dual_stream_gmu_aux`。
- 决策理由：GMU 已经是 A4c 的端到端多模态主推架构；在 A6b 没有稳定收益的前提下，继续叠加有序损失会增加变量并削弱主线清晰度。

### A7：最终多模态架构的手工物理特征增益验证

状态：已完成。

实验目的：在已经确定的最终端到端多模态架构上，只验证 A5 选出的五维物理标量 `main_5feat` 是否还能提供额外补充价值。A7 不再扩展 loss、feature group 或新的多模态架构。

关键决策：

- A6b 已否定 Alpha 上的 `CE+EMD lambda=0.02`，因此 A7 loss 固定回 `CE one-hot`。
- 最终端到端多模态架构固定为 A4c-2 `dual_stream_gmu_aux`，输入为 `ToT + relative_minmax ToA, no mask`。
- `A7-0` 不重跑，直接复用 A4c 中 `dual_stream_gmu_aux` 的 three-seed 结果。
- `A7-1` 只跑一组：`GMU_aux + CE one-hot + main_5feat gated`，三 seed。
- 不运行 `toa_only_diag`。原因是它在 A5d 中是 side diagnostic；A7 的问题是低维几何、ToT 与 ToA 物理摘要整体是否还能补充 GMU，而不是继续比较特征组。
- 不运行 `GMU + CE+EMD`、`GMU + CE+EMD + handcrafted`、更多 Gaussian / ExpectedMAE / EMD 或 MMTM 等新结构。

固定配置：

- Dataset: `Alpha_100`
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Image input: `ToT + ToA`
- ToA transform: `relative_minmax`
- Hit mask: disabled
- Model: `dual_stream_gmu_aux`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Training config: A2 best
- Loss: `cross_entropy`, `onehot`
- Handcrafted fusion: `gated`
- Handcrafted source: `ToT + ToA`
- Seeds: `42, 43, 44`

`main_5feat`：

```text
active_pixel_count
bbox_fill_ratio
ToT_density
ToA_span
ToA_major_axis_corr_abs
```

配置文件：

```text
configs/experiments/a7_final_gmu_main5feat_gated_3seed.yaml
```

服务器命令：

```bash
tmux new -s a7
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a7_final_gmu_main5feat_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a7_final_gmu_main5feat_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a7_final_gmu_main5feat_gated_3seed --out outputs/a7_final_gmu_main5feat_gated_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a7_final_gmu_main5feat_gated_3seed_runs.csv --out outputs/a7_final_gmu_main5feat_gated_3seed_mean_std.csv
```

选择与解释规则：

- 只用 validation 判断 `main_5feat` 是否进入最终模型。
- Primary: `Val Acc`。
- Tie-break: `Val MAE` 更低、`Val Macro-F1` 更高。
- 如果 A7-1 的 Val Acc 高于 A7-0，则手工物理标量进入最终模型。
- 如果 Val Acc 基本持平，但 Val MAE 和 Val Macro-F1 同时改善，可作为 error-balanced final variant。
- 如果 validation 弱于 A7-0，即使 test 偶然更好，也不进入最终主模型，只作为诊断说明 GMU 图像分支已经吸收了大部分物理标量信息。
- 论文中需要区分两层 gate：GMU gate 融合 ToT/relative-ToA 图像分支；handcrafted gated fusion 融合 GMU 深度特征与五维物理标量。

A7 主表：

| 方案 | Val Acc | Val MAE | Val Macro-F1 | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A7-0 GMU_aux，复用 A4c | **70.20±0.67%** | **6.274±0.129** | 0.668±0.007 | **71.94±0.51%** | 5.721±0.009 | **0.691±0.009** |
| A7-1 GMU_aux + main_5feat gated | **70.20±0.61%** | 6.359±0.277 | **0.669±0.019** | 71.80±1.09% | **5.706±0.145** | 0.687±0.008 |

逐 seed 结果：

| Seed | 方案 | Val Acc | Val MAE | Val Macro-F1 | Test Acc | Test MAE | Test Macro-F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | A7-0 | 69.63% | 6.414 | 0.662 | 71.37% | 5.726 | 0.680 |
| 42 | A7-1 | 69.53% | 6.638 | 0.649 | 72.66% | 5.547 | 0.681 |
| 43 | A7-0 | 70.03% | 6.159 | 0.664 | 72.07% | 5.726 | 0.697 |
| 43 | A7-1 | 70.33% | 6.354 | 0.672 | 72.17% | 5.741 | 0.696 |
| 44 | A7-0 | 70.93% | 6.249 | 0.676 | 72.37% | 5.711 | 0.696 |
| 44 | A7-1 | 70.73% | 6.084 | 0.686 | 70.58% | 5.830 | 0.683 |

逐类 F1：

| 方案 | Split | 15 deg F1 | 30 deg F1 | 45 deg F1 | 60 deg F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| A7-0 GMU_aux | Val | **0.732** | **0.536** | 0.767 | 0.635 |
| A7-1 + main_5feat | Val | 0.724 | 0.526 | **0.769** | **0.656** |
| A7-0 GMU_aux | Test | **0.736** | **0.548** | 0.785 | **0.695** |
| A7-1 + main_5feat | Test | 0.734 | 0.527 | **0.792** | 0.694 |

额外观察：

- A7-1 的表现不是稳定提升：seed42 测试集更好但验证集变差；seed44 验证集部分指标更好但测试集 accuracy 明显下降。
- 手工特征对 `45 deg` / `60 deg` 有一点帮助，尤其 Val 60 deg F1 从 `0.635` 到 `0.656`；但代价是 `15 deg` / `30 deg` 下降。
- 最关键的 `30 deg` 类别没有改善，Val/Test F1 均低于 A7-0。
- A7-1 的 GMU 图像门控更偏向 ToT：Val `gate_tot` 从 `0.776` 增到 `0.793`，`gate_toa` 从 `0.224` 降到 `0.207`。这里记录的是 GMU 的 ToT/ToA 图像 gate，不等价于 handcrafted fusion gate。

A7 阶段判断：

- 按预先规则，不能把 A7-1 作为最终主模型。
- A7-1 与 A7-0 的 Val Acc 完全持平，但 Val MAE 变差 `+0.085 deg`；Val Macro-F1 仅提升 `+0.001`，属于极小变化，不满足“Val MAE 与 Val Macro-F1 同时改善”的条件。
- A7-1 可以作为物理标量补充诊断实验写入论文：五维物理标量能轻微影响部分类别和误差指标，但没有稳定改善最终 GMU 多模态模型。
- Alpha 最终端到端多模态主模型确定为 `dual_stream_gmu_aux + ToT/relative_minmax ToA + CE one-hot + no handcrafted`。
- 论文解释：GMU 图像分支已经吸收了大部分可由这些低维物理标量表达的信息；显式加入 `main_5feat` 不能进一步稳定提升 validation 指标，且会损伤 30 deg 困难类别。

## 四、Proton_C_7 主线

### B1：Proton_C_7 训练超参数搜索

状态：已完成。

总体目的：在 Proton_C_7 七分类 ToT 单模态上，复用 Alpha A1 确定的 ResNet18 no-maxpool 结构，只搜索训练超参数。

固定结构：

- Dataset: `Proton_C_7`
- Modality: ToT
- Model: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`、`conv1_stride=1`、`conv1_padding=0`
- Loss: `cross_entropy`
- Label: `onehot`
- Handcrafted: disabled

#### B1-1：learning rate × batch size

实验目的：第一轮搜索 `learning_rate × batch_size`，固定 `weight_decay=1e-4`。

搜索矩阵：

- `learning_rate=[1e-4,3e-4,1e-3]`
- `batch_size=[64,128,256]`

配置文件：

- `configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml`
- 历史旧配置 `configs/experiments/b1_proton_resnet18_tot_lr_batch.yaml` 不再用于正式训练。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_lr_batch_ep25 --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_runs.csv
```

关键结果：

- 20 epoch 旧结果和 25 epoch 中继结果均支持 `lr=3e-4`、`batch_size=128`。
- 中继到 25 epoch 后，未早停的若干 run 有小幅提升，但最佳组合不变。

#### B1-1 epoch-20 中继恢复方案

状态：已执行为诊断。

目的：B1-1 最初部分 run 只有 20 epoch，部分曲线仍在上升。为节约算力，使用中继方式将未早停 run 继续至 25 epoch。

脚本：

- `scripts/extend_runs.py`

决策：

- 中继结果带有 scheduler 续跑差异，因此只用于确认趋势，不作为严格等价新训练。
- 最佳超参数未改变。

#### B1-2：weight decay 搜索

实验目的：固定 B1-1 最佳 `lr=3e-4`、`batch_size=128`，搜索 `weight_decay`。

搜索矩阵：

- `weight_decay=[0,1e-5,1e-4]`

配置文件：

- `configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_weight_decay_ep25 --out outputs/b1_proton_c7_resnet18_tot_weight_decay_ep25_runs.csv
```

关键结果：

- `weight_decay=1e-4` 最优或与最优非常接近。
- `weight_decay=1e-5` 明显更差。

决策：

- B1-best 固定 `learning_rate=3e-4`、`batch_size=128`、`weight_decay=1e-4`。

#### B1-best：patience=8 three-seed baseline

状态：已完成。

背景决策：

- 最初 `early_stopping_patience=5` 对 Proton_C_7 过激，seed43/44 可能在后期恢复前被截断。
- 正式 B1-best 改为 `patience=8`，旧 patience=5 结果只作为早停诊断。

配置文件：

- `configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml`
- `configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml` 是旧 patience=5 版本，不作为正式 B1-best。

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_best_patience8_3seed --out outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv --out outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_mean_std.csv
```

关键结果：

- Test Acc `93.26 ± 1.64%`
- Test MAE `0.640 ± 0.161`
- Test Macro-F1 `0.952 ± 0.011`

决策：

- B1-best 是 Proton_C_7 的正式 CE one-hot baseline。
- 主要错误集中在相邻大角度，尤其 `60° ↔ 70°`，其次 `45° ↔ 50°`。

### B2：Proton_C_7 手工特征低成本验证

状态：已完成并收口。

实验目的：验证从 Alpha A5 筛选出的 ToT-only 可迁移低冗余手工特征是否能补充 Proton_C_7 ToT CNN。

固定配置：

- Dataset: `Proton_C_7`
- Image input: ToT only
- Scalar feature source: ToT only
- Training config: B1-best patience=8
- Seed: 42

#### B2a：concat

配置文件：

- `configs/experiments/b2_proton_c7_handcrafted_lowcorr_seed42.yaml`

实验组：

| 编号 | 组名 | 特征 | Fusion |
| --- | --- | --- | --- |
| B2a-1 | `geometry_lowcorr` | `active_pixel_count`, `bbox_fill_ratio` | concat |
| B2a-2 | `tot_lowcorr` | B2a-1 + `ToT_density` | concat |

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b2_proton_c7_handcrafted_lowcorr_seed42.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group b2_proton_c7_handcrafted_lowcorr_seed42 --out outputs/b2_proton_c7_handcrafted_lowcorr_seed42_runs.csv
```

关键结果：

- B2a-1 geometry concat: Test Acc `94.26%`，相对 B1-best seed42 `+0.17 pp`
- B2a-2 加 `ToT_density` 后明显下降到 `91.63%`

#### B2b：gated

配置文件：

- `configs/experiments/b2_proton_c7_handcrafted_gated_seed42.yaml`

实验组：

| 编号 | 组名 | 特征 | Fusion |
| --- | --- | --- | --- |
| B2b-1 | `geometry_lowcorr` | `active_pixel_count`, `bbox_fill_ratio` | gated |
| B2b-2 | `tot_lowcorr` | B2b-1 + `ToT_density` | gated |

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b2_proton_c7_handcrafted_gated_seed42.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group b2_proton_c7_handcrafted_gated_seed42 --out outputs/b2_proton_c7_handcrafted_gated_seed42_runs.csv
```

关键结果：

- B2b gated 可以抑制 `ToT_density` 在 concat 下造成的负面影响。
- B2b 没有显著超过 B1-best。

#### B2c：three-seed verification

状态：已配置，待运行。

配置文件：

- `configs/experiments/b2_proton_c7_handcrafted_transfer_TEMPLATE.yaml`
- `configs/experiments/b2c_proton_c7_geometry_handcrafted_3seed.yaml`

决策：

- B2c 只验证 `geometry_lowcorr`，即 `active_pixel_count` 和 `bbox_fill_ratio`。
- 不再加入 `ToT_density`。原因是 B2a 中 `geometry + ToT_density` concat 明显伤害结果，B2b gated 只是抑制其负面影响，并没有形成稳定增益。
- 同时比较 `concat` 与 `gated`，形成 `2 fusion modes × 3 seeds = 6 runs` 的小矩阵。
- 目的不是重新打开手工特征大网格，而是用 three-seed 正式确认 Proton_C_7 上最简单、最可迁移的几何标量是否真的能稳定补充 B1-best CNN。

B2c 实验矩阵：

| 编号 | 特征 | Fusion | Seeds |
| --- | --- | --- | --- |
| B2c-1 | `active_pixel_count`, `bbox_fill_ratio` | concat | 42, 43, 44 |
| B2c-2 | `active_pixel_count`, `bbox_fill_ratio` | gated | 42, 43, 44 |

服务器命令：

```bash
tmux new -s b2c
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/b2c_proton_c7_geometry_handcrafted_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b2c_proton_c7_geometry_handcrafted_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b2c_proton_c7_geometry_handcrafted_3seed --out outputs/b2c_proton_c7_geometry_handcrafted_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b2c_proton_c7_geometry_handcrafted_3seed_runs.csv --out outputs/b2c_proton_c7_geometry_handcrafted_3seed_mean_std.csv
```

选择与解释规则：

- 对照基线为 B1-best patience=8 three-seed。
- 只用 validation 判断几何标量是否形成稳定增益。
- Primary: Val Acc。
- Tie-break: Val MAE 更低、Val Macro-F1 更高，特别关注 45/50/60/70 deg 的 F1 和相邻混淆。
- 若 B2c 无稳定收益，则 B2 最终口径为：Proton_C_7 的 ToT 图像形态已被 CNN 较充分利用，低维几何标量主要与 CNN 表征冗余；gated 的价值更多是稳定化，而非显著增益。

### B3：Proton_C_7 有序角度损失

状态：已完成。

总体目的：针对 Proton_C_7 主要错误集中在相邻大角度的问题，引入角度有序性 loss/label strategy，降低 MAE、减少远距离误差和相邻大角度混淆。

固定配置：

- Dataset: `Proton_C_7`
- Input: ToT only
- Model: `resnet18_no_maxpool`
- Training config: B1-best patience=8
- Handcrafted: disabled
- Fusion: none

#### B3a：seed42 screening

配置文件：

- `configs/experiments/b3a_proton_c7_ordinal_loss_seed42.yaml`

实验矩阵：

| 方法 | 参数 |
| --- | --- |
| CE one-hot | baseline，复用 B1-best seed42 |
| Gaussian soft label | `sigma=5,10,15` |
| CE + ExpectedMAE | `lambda=0.05,0.10,0.20` |
| CE + EMD | `lambda=0.05,0.10,0.20` |

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b3a_proton_c7_ordinal_loss_seed42.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group b3a_proton_c7_ordinal_loss_seed42 --out outputs/b3a_proton_c7_ordinal_loss_seed42_runs.csv
```

关键结果：

- `CE + ExpectedMAE λ=0.05` 是 B3a 按 validation accuracy 选出的主候选。
- `CE + EMD λ=0.05` 非常接近，Test MAE 最低，作为 optional 对照进入 B3b。
- Gaussian soft label 不建议继续，尤其较大 sigma 会过度软化分类边界。

#### B3b：three-seed verification

配置文件：

- `configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml`
- `configs/experiments/b3b_proton_c7_ce_emd_optional_3seed.yaml`

服务器命令：

```bash
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml --skip-existing --continue-on-error
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_ce_emd_optional_3seed.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group b3b_proton_c7_expected_mae_3seed --out outputs/b3b_proton_c7_expected_mae_3seed_runs.csv
python scripts/summarize.py --group b3b_proton_c7_ce_emd_optional_3seed --out outputs/b3b_proton_c7_ce_emd_optional_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/b3b_proton_c7_expected_mae_3seed_runs.csv --out outputs/b3b_proton_c7_expected_mae_3seed_mean_std.csv
python scripts/aggregate_seeds.py --summary outputs/b3b_proton_c7_ce_emd_optional_3seed_runs.csv --out outputs/b3b_proton_c7_ce_emd_optional_3seed_mean_std.csv
```

关键结果：

| 设置 | Test Acc | Test MAE | Test Macro-F1 | Test High-F1 |
| --- | ---: | ---: | ---: | ---: |
| B1-best CE | `93.26 ± 1.64%` | `0.640 ± 0.161` | `0.952 ± 0.011` | `0.917 ± 0.020` |
| CE + ExpectedMAE `λ=0.05` | `94.38 ± 0.08%` | `0.532 ± 0.010` | `0.960 ± 0.001` | `0.930 ± 0.001` |
| CE + EMD `λ=0.05` | `94.35 ± 0.09%` | `0.532 ± 0.011` | `0.959 ± 0.001` | `0.930 ± 0.001` |

决策：

- B3b 是强正结果。
- `CE + ExpectedMAE λ=0.05` 是 Proton_C_7 当前推荐 loss，因为它是 B3a validation-selected 主候选，且 B3b 中 Val Acc、Val Macro-F1、Val High-angle F1 最优或并列最优。
- `CE + EMD λ=0.05` 作为有序损失对照，证明角度距离约束确实有效。

### B4：Proton_C_7 最终模型确认

状态：已形成最终建议；无需新增训练组。

最终配置建议：

- 以 B1-best 结构和训练超参为基础。
- 将 B3b `CE + ExpectedMAE λ=0.05` 作为推荐 loss。
- 若论文需要最终 Proton 模型表，可将 B1-best CE、B3b ExpectedMAE、B3b CE+EMD 并列展示，突出有序损失对相邻角度混淆的改善。

## 五、数据分析链路

状态：已实现，独立于训练主链路。

目的：为论文提供数据集统计、特征分布、代表性样本、近垂直角度分辨极限分析。

新增结构：

- `timepix/analysis/io.py`
- `timepix/analysis/features.py`
- `timepix/analysis/stats.py`
- `timepix/analysis/ml.py`
- `timepix/analysis/plotting.py`
- `timepix/analysis/representative.py`
- `timepix/analysis/tables.py`
- `timepix/analysis/reports.py`
- `timepix/analysis/progress.py`
- `timepix/analysis/workbook.py`

入口脚本：

- `scripts/analyze_datasets.py`
- `scripts/analyze_resolution_limit.py`
- `scripts/make_analysis_report.py`

命令：

```bash
python scripts/analyze_datasets.py
python scripts/analyze_resolution_limit.py
python scripts/make_analysis_report.py
```

关键决策：

- `Proton_C` 用于全量数据分析，`Proton_C_7` 用于训练。
- `Proton_C_7` 在数据分析脚本中不要求独立目录，直接从全量 `Proton_C` 派生，角度为 `10,20,30,45,50,60,70`；已修正为 45°，不是 40°。
- 数据分析链路只按 ToT 分析 Proton_C，不假设 Proton 有 ToA。
- Windows 本地当前环境为 `timepix-local`；UMAP 已可用，t-SNE 默认跳过，需显式加 `--tsne`。
- 输出 CSV + Markdown 表格，汇总表另存 xlsx；图片保存 PNG/PDF，PNG 使用 300 dpi。
- `make_analysis_report.py` 默认不把 `04_event_features_*` 和 `proton_c_near_vertical_features.csv` 这类原始长表写入总 xlsx，避免工作簿巨大；这些长表仍以 CSV 保留。

本地 v2 完整分析已完成：

| 输出 | 路径 | 备注 |
| --- | --- | --- |
| 数据集分析 | `outputs/data_analysis_v2_local/` | 包含 `Alpha_100`、全量 `Proton_C`、派生 `Proton_C_7` |
| 近垂直分析 | `outputs/resolution_limit_v2_local/` | `Proton_C` 的 `80,82,84,86,88,90`，ToT-only |
| 汇总报告 | `outputs/analysis_report_v2_local.md` | 合并数据集分析与近垂直分析 |
| 汇总表工作簿 | `outputs/analysis_tables/timepix_analysis_tables_v2_local.xlsx` | 37 个 sheet，汇总表为主 |

本地数据审计结果：

- `Proton_C` 实际角度为 `10,20,30,45,50,60,70,80,82,84,86,88,90`。
- `Proton_C_7` 派生角度为 `10,20,30,45,50,60,70`，对应样本数分别为 `5620,7605,10985,12588,14075,22589,29476`。
- `proton_input_shape_audit.csv` 确认 `Proton_C` 和派生 `Proton_C_7` 文件保存尺寸均为 `50x50`；训练配置 `data.crop_size=0`，Dataset loader 不 resize，ResNet 有效输入为 `1x50x50`。此前“32x32”属于人工记录冲突，不符合当前本地文件与训练配置。
- `10_alpha_class_summary.csv` 已修复 Alpha split 翻倍：按样本 key 统计，15/30/45/60 的 train/val/test 分别为 `2803/350/351`、`1152/144/145`、`2820/352/354`、`1241/155/156`。
- `alpha_toa_negative_audit.csv` 发现 `Alpha_100` 的 45° ToA 只有一个负值样本：`45/1_r1252__0003.txt`，对应 ToT 文件存在；raw ToA 非零值中 34 个为负，初步标记为单样本 raw ToA/timestamp 异常，后续论文中不宜直接扩大解释。
- `proton_angle_consistency_audit.csv` 发现 75、83、85° 只存在于 `outputs/splits/Proton_C_ToT_seed42_0.8_0.1_0.1.json`，当前本地 `Proton_C` inventory 没有对应目录，标记为 `split_residual_no_current_data`，即旧 full split 残留或未同步旧数据。
- 近垂直 `80-90°` 六分类传统 ML 基线最高约 `18.8%` accuracy，接近随机基线 `16.67%`，支持后续论文采用谨慎的“当前表示和已测模型/特征族下可分性弱”表述。
- 近垂直单特征相邻角度 AUC 已输出为 `near_vertical_single_feature_auc.csv`，核心特征大多贴近 `0.50-0.52`；相邻角度效应量输出为 `near_vertical_pairwise_effect_size.csv`。
- 小样本过拟合 sanity check 已输出 `near_vertical_overfit_experiment.csv` 和 `figures/near_vertical_overfit_learning_curve.png`；本地 CPU 跑到 200 epoch 后，每类 5/10/50/100 样本的最终 train acc 约为 `50.0%/50.0%/30.3%/29.0%`，没有接近 100%。这支持“需要进一步排查表示/标签/训练可分性”的谨慎表述，不能单独作为绝对不可分证明。
- 未提供传感器厚度，因此几何投影长度只生成缺失说明；未发现原始候选事件和清洗日志，因此清洗阈值表基于清洗后最终数据集的 `active_count` / `active_sum` 范围推断。
- 本地 `timepix-local` 的 PyTorch 是 `2.11.0+cpu`，`torch.cuda.is_available()` 为 `False`。未修改显卡驱动、系统 CUDA 或全局环境；如需 GPU，只建议新建独立 conda 环境安装匹配驱动的 CUDA 版 PyTorch。

## 六、历史配置、模板与过渡配置

| 配置 | 当前定位 |
| --- | --- |
| `alpha_resnet18_tot.yaml` | 早期 baseline 配置，保留兼容，不作为最新 Alpha 主线 |
| `alpha_resnet18_tot_toa.yaml` | 早期模态配置，正式 A4 使用 `a4_modality_comparison*.yaml` |
| `alpha_resnet18_tot_handcrafted_concat.yaml` | 早期手工特征配置，正式 A5 使用 `a5*.yaml` |
| `alpha_resnet18_tot_handcrafted_gated.yaml` | 早期手工特征配置，正式 A5 使用 `a5*.yaml` |
| `proton_resnet18_tot.yaml` | 早期 Proton baseline，正式 B 系列使用 `Proton_C_7` 配置 |
| `b1_proton_resnet18_tot_lr_batch.yaml` | Proton_C 旧命名配置，不作为正式 B1 |
| `compare_models.yaml` | 早期模型对比配置，正式 A3 使用 `a3_backbone_comparison*.yaml` |
| `compare_losses.yaml` | 早期 loss 对比配置，正式 A6/B3 使用 ordinal loss 配置 |
| `a5b/a5c/a5d *_TEMPLATE.yaml` | 生成或迁移实验时的模板，不代表已运行正式组 |
| `b2_proton_c7_handcrafted_transfer_TEMPLATE.yaml` | B2c 预留模板；B2c 当前不推进 |

## 七、当前待办

1. 另一个窗口整理最终 CSV 结果后，回填本文档中仍需更精确的数值表。
2. 论文写作时应将 A4c GMU 的选择依据写为 validation Macro-F1 接近最优、Val MAE 更好、结构解释更符合物理结论；不能用 test 结果反向选择 GMU。
3. 5.5 Pro 交接文档需要与本日志同步，尤其是 A4c 最终架构口径、A5 收口口径、A6 负结果口径、A7 最终组件确认口径、B3 最终 loss 口径。

## 八、Particle 粒子识别数据处理记录

### Particle Stage-1：单粒子候选响应矩阵提取

数据路径：

- 原始数据：`E:\TimepixData\particle\raw`
- 脚本目录：`ProcessProgram\Particle`
- Stage-1 输出：`E:\TimepixData\particle\stage1_single_particle_candidates_100x100`

关键决策：

- 第一阶段目标是从原始 `256x256` 探测器帧中稳定提取“单粒子候选响应矩阵”，不在此阶段进行物理质量筛选。
- 使用 ToT 连通域作为候选粒子区域，ToA/ToT 文件必须一一配对。
- 采用 bbox 几何居中：以连通域 bbox 放入 `100x100` 画布中央，只复制连通域自身像素，周围补 0。
- 默认拒绝触边事件，因为触边响应可能被探测器边缘截断。
- 要求 ToT 与 ToA 候选连通区域完全一致；如果区域不一致，记录为 `toa_tot_region_mismatch` 并剔除。
- 由于已经改为 bbox 几何居中，不再存在质心平移溢出这一失败路径。

Stage-1 完成状态：

- paired frames: `2522`
- saved candidates: `119667`
- rejected components: `4619`
- failed pairs: `0`
- Am: saved `5736`, rejected `451`
- Co60: saved `98079`, rejected `3094`
- Sr: saved `15852`, rejected `1074`
- 主要拒绝原因：`touches_detector_edge = 4595`，`toa_tot_region_mismatch = 24`

### Particle Stage-2a：原始 ToT/形态簇特征统计

当前阶段不做聚类。先根据本数据集的原始特征分布判断是否需要偏态校正、特征删除或标准化策略，再进入 Stage-2b 聚类。

输出路径：

```text
E:\TimepixData\particle\stage2_cluster_features_v1
```

命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2_extract_cluster_features.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --output-root E:\TimepixData\particle\stage2_cluster_features_v1
```

当前特征只使用 ToT 与几何形态，不使用 ToA 特征，并按用户要求去除 `mean_ToT`：

- `Npix`: active pixel count
- `S_total_ToT`: active pixels 上的 ToT 总和
- `Pmax`: `max(ToT) / S_total_ToT`
- `Rg`: active-pixel 坐标的 radius of gyration
- `E_pca`: active-pixel 坐标 PCA 主/次轴伸长率，带小簇正则
- `Fbox`: `active pixels / bbox area`

Stage-2a 产物：

- `features_raw.csv`
- `feature_summary.csv`
- `feature_correlation_pearson.csv`
- `feature_correlation_spearman.csv`
- `feature_notes.md`
- `figures/stage2_raw_feature_histograms_by_particle_count.*`
- `figures/stage2_raw_feature_scatter_pairs.*`
- `figures/stage2_<particle>_raw_feature_histograms_by_angle_count.*`

初步观察：

- `Npix`、`S_total_ToT`、`Rg` 均明显偏态，后续聚类前大概率需要 `log1p` 或类似压缩。
- `Npix` 与 `Rg` 的 Spearman correlation 约 `0.990`，二者高度冗余；`Npix` 与 `S_total_ToT` 也高度相关。
- Am 呈现高 ToT / 高 Npix 主团，同时混有低 Npix / 低 ToT 小簇。
- Co60 以小而紧凑候选为主，`Npix` 中位数约为 `3`，`S_total_ToT` 中位数约为 `87.9`。
- Sr 分布比 Co60 更宽、更拉长，`Npix` 中位数约为 `6`，`E_pca` 中位数约为 `1.62`，更接近 electron-like 轨迹响应。
- Stage-2b 不应直接使用原始数值聚类；下一步需要先决定变换、标准化和保留特征集合。

### Particle Stage-2b：分粒子变换与无监督聚类诊断

关键决策：

- 聚类必须按粒子源独立进行：`Am`、`Co60`、`Sr` 分别拟合变换、scaler、HDBSCAN 和 GMM，不把不同粒子混在同一个 clustering 空间里。
- 聚类输入仍然只使用 ToT/形态特征，不使用 ToA 特征。
- 本阶段目标不是直接生成最终 beta/gamma/alpha 标签，而是检查每个源内部是否存在稳定可解释的响应形态结构。

变换策略：

- `Npix`、`S_total_ToT`、`Rg` 使用 `log1p`，压缩长尾。
- `E_pca` 使用 `log1p(E_pca - 1)`，因为伸长率从 1 起。
- `Pmax`、`Fbox` 保持 identity。
- 每个粒子源单独做 robust scaling：`(x - median) / IQR`。

命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2b_particlewise_clustering.py `
  --stage2a-root E:\TimepixData\particle\stage2_cluster_features_v1 `
  --output-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1
```

输出路径：

```text
E:\TimepixData\particle\stage2b_particlewise_clustering_v1
```

主要产物：

- `features_transformed_clustered.csv`
- `transform_parameters.csv`
- `gmm_model_selection.csv`
- `particle_cluster_summary.csv`
- `cluster_feature_summary.csv`
- `stage2b_notes.json`
- `figures/stage2b_gmm_bic_by_particle.*`
- `figures/<particle>_gmm_k2_label_*.png/pdf`
- `figures/<particle>_gmm_k3_label_*.png/pdf`
- `figures/<particle>_hdbscan_label_*.png/pdf`

Stage-2b 结果摘要：

| 粒子源 | 样本数 | GMM BIC 最优成分数 | HDBSCAN clusters | HDBSCAN noise rate | GMM k=3 主要结构 |
| --- | ---: | ---: | ---: | ---: | --- |
| Am | 5736 | 3 | 24 | 33.42% | 单像素低 ToT 小簇；高 Npix/高 ToT 主簇；低到中 Npix 混入簇 |
| Co60 | 98079 | 3 | 47 | 4.33% | 单像素紧凑簇；2-4 像素紧凑主簇；更大/更分散扩展簇 |
| Sr | 15852 | 3 | 14 | 24.57% | 小而峰值集中的紧凑簇；中等簇；高 Npix/低 Pmax/高 E_pca 延展簇 |

GMM k=3 中位特征示例：

- `Co60`：
  - label 1: `n=14329`，`Npix median=1`，`S_total_ToT median=20.17`，`Pmax median=1.0`，极紧凑单像素簇。
  - label 0: `n=57562`，`Npix median=3`，`S_total_ToT median=78.67`，`Pmax median=0.638`，小而紧凑主簇。
  - label 2: `n=26188`，`Npix median=5`，`S_total_ToT median=167.79`，`Pmax median=0.434`，相对更大、更分散，可能对应 electron-like/track-like 候选。
- `Sr`：
  - label 0: `n=4168`，`Npix median=2`，`Pmax median=0.762`，紧凑簇。
  - label 2: `n=6768`，`Npix median=6`，`Pmax median=0.385`，中等簇。
  - label 1: `n=4916`，`Npix median=15`，`Pmax median=0.191`，`E_pca median=3.399`，明显延展轨迹簇。
- `Am`：
  - label 0: `n=1000`，单像素低 ToT 小簇。
  - label 1: `n=3577`，`Npix median=39`，`S_total_ToT median=2711.93`，高 ToT/大簇主响应。
  - label 2: `n=1159`，`Npix median=3` 但覆盖到较宽范围，属于混入/过渡候选。

阶段判断：

- GMM 的 3 成分结果有较清楚的形态解释：紧凑小簇、中等簇、延展轨迹/大响应簇。
- HDBSCAN 对 Co60 噪声率较低，可作为异常/稀有形态诊断；但对 Am/Sr 噪声率较高，并且会把离散小簇和不同 Npix 模式切成许多小簇，暂不宜直接作为最终清洗标签。
- 当前不能把某个 GMM label 直接命名为“纯 beta”或“纯 gamma”。下一步更适合抽样查看每个粒子源内部 GMM 簇的代表性 10x10 图像，再结合 Sr 作为 electron-like reference 和 Co60 的混合场性质解释簇。

### Particle Stage-2c：GMM 簇代表样本图

目的：从 Stage-2b 的分粒子 GMM 结果中抽取每个粒子/簇的代表样本，画出 `10x10` ToT crop，用于人工确认形态。此阶段仍不直接给 cluster 贴物理真值标签。

命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2c_cluster_representatives.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --stage2b-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1 `
  --output-root E:\TimepixData\particle\stage2c_cluster_representatives_v1 `
  --label-column gmm_k3_label `
  --confidence-column gmm_k3_confidence `
  --samples-per-cluster 10 `
  --min-confidence 0.90
```

输出路径：

```text
E:\TimepixData\particle\stage2c_cluster_representatives_v1
```

产物：

- `cluster_sample_manifest.csv`
- `cluster_sample_summary.csv`
- `stage2c_notes.json`
- `figures/Am_gmm_k3_label_tot_samples_10x10.*`
- `figures/Co60_gmm_k3_label_tot_samples_10x10.*`
- `figures/Sr_gmm_k3_label_tot_samples_10x10.*`

抽样规模：

- 共抽取 `90` 个代表事件：`3` 个粒子源 × `3` 个 GMM cluster × 每簇 `10` 个样本。
- 抽样优先使用 `gmm_k3_confidence >= 0.90` 的高置信候选。

代表图观察：

- `Am gmm_k3 label 1`：大而圆、ToT 很高的主响应，`Npix` 约 `25-46`，形态非常稳定，适合作为 alpha-like / heavy charged main response 的候选核心。
- `Am label 0`：单像素低 ToT 小簇，明显不像 Am 主 alpha 响应，更可能是混入、低能次级响应或噪声候选。
- `Am label 2`：混合性更强，既有 2-3 像素小簇，也有少量大形态/异常轨迹，不宜直接纳入高纯 alpha-like 主簇。
- `Co60 label 1`：单像素紧凑簇。
- `Co60 label 0`：2-4 像素小型紧凑簇，是 Co60 内部数量最大的紧凑响应之一。
- `Co60 label 2`：更大、更分散，包含短轨迹/小段拖尾事件，比 label 0/1 更接近 electron-like 或 beta-like 候选响应。
- `Sr label 1`：明显长轨迹/弯曲轨迹，`E_pca` 高、`Pmax` 低，更接近 electron-like/beta-like 主簇。
- `Sr label 0`：小而峰值集中的紧凑簇。
- `Sr label 2`：中等像素数的小簇/短簇，介于紧凑簇和明显轨迹簇之间。

阶段判断：

- 如果目标是先构建高置信训练子集，当前最稳的候选是：
  - alpha-like: `Am gmm_k3 label 1`
  - beta/electron-like: `Sr gmm_k3 label 1`
  - gamma-like/photon-initiated-like: 还不能直接定，需要重点比较 Co60 紧凑簇 `label 0/1` 与 Sr 紧凑簇 `label 0/2` 的差异；仅凭这些形态图不能保证 Co60 紧凑簇就是纯 gamma。
- 下一步建议不是立即导出最终数据集，而是先做候选簇筛选方案讨论：确定哪些 label 保留、哪些 label 作为不确定/剔除，再生成清洗后的 candidate dataset。

### Particle Stage-2d：分粒子可视化聚类结构检查

目的：在 Stage-2b/2c 的自动聚类结果之后，增加一个更直观的人工检查阶段。用户明确提出，不希望直接依赖 GMM/HDBSCAN 自动标签来生成训练标签，而是希望先像 PCA/KMeans 二维主轴图那样，肉眼检查每个放射源内部是否存在明显分离结构。因此 Stage-2d 定位为“可视化诊断”，不是“自动打标签”。

关键决策：

- 仍然按粒子源分别处理 `Am`、`Co60`、`Sr`，不把不同源混在同一个 PCA 或聚类空间。
- 输入使用 Stage-2b 已经变换和 robust scaling 后的六个 ToT/形态特征，不引入 ToA 特征。
- 输出四类图：
  - `density_*`：二维密度/计数图，颜色条表示每个 bin 内的候选个数。
  - `pca_kmeans_*_reference`：PCA 二维图上叠加 KMeans `k=2/3` 的参考颜色，仅用于观察可能边界，不能作为最终物理标签。
  - `pca_gmm_*_reference` 与 `pca3_gmm_*_reference`：GMM 在 PCA 空间中的概率云参考颜色，其中 3D 图使用 `PC1/PC2/PC3` 重新拟合 `GMM k=3`。
  - `angle_*`：按角度分面的密度图和参考聚类图，用于检查所谓簇结构是否主要来自角度效应。
- 当前阶段不生成 cleaned dataset，也不把 KMeans 或 GMM label 直接写成 alpha/beta/gamma 标签。
- 用户希望像 MATLAB 一样拖拽旋转 3D 点云，因此新增 `--interactive-html` 可选输出；该输出只服务人工观察，不改变聚类、清洗或标签策略。

命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2d_visual_cluster_inspection.py `
  --stage2b-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1 `
  --output-root E:\TimepixData\particle\stage2d_visual_cluster_inspection_v1 `
  --sample-size 30000
```

交互式 3D 输出命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2d_visual_cluster_inspection.py `
  --stage2b-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1 `
  --output-root E:\TimepixData\particle\stage2d_visual_cluster_inspection_v1 `
  --sample-size 30000 `
  --interactive-html `
  --interactive-sample-size 30000
```

本地依赖：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe -m pip install plotly
```

输出路径：

```text
E:\TimepixData\particle\stage2d_visual_cluster_inspection_v1
```

主要产物：

- `pca_scores_with_kmeans_reference.csv`
- `pca_loadings.csv`
- `pca_summary.csv`
- `stage2d_notes.json`
- `figures/<particle>_density_log1p_Npix_vs_Pmax.*`
- `figures/<particle>_density_log1p_Rg_vs_log1p_Epca_minus1.*`
- `figures/<particle>_pca_density_PC1_vs_PC2.*`
- `figures/<particle>_pca_kmeans_k2_reference.*`
- `figures/<particle>_pca_kmeans_k3_reference.*`
- `figures/<particle>_pca_gmm_k2_reference.*`
- `figures/<particle>_pca_gmm_k3_reference.*`
- `figures/<particle>_pca3_gmm_k3_reference.*`
- `figures/<particle>_angle_density_log1p_Npix_vs_Pmax.*`
- `figures/<particle>_angle_density_log1p_Rg_vs_log1p_Epca_minus1.*`
- `figures/<particle>_angle_pca_density_PC1_vs_PC2.*`
- `figures/<particle>_angle_pca_kmeans_k3_reference.*`
- `figures/<particle>_angle_pca_gmm_k3_reference.*`
- `interactive/<particle>_pca3_gmm_k3_interactive.html`

PCA 解释率：

| 粒子源 | PC1 explained | PC2 explained | PC3 explained | PC1+PC2 | PC1+PC2+PC3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Am | 0.804 | 0.183 | 0.011 | 0.986 | 0.997 |
| Co60 | 0.807 | 0.082 | 0.048 | 0.889 | 0.937 |
| Sr | 0.833 | 0.096 | 0.035 | 0.928 | 0.964 |

PCA loading 摘要：

- `Am` 的 PC1 主要由 `scaled_E_pca` 主导，PC2 主要区分峰值集中度与尺寸/总 ToT/扩展程度。
- `Co60` 的 PC1 主要是尺寸/扩展轴：`Npix`、`Rg`、`S_total_ToT` 为正，`Pmax`、`Fbox` 为负；PC2 主要反映伸长率。
- `Sr` 的 PC1 同样主要表示尺寸/扩展程度，PC2 在总 ToT 与伸长率之间形成对照。

阶段观察：

- `Am` 内部结构最明显，存在稳定的大响应主区域，和 Stage-2c 中 `Am gmm_k3 label 1` 的 alpha-like / heavy charged main response 观察一致。
- `Sr` 呈现从紧凑簇到延展轨迹簇的连续过渡，右侧延展区域更符合 electron-like/beta-like 候选，但边界不是天然断裂。
- `Co60` 也呈现紧凑到延展的连续结构。紧凑区域不能仅凭当前图称为 pure gamma，因为硅探测器记录到的 gamma 响应本质上也是 photon-initiated secondary electron response，并且 Sr 内部也存在紧凑小簇。
- KMeans/GMM 参考图可以帮助定位“左侧紧凑 / 右侧延展”的视觉区域，但边界是算法强行切分或概率模型拟合出来的，不应直接作为训练标签。
- 分角度图显示，Co60 和 Sr 在每个角度内部仍保留从紧凑到延展的形态梯度，说明结构并非完全由角度混合造成；但边界仍是连续过渡，不是天然断裂。
- 3D PCA + GMM 图比二维图更直观地展示了条带状小簇与延展云团：`Am` 主响应与小簇/异常形态更清楚，`Co60` 和 `Sr` 仍表现为连续形态云团，不能直接用颜色当物理真值。
- Plotly 交互式 HTML 已生成 `Am`、`Co60`、`Sr` 三个文件，可在浏览器中拖拽旋转、滚轮缩放，并通过 hover 查看 `sample_key`、粒子源、角度、核心特征、`GMM3D` label 与 confidence。该交互视图用于人工理解 3D 点云结构，不能作为自动标签来源。

阶段决策更新：

- 从 GMM/PCA 图来看，`Co60` 和 `Sr` 内部均没有可以直接切分的稳定结构簇；不再尝试从二者中自动分离纯 beta/gamma 事件。
- `Am` 主响应与低像素小簇分离较明显，可以用简单 `Npix` 阈值清掉非主响应候选。
- 后续标签改为放射源标签：`Am`、`Co60`、`Sr`。清洗目标改为去除各源内部明显异常事件，而不是生成逐事件粒子真值标签。

### Particle Stage-3a：放射源标签保守清洗审计

目的：基于 Stage-2a 的 ToT/形态特征，生成一个保守的 source-label cleaning audit。该阶段只提出 keep/reject 建议和审计图，不直接导出最终 cleaned dataset。

关键决策：

- 标签保持为放射源：`Am`、`Co60`、`Sr`。
- 不使用 GMM/KMeans/HDBSCAN label 作为物理标签。
- `Am` 使用简单 active pixel count 阈值清除低像素非主响应。
- `Co60` 和 `Sr` 只剔除明显异常尾部，包括低信号噪声样、疑似堆叠的大事件、极端稀疏形态、多特征极端离群。
- 该阶段输出审计结果，供人工确认后再决定 Stage-3b 是否正式导出清洗数据集。

命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage3a_source_cleaning_audit.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --stage2a-root E:\TimepixData\particle\stage2_cluster_features_v1 `
  --output-root E:\TimepixData\particle\stage3a_source_cleaning_audit_v2
```

输出路径：

```text
E:\TimepixData\particle\stage3a_source_cleaning_audit_v2
```

主要产物：

- `source_cleaning_audit.csv`
- `rejected_candidate_audit.csv`
- `rejected_sample_manifest.csv`
- `cleaning_rule_summary.csv`
- `cleaning_counts_by_source_angle.csv`
- `stage3a_notes.json`
- `figures/stage3a_keep_reject_feature_histograms.*`
- `figures/stage3a_keep_reject_pca_overlay.*`
- `figures/Am_stage3a_rejected_samples_10x10.*`
- `figures/Co60_stage3a_rejected_samples_10x10.*`
- `figures/Sr_stage3a_rejected_samples_10x10.*`

Stage-3a v1 图像检查结论：

- `Am` rejected crop 基本都是 `Npix=1-3` 的小点或小块，作为非主响应剔除合理。
- `Co60`/`Sr` 的 `low_signal_noise_like` rejected crop 主要是 `Npix=1` 且低 ToT 的单像素点，剔除合理。
- 但 `Co60`/`Sr` 的 `extreme_large_component`、`extreme_sparse_shape` 和 `multi_feature_outlier` 包含大量清楚的细长轨迹；这些在统计上是尾部，但物理上可能是有效 source response，尤其是 Sr 的 beta/electron-like 长轨迹。
- 因此 v2 调整为：Co60/Sr 只拒绝 `low_signal_noise_like`；大轨迹、稀疏形态和多特征离群仅写入 `review_flags`，不影响 `recommended_keep`。

Stage-3a v2 当前审计结果：

| 粒子源 | Total | Kept | Rejected | Reject rate | 主要拒绝逻辑 |
| --- | ---: | ---: | ---: | ---: | --- |
| Am | 5736 | 3757 | 1979 | 34.50% | `am_low_npix` |
| Co60 | 98079 | 93714 | 4365 | 4.45% | `low_signal_noise_like` |
| Sr | 15852 | 15279 | 573 | 3.61% | `low_signal_noise_like` |
| All | 119667 | 112750 | 6917 | 5.78% | 保守异常剔除 |

规则摘要：

- `Am` 的全局 `Npix` 阈值由 1D Otsu 在 Am `Npix` 分布上估计，当前为 `21`；即 `Npix < 21` 的 Am 候选标记为非主响应。
- `Co60`/`Sr` 按 `particle + angle` 统计低信号阈值，避免角度依赖形态造成误删。
- `Co60`/`Sr` 中小而紧凑的事件不会因为小 `Npix` 被直接删除；只有同时处于低像素与低 ToT 的低信号候选才标记为 `low_signal_noise_like` 并剔除。
- `Co60`/`Sr` 的 `extreme_large_component`、`extreme_sparse_shape` 和 `multi_feature_outlier` 保留为 `review_flags`，对应保留候选数量为：Co60 `744/496/231`，Sr `140/81/0`。

下一步建议：

- 人工检查 `stage3a_keep_reject_feature_histograms`、`stage3a_keep_reject_pca_overlay` 和三张 rejected sample crop 图。
- 先检查 v2 的 `Co60_stage3a_rejected_samples_10x10` 和 `Sr_stage3a_rejected_samples_10x10` 是否只剩低信号单像素/小簇。
- 如果 v2 rejected samples 符合预期，则进入 Stage-3b，按 `source_cleaning_audit.csv` 中的 `recommended_keep` 生成 `particle_multimodal_cleaned_tot_toa`。
- `review_flags` 可保留到最终 manifest 中，供后续误差分析或更严格清洗版本使用，但不作为当前主清洗规则。

### Particle Stage-3b：放射源标签 cleaned ToT/ToA 数据集导出

目的：根据 Stage-3a v2 的 `recommended_keep` 正式导出当前粒子识别数据集。标签保持为放射源标签 `Am`、`Co60`、`Sr`；不声明为逐事件纯 alpha/beta/gamma 真值。

关键决策：

- 只导出 `recommended_keep == true` 的 ToT/ToA 成对文件。
- 原始 Stage-3a 的 `reject_reasons` 和 `review_flags` 保留在 manifest 中。
- `review_flags` 对应的长轨迹/稀疏/离群样本仍被保留，用于后续训练或误差分析。
- 输出目录使用清晰命名：`particle_source_label_cleaned_tot_toa_v1`。

命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage3b_export_cleaned_dataset.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --audit-path E:\TimepixData\particle\stage3a_source_cleaning_audit_v2\source_cleaning_audit.csv `
  --output-root E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1
```

输出路径：

```text
E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1
```

主要产物：

- `dataset/Am/ToT/*.txt`
- `dataset/Am/ToA/*.txt`
- `dataset/Co60/ToT/*.txt`
- `dataset/Co60/ToA/*.txt`
- `dataset/Sr/ToT/*.txt`
- `dataset/Sr/ToA/*.txt`
- `manifests/cleaned_manifest.csv`
- `manifests/cleaned_rejected_manifest.csv`
- `manifests/cleaned_counts_by_particle.csv`
- `manifests/cleaned_review_flags.csv`
- `summary.json`

导出结果：

| 粒子源 | Exported ToT | Exported ToA | Rejected retained in audit |
| --- | ---: | ---: | ---: |
| Am | 3757 | 3757 | 1979 |
| Co60 | 93714 | 93714 | 4365 |
| Sr | 15279 | 15279 | 573 |
| All | 112750 | 112750 | 6917 |

注意：

- 第一次导出因命令超时被中断，产生了 partial output；用户手动删除 v1/v2 目录后重新导出 v1，最终导出成功。
- `summary.json` 中记录 `total_count=119667`、`exported_count=112750`、`rejected_count=6917`、`reject_rate=5.78%`。
- 后续训练应使用 `particle_source_label_cleaned_tot_toa_v1` 作为当前 source-label cleaned 数据集，而不是 Stage-1 技术候选目录。

### Particle Framework-1：source / particle 分类任务框架适配

状态：第一步框架适配已实施，尚未开始正式训练配置与实验编号。

目的：

- 在不破坏既有 Alpha / Proton 角度识别任务的前提下，让新训练框架支持 Timepix3 `ToT + ToA` 放射源/粒子类别识别任务。
- 当前粒子数据集标签仍为 source label：`Am`、`Co60`、`Sr`；后续若人工或物理规则进一步分离出纯响应，标签可能变为 `Alpha`、`Beta`、`Gamma`。
- 因此类别名称不能在代码中硬编码，必须能从数据集顶层文件夹自动提取。

数据集：

```text
E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1\dataset
```

当前导出计数：

| 类别文件夹 | ToT | ToA | shape |
| --- | ---: | ---: | --- |
| `Am` | 3757 | 3757 | 100x100 |
| `Co60` | 93714 | 93714 | 100x100 |
| `Sr` | 15279 | 15279 | 100x100 |

关键决策：

- 新增 `dataset.label_type`，用于区分两类任务：
  - `angle_folder`：原有角度识别任务，文件夹名必须能转为数值角度，继续输出 `MAE`、`P90`、角度混淆等指标。
  - `categorical_folder`：普通类别识别任务，文件夹名作为类别名，不再计算角度 `MAE`、`P90`、`high-angle F1` 或相邻角度混淆。
- `categorical_folder` 默认按顶层文件夹名字典序自动生成 `class_names` 和 `label_map`；如后续需要固定顺序，可在 dataset config 中显式写入 `class_names`。
- `categorical_folder` 只允许无序分类监督，例如 `cross_entropy + onehot`。`gaussian` soft label、`emd`、`ce_expected_mae`、`ce_emd` 和 regression 只允许用于 `angle_folder`。
- 类别识别任务的主要指标改为：
  - `accuracy`
  - `balanced_accuracy`
  - `macro_f1`
  - `weighted_f1`
  - per-class `precision / recall / F1 / support`
  - confusion matrix
- 鉴于当前 `Co60` 数量远高于 `Am` 和 `Sr`，后续 Particle 实验不应只看 accuracy；建议优先记录 `macro_f1` 和 `balanced_accuracy`。

已新增/修改的适配点：

- `configs/datasets/particle_source_3.yaml`
  - 新增 `Particle_Source_3` 数据集配置。
  - `default_modalities: [ToT, ToA]`
  - `label_type: categorical_folder`
- `timepix/data/dataset.py`
  - `collect_samples()` 支持 `angle_folder` 与 `categorical_folder`。
  - 类别任务从文件夹名自动提取类别名；角度任务继续按数值角度排序。
- `timepix/data/builders.py`
  - 将 `label_type`、`class_names`、`angle_values` 写入 `data_info`。
- `timepix/losses.py`
  - 为 categorical task 禁用角度感知 loss / label encoding。
- `timepix/training/metrics.py`
  - 拆分 angle metrics 与 categorical metrics。
- `timepix/training/runner.py`
  - 根据 `label_type` 选择指标、训练日志字段、预测文件输出格式。
- `scripts/train.py`
  - 类别任务结束时打印 `balanced_accuracy` 与 `weighted-F1`，不打印角度 MAE/P90。
- `scripts/summarize.py` / `scripts/aggregate_seeds.py`
  - 汇总表增加 `label_type`、`class_names`、`balanced_accuracy`、`weighted_f1`。
- `timepix/config_validation.py`
  - 校验 `dataset.label_type`、可选 `dataset.class_names`，并阻止 categorical task 使用角度损失。

### C1：Particle source classification 单 seed 模态/融合基线

状态：已撰写实验配置，等待服务器运行。

目的：

- 在新的 `categorical_folder` 任务上确认训练框架可以正常处理非角度类别识别。
- 比较 `ToT`、`RToA`、输入层拼接、双分支高层拼接和 GMU 门控融合在当前 source-label 数据集上的基础表现。
- 单独保留 `RToA` 单模态实验，用来确认 ToA 时间结构本身是否具有 source / particle 判别能力。

命名决策：

- Particle/source 分类实验使用 `C` 系列编号，避免与 Alpha 角度实验 `A` 系列和 Proton/C 角度实验 `B` 系列混淆。
- C1 是 single-seed screening，不作为最终模型结论；后续若需要正式结论，再根据 C1 validation 结果选择 1-2 个设置做多 seed。

固定设置：

| 项目 | 设置 |
| --- | --- |
| Dataset | `Particle_Source_3` |
| 本地数据路径 | `E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1\dataset` |
| 服务器数据路径 | `/root/autodl-tmp/particle_source_label_cleaned_tot_toa_v1/dataset` |
| Label type | `categorical_folder` |
| 当前类别 | `Am`, `Co60`, `Sr`，从文件夹自动提取 |
| Split | stratified `0.8/0.1/0.1` |
| Shared split | `outputs/splits/Particle_Source_3_ToT-ToA_seed42_0.8_0.1_0.1.json` |
| Training seed | `42` |
| Loss | `cross_entropy + onehot` |
| Primary metric | `val_macro_f1` |
| 关键报告指标 | `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`, per-class F1, confusion matrix |
| 不使用指标 | angle `MAE`, `P90`, high-angle F1, adjacent-angle confusion |

训练设置：

```yaml
epochs: 15
batch_size: 64
learning_rate: 3e-4
weight_decay: 1e-4
scheduler: cosine
eta_min: 1e-7
early_stopping_patience: 5
mixed_precision: true
num_workers: 4
```

该训练设置是 C1 screening 默认值。由于当前数据有 `112750` 个配对样本，C1 先关注模态与架构排序，不在第一轮做训练超参数搜索。

C1 实验矩阵：

| 编号 | 配置文件 | 输入 | 模型 | 目的 |
| --- | --- | --- | --- | --- |
| C1a | `configs/experiments/c1a_particle_source_tot_seed42.yaml` | `ToT` | `resnet18_no_maxpool` | ToT source 分类基线 |
| C1b | `configs/experiments/c1b_particle_source_rtoa_seed42.yaml` | `RToA` | `resnet18_no_maxpool` | 单独确认 ToA/relative-time 表达能力 |
| C1c | `configs/experiments/c1c_particle_source_tot_rtoa_input_concat_seed42.yaml` | `[ToT, RToA]` | `resnet18_no_maxpool` | 输入层 concat baseline |
| C1d | `configs/experiments/c1d_particle_source_tot_rtoa_dual_concat_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_concat_aux` | 双分支高层 concat |
| C1e | `configs/experiments/c1e_particle_source_tot_rtoa_gmu_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_gmu_aux` | 双分支 GMU 门控 |

ToA 表达决策：

- C1b/C1c/C1d/C1e 均使用 `data.toa_transform: relative_minmax`，记为 `RToA`。
- 第一轮不扩展 raw ToA、`relative_centered`、`relative_rank` 或 mask 网格。

类别不均衡注意：

- 当前 `Co60` 样本远多于 `Am` 和 `Sr`，因此 C1 不能只看 overall accuracy。
- C1 按 `val_macro_f1` 保存 best checkpoint；`balanced_accuracy` 和每类 F1 是必须报告的辅助指标。
- C1 暂不启用 class-weighted loss 或 weighted sampler；如果 C1 显示 minority class 表现不足，后续 C2 再专门比较类别不均衡策略。

服务器运行命令：

```bash
tmux new -s c1_particle
cd /root/Timepix
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_source_label_cleaned_tot_toa_v1/dataset

$PY scripts/train.py --config configs/experiments/c1a_particle_source_tot_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1b_particle_source_rtoa_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1c_particle_source_tot_rtoa_input_concat_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1d_particle_source_tot_rtoa_dual_concat_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1e_particle_source_tot_rtoa_gmu_seed42.yaml --data-root $DATA && \
$PY scripts/summarize.py --group c1_particle_source_baseline_seed42 --out outputs/c1_particle_source_baseline_seed42_runs.csv
```

本地结果拉取命令：

```powershell
rclone copy autodl37655:/root/Timepix/outputs/ D:/Project/Timepix/outputs/ `
  --progress `
  --transfers 8 `
  --checkers 16 `
  --create-empty-src-dirs `
  --exclude "**/best_model.pth" `
  --exclude "**/last_checkpoint.pth" `
  --exclude "**/*.pt" `
  --log-file D:/Project/Timepix/outputs/rclone_autodl37655_pull.log `
  --log-level INFO
```
