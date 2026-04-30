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
| `Proton_C_7` | 质子/C 七分类训练数据，用于 B 系列训练实验 | Proton 后续训练默认使用 |

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
| A6 | Alpha 有序角度损失 | A6a 已完成，A6b 配置已撰写 | A6b 只验证 `CE + EMD λ=0.02`；CE baseline 复用 A2-best，不重跑 |
| B1 | Proton_C_7 训练超参数搜索 | 已完成 | B1-best 使用 `lr=3e-4`、`batch=128`、`wd=1e-4`、`patience=8` |
| B2 | Proton_C_7 手工特征验证 | 已完成 | 手工特征增益极小；gated 可抑制坏特征但不形成显著提升 |
| B3 | Proton_C_7 有序角度损失 | 已完成 | `CE + ExpectedMAE λ=0.05` 是 Proton_C_7 当前推荐 loss |
| B4 | Proton_C_7 最终模型确认 | 待定 | 将基于 B1-best + B3 best loss 形成最终配置 |

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

状态：A6a 已完成；A6b 配置已撰写，待运行；A6c 待 A6b 结果。

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

状态：配置已撰写，待运行。

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

#### A6c：迁移到 GMU 多模态架构

状态：待 A6b 结果。

计划：

- 若 A6b 证明 `CE+EMD lambda=0.02` 在 ToT baseline 上有稳定 validation/MAE/F1 收益，则再考虑迁移到 A4c-2 `dual_stream_gmu_aux`。
- 如果 A6b 仍然只是 tie-break 级别或不稳定收益，则 A6c 不优先推进，避免给 GMU 主线引入复杂 loss 变量。
- 不优先迁移到 A4b frozen expert 系统，因为那需要重新训练 primary/candidate expert 后再重新 gate，工程复杂度较高。

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

状态：不推进。

模板配置：

- `configs/experiments/b2_proton_c7_handcrafted_transfer_TEMPLATE.yaml`

决策：

- B2a/B2b 的提升幅度过小，且未明显改善高角度 F1。
- 不优先消耗算力做 B2c three-seed。
- B2 论文口径：Proton_C_7 的 ToT 图像形态已被 CNN 较充分利用，手工标量主要与 CNN 表征冗余；gated 的价值更多是稳定化，而非显著增益。

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

状态：待定。

计划：

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
- 数据分析链路只按 ToT 分析 Proton_C，不假设 Proton 有 ToA。
- Windows 本地可跳过 UMAP；Linux 服务器可安装 `requirements-analysis.txt` 后运行完整分析。
- 输出 CSV + Markdown 表格，图片保存 PNG/PDF。

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

1. 运行 A6b：`CE+EMD lambda=0.02` three-seed；不运行 ExpectedMAE。
2. A6b 完成后，对照 A2-best three-seed 判断收益是否稳定；若仍是弱 tie-break 收益，则不优先推进 A6c。
3. 另一个窗口整理最终 CSV 结果后，回填本文档中仍需更精确的数值表。
4. 论文写作时应将 A4c GMU 的选择依据写为 validation Macro-F1 接近最优、Val MAE 更好、结构解释更符合物理结论；不能用 test 结果反向选择 GMU。
5. 5.5 Pro 交接文档需要与本日志同步，尤其是 A4c 最终架构口径、A5 收口口径、B3 最终 loss 口径。
