# 代码工程上下文

本文档说明当前 Timepix 新实验系统的工程结构、运行链路和主要扩展点。它面向后续代码修改者，重点回答“实验配置如何进入训练流程”和“各模块负责什么”。旧版 `agent/CODE_CONTEXT.md` 因编码损坏已归档为 `agent/CODE_CONTEXT.old.md`。

## 1. 当前主链路

新的训练与评估主链路为：

```text
configs/*.yaml
  -> scripts/train.py 或 scripts/run_grid.py
  -> timepix/data
  -> timepix/models
  -> timepix/losses.py
  -> timepix/training
  -> outputs/experiments
```

legacy `Program/` 保留为历史参考，不再作为新实验入口。新的 A/B 系列实验应优先通过 YAML 配置和 CLI 参数完成，避免为了单个实验直接修改 Python 逻辑。

## 2. 配置层

主要配置目录：

- `configs/datasets/`：描述数据集事实，例如 `Alpha_100` 支持 `ToT`/`ToA`，`Proton_C_7` 只支持 `ToT`，`Particle_Source_3` 支持 `ToT`/`ToA` 和普通类别文件夹标签。
- `configs/experiments/`：描述单次训练、网格对比、三 seed 验证或诊断实验。
- `configs/search/`：描述 Optuna/TPE 超参数搜索空间。

当前命名原则：

- Alpha 正式主线使用 `configs/datasets/alpha_100.yaml`。
- `configs/datasets/alpha_50.yaml` 只作为历史/对照入口。
- Proton/C 训练使用 `configs/datasets/proton_c_7.yaml`。
- `configs/datasets/proton_c.yaml` 是兼容入口，训练配置不应优先使用它。
- Particle/source 分类使用 `configs/datasets/particle_source_3.yaml`；类别名从数据集顶层文件夹自动提取，后续可从 `Am`/`Co60`/`Sr` 平滑切换到 `Alpha`/`Beta`/`Gamma` 等新类别名。

更完整的配置与命令索引见 `configs/README.md`。

## 3. 数据层

核心模块位于 `timepix/data/`。其主要职责包括：

- 扫描标签目录并建立 label map。`dataset.label_type: angle_folder` 要求文件夹名为数值角度；`dataset.label_type: categorical_folder` 将普通文件夹名作为类别名。
- 按启用模态收集和配对样本。
- 生成或复用 train/validation/test split manifest。
- 读取 `.txt` 矩阵并转换为 tensor。
- 执行 ToA 表达变换，例如 `relative_minmax`、`relative_centered`、`relative_rank`。
- 可选追加 `hit_mask` 图像通道。
- 计算训练集归一化统计量。
- 提取训练用手工特征，并基于 train split 拟合 feature scaler。

训练手工特征实现位于 `timepix/data/features.py`，故意与 `timepix/analysis/` 中的论文数据分析特征实现分离。A5 明确不复用数据分析链路中的特征，以避免分析特征和训练特征互相污染。

## 4. 模型层

核心模块位于 `timepix/models/`。当前模型族包括：

- ResNet18 系列：`resnet18_no_maxpool`、`resnet18_maxpool`、`resnet18_original`。
- 浅层 CNN：`shallow_cnn`、`shallow_resnet`。
- torchvision 主干：`densenet121`、`efficientnet_b0`、`convnext_tiny`。
- 本地 ViT：`vit_tiny`。
- A4c 多模态模型：`dual_stream_concat_aux`、`dual_stream_gmu_aux`、`toa_conditioned_film`、`warm_started_expert_gate`。
- A5 手工特征诊断模型：`handcrafted_mlp`。

统一模型工厂从配置接收：

```text
input_channels
num_classes
task type
handcrafted_dim
fusion_mode
conv1 kernel/stride/padding
dropout
feature_dim
```

论文主推的端到端多模态架构为 A4c-2 `dual_stream_gmu_aux`。它的选择依据是 validation 侧 Macro-F1 接近最优、Val MAE 更好、稳定性和物理解释性更强，而不是 test 结果反选。

## 5. 损失与指标

核心模块：

- `timepix/losses.py`
- `timepix/training/metrics.py`

当前支持的主要损失包括：

- `cross_entropy`
- `cross_entropy` with optional `loss.class_weight: balanced` or explicit class-weight list
- Gaussian soft-target CE
- `ce_expected_mae`
- `ce_emd` / angle-weighted CDF-style ordered loss
- legacy pure `emd`
- regression losses such as `mse` / `smooth_l1`

B3 和 A6 的设计原则是保留 CE 作为主监督，再加入轻量有序角度辅助项。对于已经具有较高 exact accuracy 的 `Proton_C_7`，不采用 pure EMD 作为主线，因为它可能使输出分布过宽并损害精确分类。

`categorical_folder` 任务不允许使用角度有序损失或 regression。当前只应使用无序分类监督，例如 `cross_entropy + onehot`。

主要记录指标：

- accuracy
- argmax angle MAE
- probability-weighted angle MAE
- P90 error
- macro-F1
- per-class precision / recall / F1
- confusion matrix
- ordered-loss 诊断中的 adjacent / far error rate

对于 `categorical_folder` 任务，训练与汇总只记录普通分类指标：

- accuracy
- balanced accuracy
- macro-F1
- weighted-F1
- per-class precision / recall / F1 / support
- confusion matrix

这类任务不记录角度 MAE、P90、high-angle F1 或相邻角度混淆。

## 6. 训练层

核心模块位于 `timepix/training/`。训练流程支持：

- tqdm batch progress bar。
- 每个 epoch 输出 train/validation 摘要。
- `last_checkpoint.pth` 每个 epoch 保存。
- `best_model.pth` 按 validation primary metric 更新。
- `--resume` 从 checkpoint 恢复训练。
- CUDA AMP mixed precision。
- early stopping。
- `training_log.csv`、`metrics.json`、`metadata.json`、`predictions.csv`、`confusion_matrix.csv` 输出。

服务器长期训练应使用 `tmux`。断点恢复和持久化训练说明见 `agent/SERVER_TRAINING.md`。

当实验执行或结果分析交给 subagent 时，主控 agent 仍负责所有代码、配置、文档和实验决策。subagent 只执行或分析，不在服务器直接改代码或调整实验。具体角色边界、监督命令、结果拉取和反馈格式见 `agent/SUBAGENT_WORKFLOW.md`。

## 7. 脚本入口

标准训练入口：

- `scripts/train.py`：运行单个实验。
- `scripts/run_grid.py`：运行 YAML grid 实验。
- `scripts/search_hparams.py`：运行 Optuna/TPE 搜索。
- `scripts/summarize.py`：汇总训练输出。
- `scripts/aggregate_seeds.py`：聚合三 seed mean/std。

A4b 冻结 checkpoint 诊断入口：

- `scripts/analyze_prediction_complementarity.py`
- `scripts/evaluate_oracle_complementarity.py`
- `scripts/evaluate_selector_fusion.py`
- `scripts/analyze_selector_switches.py`
- `scripts/evaluate_gated_late_fusion.py`
- `scripts/evaluate_residual_gated_fusion.py`
- `scripts/aggregate_selector_fusion.py`

数据分析入口：

- `scripts/analyze_datasets.py`
- `scripts/analyze_resolution_limit.py`
- `scripts/make_analysis_report.py`

A5a 特征筛选入口：

- `scripts/screen_handcrafted_features.py`

## 8. 数据路径语义

本地 Windows 路径：

```text
Alpha_100  -> D:\Project\Timepix\Data\Alpha_100
Proton_C   -> E:\C1Analysis\Proton_C
Proton_C_7 -> E:\C1Analysis\Proton_C_7
Particle_Source_3 -> E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1\dataset
```

训练脚本的 `--data-root` 指向具体数据集目录，例如：

```powershell
python scripts\run_grid.py --config configs\experiments\b3a_proton_c7_ordinal_loss_seed42.yaml --data-root E:\C1Analysis\Proton_C_7 --dry-run
```

数据分析脚本的 `--data-root` 指向数据集父目录，例如：

```powershell
python scripts\analyze_datasets.py --data-root E:\C1Analysis --datasets Proton_C
```

Linux 服务器命令默认使用：

```bash
/root/autodl-tmp/Alpha_100
/root/autodl-tmp/Proton_C_7
/root/autodl-tmp/particle_source_label_cleaned_tot_toa_v1/dataset
```

## 9. 当前实验状态对代码工作的影响

- A4 结构/多模态探索已经收束，不建议继续扩展 MMTM、复杂 attention 或更多 ToA transform 网格。
- A5 手工特征路线已经完成三 seed 验证，不建议扩展到 25 维大特征池或逐特征开关网格。
- B2 手工特征路线已经收口，不优先推进 B2c。
- B3 已完成，`CE+ExpectedMAE lambda=0.05` 是当前 Proton_C_7 推荐损失。
- A6b 已完成，Alpha-ToT 的 `CE+EMD lambda=0.02` 不稳定且弱于 A2 CE baseline，Alpha 后续保持 CE one-hot。
- A7 已完成，最终 Alpha 端到端多模态主模型保持 `dual_stream_gmu_aux + ToT/relative_minmax ToA + CE one-hot + no handcrafted`。
- Particle/source 分类 C1 single-seed screening 已完成：C1d dual-stream concat 按 `val_macro_f1` 最强，C1c input concat 是轻量备选，C1b `RToA` 单模态显著优于 C1a `ToT`，C1e GMU 本轮异常。
- Particle/source C2 weighted-CE 稳定性复跑已完成：balanced CE 能把 GMU 从 C1 的塌缩中救回，C2e 是 C2 内部最强；但 balanced 权重对 `Co60` 惩罚过强，整体未超过 C1c/C1d。后续 C3 若推进，应优先尝试更温和的类别权重或 focal-style 方案，不要直接叠加 weighted sampler。

新增代码时，应优先服务 Particle/source 分类的类别不均衡、模态融合和结果整理，而不是继续扩大已经收束的 A4/A5/A6/B2 搜索空间。
