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

- `configs/datasets/`：描述数据集事实，例如 `Alpha_100` 支持 `ToT`/`ToA`，`Proton_C_7` 只支持 `ToT`。
- `configs/experiments/`：描述单次训练、网格对比、三 seed 验证或诊断实验。
- `configs/search/`：描述 Optuna/TPE 超参数搜索空间。

当前命名原则：

- Alpha 正式主线使用 `configs/datasets/alpha_100.yaml`。
- `configs/datasets/alpha_50.yaml` 只作为历史/对照入口。
- Proton/C 训练使用 `configs/datasets/proton_c_7.yaml`。
- `configs/datasets/proton_c.yaml` 是兼容入口，训练配置不应优先使用它。

更完整的配置与命令索引见 `configs/README.md`。

## 3. 数据层

核心模块位于 `timepix/data/`。其主要职责包括：

- 扫描角度目录并建立 label map。
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
- Gaussian soft-target CE
- `ce_expected_mae`
- `ce_emd` / angle-weighted CDF-style ordered loss
- legacy pure `emd`
- regression losses such as `mse` / `smooth_l1`

B3 和 A6 的设计原则是保留 CE 作为主监督，再加入轻量有序角度辅助项。对于已经具有较高 exact accuracy 的 `Proton_C_7`，不采用 pure EMD 作为主线，因为它可能使输出分布过宽并损害精确分类。

主要记录指标：

- accuracy
- argmax angle MAE
- probability-weighted angle MAE
- P90 error
- macro-F1
- per-class precision / recall / F1
- confusion matrix
- ordered-loss 诊断中的 adjacent / far error rate

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
```

## 9. 当前实验状态对代码工作的影响

- A4 结构/多模态探索已经收束，不建议继续扩展 MMTM、复杂 attention 或更多 ToA transform 网格。
- A5 手工特征路线已经完成三 seed 验证，不建议扩展到 25 维大特征池或逐特征开关网格。
- B2 手工特征路线已经收口，不优先推进 B2c。
- B3 已完成，`CE+ExpectedMAE lambda=0.05` 是当前 Proton_C_7 推荐损失。
- A6a 正在运行；A6b 和 A6c 需要等待 A6a validation 结果后再写配置。

新增代码时，应优先服务 A6 后续和论文结果整理，而不是继续扩大已经收束的 A4/A5/B2 搜索空间。
