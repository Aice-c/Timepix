# 代码工程层面说明

这份文档解释“代码如何实现物理任务”。它和 `PHYSICS_CONTEXT.md` 是配套关系：前者讲为什么，本文讲怎么做。

## 1. 当前代码主线

第一阶段重构后，新主流程位于：

```text
configs/*.yaml
  -> scripts/train.py 或 scripts/run_grid.py
  -> timepix/data
  -> timepix/models
  -> timepix/losses.py
  -> timepix/training
  -> outputs/experiments
```

旧 `Program/main.py` 和 `Program/Config.py` 暂时保留为 legacy 参考，不直接删除。

从工程角度看，一次新实验可以概括为：

```text
读取 YAML 配置
  -> 校验配置字段、模型名、损失函数、模态约束和 split 比例
  -> 校验数据集支持的模态
  -> 收集数据文件并配对已启用模态
  -> 划分 train/val/test，并保存/复用 split manifest
  -> 计算标准化统计量
  -> 构建 Dataset 和 DataLoader
  -> 构建模型
  -> 构建损失函数
  -> 按配置启用可选 CUDA AMP 混合精度
  -> 训练与验证
  -> 每个 epoch 保存 last checkpoint
  -> 用最佳模型测试
  -> 保存模型、日志、预测、指标和 metadata
```

## 2. 配置入口

新系统使用 YAML 配置，不再依赖旧的全局 `Config.py` class。

主要配置目录：

- `configs/datasets/`：描述数据集事实，例如 `Alpha_100` / `Alpha_50` 有 ToT/ToA，`Proton_C` 只有 ToT。
- `configs/experiments/`：描述一次实验怎么跑。
- `configs/search/`：描述 Optuna/TPE 超参数搜索空间。

示例命令：

```powershell
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml
```

服务器路径不同，可以覆盖：

```powershell
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml --data-root /root/autodl-tmp/Alpha_100
```

也可以用环境变量 `TIMEPIX_DATA_ROOT`，这样同一份配置可以在笔记本和服务器复用。

## 3. 数据集模态约束

- `Alpha_100` 和 `Alpha_50` 数据集可以使用 `['ToT']`、`['ToA']` 或 `['ToT', 'ToA']`；当前正式实验配置统一选择 `Alpha_100`。
- `Proton_C` 数据集只有 ToT，应使用 `['ToT']`。
- 如果 `Proton_C` 数据误配为 `['ToT', 'ToA']`，新系统会在训练开始前报错。

## 4. 数据加载

新核心文件在 `timepix/data/`。

主要职责：

- 检查数据目录是否存在。
- 找到所有数字角度文件夹。
- 把角度文件夹排序后映射为连续类别标签。
- 根据启用模态配对同一个事件的文件；alpha 可配对 ToT/ToA，C/质子通常只有 ToT。
- 按类别分层划分训练、验证、测试集。
- 保存或复用 split manifest，保证不同实验划分一致；`split.seed` 控制划分，`training.seed` 控制训练随机性。
- 读取 `.txt` 矩阵为 numpy 数组。
- 按 `data.dtype` 控制读取精度，默认 `float32`。
- 可选按 `data.toa_transform` 对 ToA 做样本内相对时间变换。
- 可选按 `data.add_hit_mask` 在图像末尾追加命中掩码通道。
- 转成 PyTorch tensor。
- 可选中心裁剪。
- 可选 90 度旋转增强。
- 可选按模态标准化。
- 可选提取并标准化手工特征。

Dataset 返回格式有两种：

```python
(sample_tensor, label)
```

或启用手工特征时：

```python
(sample_tensor, label, handcrafted_features)
```

## 5. 模型层

新模型文件位于 `timepix/models/`。

第一阶段已实现：

- `resnet18`
- `resnet18_no_maxpool`
- `resnet18_maxpool`
- `resnet18_original`
- `shallow_resnet`
- `shallow_cnn`
- `densenet121`
- `efficientnet_b0`
- `convnext_tiny`
- `vit_tiny`

新模型工厂在 `timepix/models/registry.py`：

```python
model = build_model(cfg, input_channels, num_classes, task, handcrafted_dim)
```

所有新模型不再读取全局 `Config.py`，而是从当前实验配置接收：

- 输入通道数。
- 类别数。
- 任务类型。
- 手工特征维度。
- 融合方式。
- ResNet18 conv1 结构参数：`conv1_kernel_size`、`conv1_stride`、`conv1_padding`。
- 分类头 dropout。

`resnet18_original` 独立文件实现，固定原始 ResNet18 stem：
`7x7/stride=2/padding=3` + 第一层 maxpool。它仍然走统一的 `FeatureFusion`
和任务 head，因此手工特征、分类/回归任务、损失函数和标签编码切换都保持兼容。

DenseNet121、EfficientNet-B0、ConvNeXt-Tiny 和 ViT-Tiny 在
`timepix/models/torchvision_backbones.py` 中适配。它们会把输入 stem 改成
当前模态通道数，输出投影到 `feature_dim`，再复用统一 `FeatureFusion` 和
task head。`vit_tiny` 是面向原生 Timepix 矩阵的本地小型 ViT，当前
`Alpha_100` 主线使用 `image_size=100`、`patch_size=10`，不提供预训练权重。

## 6. 多模态实现方式

多模态不是两个模型分支，而是通道拼接。

如果 alpha 数据集配置为：

```python
modalities = ['ToT', 'ToA']
```

则 Dataset 会分别读取 ToT 和 ToA 矩阵，并沿 channel 维度拼接为：

```text
sample_tensor.shape = [2, H, W]
```

如果同时设置：

```yaml
data:
  add_hit_mask: true
```

则输入会追加命中掩码通道：

```text
sample_tensor.shape = [3, H, W]
```

runner 不再用 `len(dataset.modalities)` 推断模型输入通道，而是使用 dataloader 写入的 `data_info.input_channels`。

如果使用 C/质子数据集，应配置为：

```python
modalities = ['ToT']
```

此时输入为单通道：

```text
sample_tensor.shape = [1, H, W]
```

## 7. 手工特征和融合方式

当前支持的手工特征：

- `total_energy`

新系统支持三种融合方式：

```text
none    不使用手工特征
concat  CNN feature + handcrafted feature 后直接分类
gated   拼接后做 feature-wise gate，再分类
```

旧代码里的“注意力机制”在新系统中命名为 `gated`，更准确地说是门控式特征重标定，不是 Transformer attention。

## 8. 损失函数与指标

新核心文件：

- `timepix/losses.py`
- `timepix/training/metrics.py`

支持：

- `cross_entropy`：标准分类损失。
- `emd`：利用角度类别有序性的 Earth Mover's Distance 风格损失。
- `mse` / `smooth_l1`：回归任务预留。

记录指标：

- accuracy。
- angle MAE by argmax。
- angle MAE by probability-weighted angle。
- P90 Error：角度绝对误差的 90 分位数，分类任务默认基于 argmax 预测角度。
- macro-F1。
- per-class precision/recall/F1。
- confusion matrix。

主优化指标暂时以 validation accuracy 为主。

训练过程支持：

- tqdm batch 进度条。
- 每个 epoch 结束打印训练摘要。
- `last_checkpoint.pth` 每个 epoch 原子更新，并包含当前最佳模型权重。
- `best_model.pth` 在验证集刷新最佳结果时同步保存。
- `--resume` 从 checkpoint 恢复训练；新 checkpoint 可直接恢复，不必重复传 `--config`。
- `training.mixed_precision` 开关 CUDA AMP；FP16 训练使用 GradScaler，checkpoint 会保存 scaler 状态。
- `training_log.csv` 记录每个 epoch 的耗时，`metadata.json` 记录 fit/test/total 秒数。

## 9. 实验脚本

新实验入口：

- `scripts/train.py`：跑单个实验。
- `scripts/run_grid.py`：跑一组网格对比实验。
- `scripts/search_hparams.py`：在代表性实验设置上做 Optuna/TPE 训练超参数搜索。
- `scripts/summarize.py`：汇总全部实验或某个实验组。
- `scripts/aggregate_seeds.py`：把多 seed summary 聚合成 mean/std。

实验可以通过 `experiment_group` 分组保存，例如：

```yaml
experiment_name: alpha_resnet18_tot
experiment_group: baseline
```

输出目录会变成：

```text
outputs/experiments/baseline/<timestamp>_alpha_resnet18_tot/
```

`metadata.json` 中也会记录 `experiment_group`，便于后续汇总和论文筛选。

汇总全部实验：

```powershell
python scripts/summarize.py --all
```

无参数运行 `python scripts/summarize.py` 也会汇总全部实验组。

汇总 CSV 会包含结构超参数列：`conv1_kernel_size`、`conv1_stride`、`conv1_padding`、`dropout`、`feature_dim`、`hidden_dim`、`image_size`、`patch_size`，也会包含 `input_channels`、`toa_transform`、`add_hit_mask`、`seed`、`split_seed`、`split_manifest_hash`、早停状态、训练超参数、混合精度状态、训练耗时和 git commit。

训练超参数搜索入口，A2 使用：

```powershell
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml --dry-run
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml
```

搜索配置继承一个普通实验配置，`search.parameters` 用 dotted key 指向要采样的字段，例如 `training.learning_rate`、`training.weight_decay`、`training.batch_size`、`training.eta_min` 和 `model.dropout`。每个 trial 都调用同一个 `run_experiment`，因此 checkpoint、AMP、早停、metadata 和汇总字段都保持一致。搜索目标应使用 validation 指标，test 指标只用于最终报告。

多 seed 认证入口，固定 `split.seed`，只切换 `training.seed`：

```powershell
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml
python scripts/summarize.py --group a2_best_3seed --out outputs/a2_best_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a2_best_3seed_runs.csv --out outputs/a2_best_3seed_mean_std.csv
```

主干模型对比入口：

```powershell
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml
```

该配置继承 `configs/experiments/alpha_tot_a2_best_base.yaml`，固定 `Alpha_100`、ToT、CE、one-hot、无手工特征、A2 best 训练超参，并显式复用 `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json`，只切换 `model.name` 与 `training.seed=[42,43,44]`，对比 `shallow_cnn`、`shallow_resnet`、`resnet18_no_maxpool`、`densenet121`、`efficientnet_b0`、`convnext_tiny` 和 `vit_tiny`。A3 的 ViT-Tiny 使用 `image_size=100`、`patch_size=10`。`a3_backbone_comparison_seed42.yaml` 继承完整 A3，只保留 `training.seed=42`，用于时间紧张时先跑 7 个主干的单 seed 结果。

A4 模态对比入口：

```powershell
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml
```

该配置继承 A2 best base，固定 `resnet18_no_maxpool` 和训练超参，只切换 `dataset.modalities` 与 `training.seed=[42,43,44]`。为了公平比较 ToT、ToA 和 ToT+ToA，它显式使用 paired split manifest `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`。该文件应由历史 `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json` 复制得到，因为 `Alpha_100` 的 ToT/ToA 文件完全一一对应，split key 又已去掉模态标记。`a4_modality_comparison_seed42.yaml` 继承完整 A4，只保留 `training.seed=42`，用于时间紧张时先跑 3 个模态的单 seed 结果。

A4b ToA 表达方式对比入口：

```powershell
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml
```

第一阶段 A4b 固定 A4 的模型、split 和 A2 best 训练超参，只切换 `data.toa_transform` 与 `data.add_hit_mask`。`relative_*` 变换与 normalizer 共用同一套 ToA transform helper，确保归一化统计和实际输入一致。

A4b late logit fusion 评估入口：

```powershell
python scripts/evaluate_logit_fusion.py --group a4_modality_comparison_seed42
```

脚本会扫描实验组，自动匹配同一 `training.seed` 下的 `[ToT]` 和 `[ToA]` 单模态 run，加载各自 `best_model.pth`，在 validation set 上搜索 `alpha_toa`，再用选定的 alpha 报告 test 指标。它也支持通过 `--tot-run` 和 `--toa-run` 手动传入两个 run 目录。

A4b 预测互补性诊断入口：

```powershell
python scripts/analyze_prediction_complementarity.py --seed 42
```

该脚本只读取已有 `predictions.csv`，不训练、不加载 checkpoint。默认匹配 A4 seed42 的 ToT/ToA 单模态结果，以及 A4b-1 seed42 的 relative ToT+ToA 候选，输出 oracle accuracy、oracle MAE、ToT 错误时其他预测是否正确/误差更小，以及每个类别的 correct-overlap。

汇总某一组：

```powershell
python scripts/summarize.py --group baseline
```

按路径汇总：

```powershell
python scripts/summarize.py --root outputs/experiments/baseline --out outputs/baseline_summary.csv
```

旧的 `Program/run_ablation.py` 目前只是临时消融实验脚本，主要用于快速对比不同损失函数、软标签和回归任务的效果。它不应被理解为论文最终的消融实验框架。

`Program/sweep.py` 不建议继续补丁式维护，后续应基于新实验系统重写或替代。

## 10. 预处理代码

预处理和探索主要在 `ProcessProgram/`：

- `ProcessProgram/A/`：alpha 数据相关 Notebook 和合并脚本；alpha 有 ToT/ToA 双模态。
- `ProcessProgram/C/`：C/质子数据相关 Notebook 和脚本；当前应按 ToT 单模态理解。

这些文件多为实验性 Notebook，很多路径是硬编码的。后续如果要复现实验，应优先把稳定逻辑提取成可配置 CLI 脚本。

## 11. 下一步最值得继续做的事

1. 在服务器上用 `--set training.epochs=2` 跑通一个最小实验。
2. 根据服务器反馈修正数据路径、batch size、num_workers。
3. 用 `configs/experiments/compare_mixed_precision.yaml` 对比 FP32 与 AMP，如果指标损失可接受，再把正式实验切到 `training.mixed_precision: true`。
4. 当前 A2 best base 已沉淀，后续优先用 `configs/experiments/a3_backbone_comparison.yaml` 和 `configs/experiments/a4_modality_comparison.yaml` 在 `Alpha_100` 上做三 seed 对比；如以后单独重启 `Alpha_50` 对照，再另建独立配置。
5. 如果 txt 读取明显拖慢训练，增加 `.npy` 缓存或离线转换流程。
6. 逐步迁移/适配旧模型。
7. 把旧图表生成脚本改成读取新 `metadata.json` 和 `experiment_summary.csv`。
8. 根据论文需要补充更多 grid 配置。
