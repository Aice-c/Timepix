# Timepix Configs

这个目录放新实验系统的配置文件。

人工维护的实验日志见：

```text
agent/EXPERIMENT_LOG.md
```

## 目录

```text
configs/datasets/      数据集事实：粒子类型、路径、可用模态
configs/experiments/   具体实验：模型、损失、模态、训练参数
configs/search/        Optuna/TPE 超参数搜索配置
```

## 路径规则

数据集路径不要写死在代码里。推荐两种方式：

1. 使用环境变量：

```bash
export TIMEPIX_DATA_ROOT=/root/autodl-tmp
```

2. 运行时覆盖：

```bash
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml --data-root /root/autodl-tmp/Alpha_100
```

## 重要模态约束

- 当前正式 Alpha 主线使用 `Alpha_100`，配置文件为 `configs/datasets/alpha_100.yaml`，输入尺寸为 `100x100`。
- `Alpha_50` 保留为对照/历史数据集配置，不用于当前正式 A3/A4 后续实验主线。
- `Alpha_100` 和 `Alpha_50` 均支持 `ToT` 和 `ToA`。
- `Proton_C` 数据集只支持 `ToT`。
- 常见配置字段会在训练或 grid dry-run 前校验，拼错字段会直接报错。

## 实验组

实验配置可以设置：

```yaml
experiment_group: baseline
```

输出会保存到：

```text
outputs/experiments/baseline/<timestamp>_<experiment_name>/
```

如果不设置，默认使用 `default` 组。

汇总某个实验组：

```bash
python scripts/summarize.py --group baseline
```

汇总全部实验组：

```bash
python scripts/summarize.py --all
```

汇总 CSV 会包含模型结构超参数列，例如 `conv1_kernel_size`、`conv1_stride`、`conv1_padding`、`dropout`、`feature_dim`、`hidden_dim`、`image_size` 和 `patch_size`，也会记录 `seed`、`split_seed`、`split_manifest_hash`、`mixed_precision` / `mixed_precision_enabled` 与 `fit_seconds`，方便直接筛选 A1、AMP、主干模型或多 seed 对比结果。

长网格实验可以使用：

```bash
python scripts/run_grid.py \
  --config configs/experiments/a1_structure_adaptation.yaml \
  --skip-existing \
  --continue-on-error
```

非 dry-run 网格会写入 `outputs/grid_manifests/`，记录每个组合的 `planned/running/done/failed/skipped_existing` 状态。

## 训练超参数搜索

代表性 Alpha ToT ResNet18 设置的 A2 搜索配置：

```bash
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml --dry-run
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml
```

该配置使用 Optuna TPE，在固定 dataset、modality、model、loss、label 和数据划分 seed 的条件下搜索训练超参数：

```yaml
search:
  objective: validation.accuracy
  parameters:
    training.learning_rate: ...
    training.weight_decay: ...
    training.batch_size: ...
    training.eta_min: ...
    model.dropout: ...
```

搜索目标只使用 validation 指标；test 指标保留在每个 trial 的输出中用于最终报告，不用于挑选超参数。搜索结果会写入 `outputs/hparam_search/`，并生成 `best_config.yaml`、`best_params.json`、`study_summary.json` 和 `trials.csv`。Optuna study 默认持久化到 `outputs/optuna/`，中断后可用同一个配置继续运行。

A2 当前最佳超参已经整理为后续实验可继承的 base：

```yaml
base: configs/experiments/alpha_tot_a2_best_base.yaml
```

该 base 固定 `Alpha_100`、ToT、CE、one-hot、无手工特征、`resnet18_no_maxpool`、`split.seed: 42`、`training.seed: 42`、AMP，以及 A2 best 训练超参：`learning_rate=4.3878e-05`、`weight_decay=4.7324e-04`、`batch_size=32`、`eta_min=1.6433e-07`、`dropout=0.1`、`scheduler=cosine`、`epochs=25`。A2 best 来自 `Alpha_100 + ToT` 历史实验，因此 base 显式复用恢复出的历史 split：

```yaml
split:
  path: outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json
```

曾短暂尝试过 `Alpha_50`，但效果和实验故事线不如 `Alpha_100` 连贯；后续正式实验配置统一回到 `Alpha_100`。

## 多 seed 认证

`split.seed` 控制 train/val/test 的分层划分；`training.seed` 控制模型初始化、DataLoader shuffle 和训练随机性。旧配置如果没有写 `split.seed`，会继续沿用 `training.seed` 生成划分；新对比实验建议显式固定：

```yaml
split:
  seed: 42
  reuse_split: true

grid:
  training.seed: [42, 43, 44]
```

A2 最优训练超参的 3 seed 认证配置：

```bash
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml
```

跑完后先汇总该实验组，再计算平均值和标准差：

```bash
python scripts/summarize.py --group a2_best_3seed --out outputs/a2_best_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a2_best_3seed_runs.csv --out outputs/a2_best_3seed_mean_std.csv
```

## 主干模型对比

正式 A3 主干对比使用 `configs/experiments/a3_backbone_comparison.yaml`。它继承 A2 best base，对 7 个模型主干进行三 seed 验证：

```text
shallow_cnn
shallow_resnet
resnet18_no_maxpool
densenet121
efficientnet_b0
convnext_tiny
vit_tiny
```

该配置固定 `Alpha_100`、ToT、CE、one-hot、无手工特征、A2 best 训练超参和恢复出的 `Alpha_100_ToT` 历史 split，只切换 `model.name` 和 `training.seed: [42, 43, 44]`。`vit_tiny` 使用原生 `100x100` 输入，A3 配置为 `image_size: 100`、`patch_size: 10`，保持 `10x10=100` 个 patch token。`model.dropout=0.1` 指统一 Timepix task head dropout；torchvision backbone 内部正则保持模型默认，不在 A3 中单独调参。

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml
```

跑完后建议计算三 seed 均值和标准差：

```bash
python scripts/summarize.py --group a3_backbone_comparison --out outputs/a3_backbone_comparison_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a3_backbone_comparison_runs.csv --out outputs/a3_backbone_comparison_mean_std.csv
```

## A4 模态对比

`configs/experiments/a4_modality_comparison.yaml` 用于比较 `Alpha_100` 数据集的 ToT、ToA 和 ToT+ToA。它继承 A2 best base，固定 `resnet18_no_maxpool`、CE、one-hot、无手工特征、`fusion_mode: none` 和 A2 best 训练超参，只切换 `dataset.modalities` 和 `training.seed: [42, 43, 44]`。

A4 使用同一个 paired split manifest：

```yaml
split:
  path: outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

`Alpha_100` 中 ToT 与 ToA 文件完全一一对应，split manifest 保存的是去掉 ToT/ToA 标记后的归一化 sample key。因此 A4 的 paired split 不重新随机生成，而是由历史 ToT split 复制得到：

```bash
cp outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json \
   outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

这样 A4 的 ToT、ToA、ToT+ToA 三组实验与 A1/A2/A3 的历史数据划分严格一致，同时文件名保留 `ToT-ToA` 以标识双模态用途。

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml
```

跑完后建议计算三 seed 均值和标准差：

```bash
python scripts/summarize.py --group a4_modality_comparison --out outputs/a4_modality_comparison_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4_modality_comparison_runs.csv --out outputs/a4_modality_comparison_mean_std.csv
```

## 混合精度训练

训练配置中可以显式开关 CUDA AMP：

```yaml
training:
  mixed_precision: true
  mixed_precision_dtype: float16
```

`mixed_precision: false` 是默认安全设置；开启后训练、验证和测试都会使用 autocast，FP16 训练会使用 GradScaler。checkpoint 会保存 scaler 状态，`--resume` 可以继续恢复。汇总表中的 `fit_seconds`、`test_seconds` 和 `total_seconds` 可用于比较速度。

在 A1 当前最佳结构上对比 FP32 与 AMP：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml
```

该配置固定 `resnet18_no_maxpool`、`conv1_kernel_size: 2`、`conv1_stride: 1`、`dropout: 0.3`，只切换 `training.mixed_precision`。

## ResNet18 结构参数

新系统中 `resnet18` 默认是不使用第一层 maxpool 的 Timepix 适配版，也可以显式写成：

```yaml
model:
  name: resnet18_no_maxpool
```

保留第一层 maxpool 的变体使用：

```yaml
model:
  name: resnet18_maxpool
```

原始 ResNet18 stem baseline 使用：

```yaml
model:
  name: resnet18_original
```

`resnet18_original` 固定使用 torchvision ResNet18 的原始 stem：`conv1` 为 `7x7/stride=2/padding=3`，并保留第一层 maxpool。它只适配输入通道数，以便接收 ToT 或 ToT+ToA 数据；该 baseline 不参与 A1 网格搜索。

`resnet18_no_maxpool` 和 `resnet18_maxpool` 都支持：

```yaml
model:
  conv1_kernel_size: 2
  conv1_stride: 1
  conv1_padding: 0
  dropout: 0.1
```

旧字段 `model.kernel_size` 仍然兼容，但新实验建议统一使用 `conv1_kernel_size`。

## A1 结构适配实验

A1 固定 alpha、ToT、CE、one-hot、无手工特征和固定 seed。先跑原始 ResNet18 baseline：

```bash
python scripts/train.py --config configs/experiments/a1_resnet18_original_baseline.yaml
```

然后跑结构适配网格，对比 ResNet18 是否保留第一层 maxpool，以及 `conv1_kernel_size`、`conv1_stride`、`dropout` 组合：

```bash
python scripts/run_grid.py --config configs/experiments/a1_structure_adaptation.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a1_structure_adaptation.yaml
```

网格配置会展开 36 个实验，baseline 和网格都会输出到：

```text
outputs/experiments/a1_structure_adaptation/
```

## 训练进度与恢复

推荐开启：

```yaml
training:
  progress_bar: true
  save_last_checkpoint: true
```

如果训练中断，可以恢复：

```bash
python scripts/train.py \
  --resume outputs/experiments/baseline/<experiment_dir>/last_checkpoint.pth
```

新的 checkpoint 中保存了配置。旧 checkpoint 或数据路径变动时，可以额外传入 `--config` 和 `--data-root`。
