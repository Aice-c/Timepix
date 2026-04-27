# Timepix Configs

这个目录放新实验系统的配置文件。

## 目录

```text
configs/datasets/      数据集事实：粒子类型、路径、可用模态
configs/experiments/   具体实验：模型、损失、模态、训练参数
```

## 路径规则

数据集路径不要写死在代码里。推荐两种方式：

1. 使用环境变量：

```bash
export TIMEPIX_DATA_ROOT=/root/autodl-tmp
```

2. 运行时覆盖：

```bash
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml --data-root /root/autodl-tmp/Alpha_Clean
```

## 重要模态约束

- alpha 数据集支持 `ToT` 和 `ToA`。
- C/质子数据集只支持 `ToT`。

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

汇总 CSV 会包含模型结构超参数列，例如 `conv1_kernel_size`、`conv1_stride`、`conv1_padding` 和 `dropout`，方便直接筛选 A1 结果。

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
