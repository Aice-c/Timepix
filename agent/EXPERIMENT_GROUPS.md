# 实验组功能说明

新自动化实验系统支持 `experiment_group`，用于把不同研究问题的实验分开保存和汇总。

## 为什么需要实验组

后续可能会有很多实验：

- A1 结构适配实验。
- 模型架构对比。
- 损失函数对比。
- 手工特征消融。
- 模态对比。
- FP32 与 CUDA AMP 混合精度对比。
- 训练超参数搜索 trial。
- alpha 与 C/质子数据集对比。

如果全部放在 `outputs/experiments/` 同一层，后续筛选、写论文、画表都会比较乱。实验组的作用就是把这些实验按研究问题分开。

## 配置方式

在实验 YAML 中写：

```yaml
experiment_name: alpha_resnet18_tot
experiment_group: baseline
```

如果没有写 `experiment_group`，系统默认使用：

```text
default
```

## 输出目录

有实验组时，输出目录是：

```text
outputs/experiments/<experiment_group>/<timestamp>_<experiment_name>/
```

例如：

```text
outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/
```

A1 原始 ResNet18 baseline 和 36 组结构适配网格都会保存到：

```text
outputs/experiments/a1_structure_adaptation/
```

模型主干对比会保存到：

```text
outputs/experiments/compare_models/
```

## metadata

每个实验的 `metadata.json` 会记录：

```json
{
  "experiment_group": "baseline"
}
```

这样即使移动实验目录，也能知道它属于哪一组。

## 汇总某一组

```bash
python scripts/summarize.py --group baseline
```

默认输出：

```text
outputs/baseline_summary.csv
```

## 汇总全部实验组

```bash
python scripts/summarize.py --all
```

默认输出：

```text
outputs/experiment_summary.csv
```

CSV 中会包含：

```text
experiment_group
conv1_kernel_size
conv1_stride
conv1_padding
dropout
early_stopped
git_commit
mixed_precision
mixed_precision_enabled
fit_seconds
```

因此可以在 Excel 或 pandas 中按实验组筛选。

网格实验还会生成 manifest CSV，记录每个组合的运行状态。配合 `--skip-existing` 和 `--continue-on-error`，服务器中断或单个组合失败后可以更容易续跑。

混合精度对比实验使用：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml
```

该组默认保存到：

```text
outputs/experiments/compare_mixed_precision/
```

训练超参数搜索使用：

```bash
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml
```

每个 trial 是一个普通实验，默认保存到：

```text
outputs/experiments/a2_hparam_search_training/
```

搜索本身的汇总文件保存到：

```text
outputs/hparam_search/
```

## 按路径汇总

仍然支持原来的方式：

```bash
python scripts/summarize.py \
  --root outputs/experiments/baseline \
  --out outputs/baseline_summary.csv
```

这种方式适合临时汇总某个自定义目录。
