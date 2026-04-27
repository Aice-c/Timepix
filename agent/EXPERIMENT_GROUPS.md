# 实验组功能说明

新自动化实验系统支持 `experiment_group`，用于把不同研究问题的实验分开保存和汇总。

## 为什么需要实验组

后续可能会有很多实验：

- 模型架构对比。
- 损失函数对比。
- 手工特征消融。
- 模态对比。
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
```

因此可以在 Excel 或 pandas 中按实验组筛选。

## 按路径汇总

仍然支持原来的方式：

```bash
python scripts/summarize.py \
  --root outputs/experiments/baseline \
  --out outputs/baseline_summary.csv
```

这种方式适合临时汇总某个自定义目录。

