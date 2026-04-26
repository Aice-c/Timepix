# 新自动化实验系统使用说明

这份文档说明第一阶段重构后的新系统怎么用。核心原则是：尽量不通过改 Python 代码来做实验，而是通过 YAML 配置和命令行参数来控制实验。

## 1. 新系统包含什么

新增目录：

```text
timepix/              新核心代码包
configs/              数据集配置和实验配置
scripts/              命令行入口
outputs/experiments/  默认实验输出目录
```

旧目录：

```text
Program/
```

暂时保留，不删除。它是旧训练流程和旧模型的参考。后续会逐步迁移，不会一次性拆掉。

## 2. 你以后主要改什么

通常只需要改：

```text
configs/experiments/*.yaml
```

比如你要换模型、换损失函数、开关手工特征、选择 ToT/ToA，都应该优先改 YAML，而不是改 Python 源码。

## 3. 数据集配置

alpha 数据集：

```yaml
name: alpha_clean
particle: alpha
root: ${TIMEPIX_DATA_ROOT:-Data}/Alpha_Clean
available_modalities: [ToT, ToA]
default_modalities: [ToT, ToA]
```

C/质子数据集：

```yaml
name: proton_c_tot
particle: proton
root: ${TIMEPIX_DATA_ROOT:-Data}/Proton_C
available_modalities: [ToT]
default_modalities: [ToT]
```

这里最重要的是：

- alpha 支持 `ToT`、`ToA`、`ToT+ToA`。
- C/质子只支持 `ToT`。
- 如果 C/质子误写 `ToA`，新系统会在训练开始前报错。

## 4. 本地和服务器路径如何处理

不要把服务器路径写死进代码。

推荐方法一：命令行覆盖。

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_Clean
```

推荐方法二：环境变量。

本地 PowerShell：

```powershell
$env:TIMEPIX_DATA_ROOT="D:\Project\Timepix\Data"
```

服务器 bash：

```bash
export TIMEPIX_DATA_ROOT=/root/autodl-tmp
```

然后同一个配置文件都可以使用：

```bash
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml
```

只要数据目录内部结构一致，路径不同不会影响实验。

## 5. 跑单个实验

alpha ToT 单模态：

```bash
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml
```

alpha ToT+ToA 双模态：

```bash
python scripts/train.py --config configs/experiments/alpha_resnet18_tot_toa.yaml
```

alpha ToT + 手工特征简单拼接：

```bash
python scripts/train.py --config configs/experiments/alpha_resnet18_tot_handcrafted_concat.yaml
```

alpha ToT + 手工特征门控融合：

```bash
python scripts/train.py --config configs/experiments/alpha_resnet18_tot_handcrafted_gated.yaml
```

C/质子 ToT：

```bash
python scripts/train.py --config configs/experiments/proton_resnet18_tot.yaml
```

## 6. 跑一组对比实验

比较不同损失函数：

```bash
python scripts/run_grid.py --config configs/experiments/compare_losses.yaml
```

比较不同模型：

```bash
python scripts/run_grid.py --config configs/experiments/compare_models.yaml
```

只查看将会跑哪些实验，不真正训练：

```bash
python scripts/run_grid.py --config configs/experiments/compare_losses.yaml --dry-run
```

## 7. 汇总实验结果

所有新实验默认输出到：

```text
outputs/experiments/
```

汇总：

```bash
python scripts/summarize.py
```

默认生成：

```text
outputs/experiment_summary.csv
```

## 8. 每个实验会保存什么

每次实验会创建一个单独目录，例如：

```text
outputs/experiments/20260426_203000_alpha_resnet18_tot/
```

里面包含：

```text
config.yaml            本次实际使用的配置
metadata.json          实验元数据
metrics.json           最佳验证结果和测试结果
training_log.csv       每个 epoch 的训练/验证指标
best_model.pth         最佳模型权重
predictions.csv        测试集预测结果
confusion_matrix.csv   测试集混淆矩阵
```

## 9. 当前第一阶段已经实现的内容

已实现：

- YAML 配置加载。
- 数据集模态校验。
- ToT/ToA 文件配对。
- train/val/test 分层划分。
- split manifest 复用。
- ToT/ToA 标准化。
- 90 度旋转增强。
- `total_energy` 手工特征。
- `none` / `concat` / `gated` 三种融合模式。
- `resnet18`、`shallow_resnet`、`shallow_cnn` 新接口模型。
- CrossEntropy 和 EMD 损失。
- accuracy、角度 MAE、macro-F1、混淆矩阵。
- 单实验运行、网格实验运行、结果汇总。

## 10. 当前限制

本地当前 Python 环境缺少 `torch`，所以这次只做了语法检查，没有在本机实际训练。

下一步建议在服务器上先跑一个小实验：

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_Clean \
  --set training.epochs=2 \
  --set training.batch_size=32
```

确认能跑通后，再开始跑正式对比实验。

