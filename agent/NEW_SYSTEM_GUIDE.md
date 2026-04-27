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

## 6. 实验组

实验配置支持：

```yaml
experiment_name: alpha_resnet18_tot
experiment_group: baseline
```

输出目录会按组保存：

```text
outputs/experiments/baseline/20260427_120000_alpha_resnet18_tot/
```

如果没有写 `experiment_group`，默认使用 `default` 组。

`metadata.json` 中也会记录：

```json
{
  "experiment_group": "baseline"
}
```

## 7. 跑一组对比实验

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

如果 grid 配置中写了：

```yaml
experiment_group: compare_losses
```

该组 grid 实验都会保存到：

```text
outputs/experiments/compare_losses/
```

## 8. 汇总实验结果

所有新实验默认输出到：

```text
outputs/experiments/
```

汇总：

```bash
python scripts/summarize.py --all
```

无参数运行也会汇总全部实验组：

```bash
python scripts/summarize.py
```

默认生成：

```text
outputs/experiment_summary.csv
```

只汇总某个实验组：

```bash
python scripts/summarize.py --group baseline
```

默认生成：

```text
outputs/baseline_summary.csv
```

也可以按路径汇总：

```bash
python scripts/summarize.py \
  --root outputs/experiments/baseline \
  --out outputs/baseline_summary.csv
```

汇总表中包含 `experiment_group` 列。

## 9. 每个实验会保存什么

每次实验会创建一个单独目录，例如：

```text
outputs/experiments/baseline/20260426_203000_alpha_resnet18_tot/
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

其中 `predictions.csv` 会同时保存每个测试样本的绝对角度误差，便于后续分析 P90 Error 对应的高误差样本。

## 10. 数据精度设置

新系统支持在实验配置中指定读取矩阵时使用的浮点精度：

```yaml
data:
  crop_size: 0
  dtype: float32
```

建议默认使用 `float32`。

原因：

- 深度学习训练本身通常使用 `float32`。
- 质子/C 的 ToT 文本如果有很多位小数，保留到 `float64` 对模型效果通常帮助不大。
- `float32` 可以降低内存占用和张量传输成本。

需要注意：如果原始数据仍是很多位小数的 `.txt`，每次训练仍然要解析文本，I/O 和文本解析可能还是慢。若服务器上训练明显受数据读取拖累，下一步应把 `.txt` 数据一次性转换成 `.npy` 或缓存格式，再训练时直接读二进制数组。

## 11. 当前第一阶段已经实现的内容

已实现：

- YAML 配置加载。
- 数据集模态校验。
- ToT/ToA 文件配对。
- train/val/test 分层划分。
- split manifest 复用。
- 配置化读取精度，默认 `float32`。
- ToT/ToA 标准化。
- 90 度旋转增强。
- `total_energy` 手工特征。
- `none` / `concat` / `gated` 三种融合模式。
- `resnet18`、`shallow_resnet`、`shallow_cnn` 新接口模型。
- CrossEntropy 和 EMD 损失。
- accuracy、角度 MAE、P90 Error、macro-F1、混淆矩阵。
- 单实验运行、网格实验运行、结果汇总。
- 实验组目录、metadata 实验组记录、按组/全部汇总。

## 12. P90 Error 指标

新系统会在 `metrics.json`、`metadata.json` 和 `training_log.csv` 中记录 `p90_error`。

它的含义是：

```text
90% 样本的角度绝对误差不超过多少度
```

分类任务中：

- `p90_error`：基于 argmax 预测类别对应角度。
- `p90_error_weighted`：基于概率加权预测角度。

回归任务中：

- `p90_error`：基于连续预测角度。

这个指标比平均误差更能反映“较差的那一部分样本”的表现，适合和 accuracy、MAE、混淆矩阵一起用于论文分析。

## 13. 当前限制

本地当前 Python 环境缺少 `torch`，所以这次只做了语法检查，没有在本机实际训练。

如果服务器报：

```text
ModuleNotFoundError: No module named 'timepix.data'
```

说明 `timepix/data/` 代码目录没有同步到服务器。原因通常是 `.gitignore` 把 `Data/` 写成了非根目录规则，导致 Git 在 Windows 上也忽略了 `timepix/data/`。正确规则应该是：

```gitignore
/Data/
/Program/data/
```

然后在笔记本上提交：

```bash
git add .gitignore timepix/data
git commit -m "Track timepix data package"
git push
```

服务器再执行 `git pull`。

下一步建议在服务器上先跑一个小实验：

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_Clean \
  --set training.epochs=2 \
  --set training.batch_size=32
```

确认能跑通后，再开始跑正式对比实验。
