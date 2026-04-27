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
name: Alpha
particle: alpha
root: ${TIMEPIX_DATA_ROOT:-Data}/Alpha
available_modalities: [ToT, ToA]
default_modalities: [ToT, ToA]
```

C/质子数据集：

```yaml
name: Proton_C
particle: proton
root: ${TIMEPIX_DATA_ROOT:-Data}/Proton_C
available_modalities: [ToT]
default_modalities: [ToT]
```

这里最重要的是：

- alpha 支持 `ToT`、`ToA`、`ToT+ToA`。
- C/质子只支持 `ToT`。
- 如果 C/质子误写 `ToA`，或配置里拼错常见字段，新系统会在训练开始前报错。

## 4. 本地和服务器路径如何处理

不要把服务器路径写死进代码。

推荐方法一：命令行覆盖。

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha
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

当前模型主干对比包含：

```text
shallow_cnn
shallow_resnet
resnet18_no_maxpool
densenet121
efficientnet_b0
convnext_tiny
vit_tiny
```

该配置固定 Alpha、ToT、CE、one-hot、无手工特征、A1 最佳 ResNet18 stem 参数和 AMP，只切换 `model.name`。所有新主干都走统一 `FeatureFusion + task head`，因此仍支持手工特征融合、分类/回归任务和现有损失配置。`vit_tiny` 是项目内适配 50x50 Timepix 矩阵的小型 ViT，默认 `image_size: 50`、`patch_size: 10`。

比较 FP32 与 CUDA AMP 混合精度：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml
```

A1 原始 ResNet18 baseline：

```bash
python scripts/train.py --config configs/experiments/a1_resnet18_original_baseline.yaml
```

A1 ResNet18 结构适配网格：

```bash
python scripts/run_grid.py --config configs/experiments/a1_structure_adaptation.yaml
```

长网格实验建议开启失败续跑能力：

```bash
python scripts/run_grid.py \
  --config configs/experiments/a1_structure_adaptation.yaml \
  --skip-existing \
  --continue-on-error
```

非 dry-run 网格会写入运行 manifest，默认位于：

```text
outputs/grid_manifests/
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

ResNet18 结构参数可以直接写进 grid。`resnet18` 是去除第一层 maxpool 的默认适配版，等价于 `resnet18_no_maxpool`；保留第一层 maxpool 时使用 `resnet18_maxpool`。常用参数名：

```yaml
model:
  name: resnet18_no_maxpool
  conv1_kernel_size: 2
  conv1_stride: 1
  conv1_padding: 0
  dropout: 0.1
```

严格原始 ResNet18 stem baseline 使用 `resnet18_original`，它固定为 `7x7/stride=2/padding=3` + 第一层 maxpool，不参与 A1 网格搜索。

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

汇总表中包含 `experiment_group`、模型名、`conv1_kernel_size`、`conv1_stride`、`conv1_padding`、`dropout`、`feature_dim`、`hidden_dim`、`image_size`、`patch_size`、早停状态、训练超参数、混合精度状态、训练/测试耗时、git commit 和主要验证/测试指标，A1 结构对比、AMP 对比或主干模型对比可以直接按这些列筛选。

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

其中 `metadata.json` 会记录配置、数据 split 信息、每个 split 的类别计数、环境信息、git 信息、混合精度实际启用状态、训练/测试耗时、是否早停、最佳 epoch 和停止 epoch。`training_log.csv` 会记录每个 epoch 的耗时。`predictions.csv` 会同时保存每个测试样本的绝对角度误差，便于后续分析 P90 Error 对应的高误差样本。

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

## 11. 混合精度训练

训练计算精度和上一节的 `data.dtype` 是两件事：`data.dtype` 控制从文本矩阵读入后的数据张量精度，`training.mixed_precision` 控制 GPU 前向/反向计算是否使用 CUDA AMP。

配置方式：

```yaml
training:
  mixed_precision: true
  mixed_precision_dtype: float16
```

默认配置保持 `mixed_precision: false`，便于把 FP32 作为基准。开启后，训练、验证和测试都会使用 autocast；FP16 训练会启用 GradScaler。`last_checkpoint.pth` 中会保存 scaler 状态，所以中断后可以继续 `--resume`。

为了判断混合精度是否影响精度，先在 A1 当前最佳结构上跑完全相同条件下的对比实验：

```bash
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/compare_mixed_precision.yaml
```

该配置固定 `resnet18_no_maxpool`、`conv1_kernel_size: 2`、`conv1_stride: 1`、`dropout: 0.3`，只切换 `training.mixed_precision: false/true`。结果汇总后重点比较 `fit_seconds`、`test_accuracy`、`test_mae_argmax` 和 `test_p90_error`。

## 12. 训练超参数搜索

新系统提供 Optuna/TPE 搜索入口，用于在代表性设置上先找一组较好的训练超参数，再固定到后续消融和模型对比中。

代表性 Alpha ToT ResNet18 A2 搜索：

```bash
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml --dry-run
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml
```

搜索配置继承 `configs/experiments/alpha_resnet18_tot.yaml`，固定 Dataset、Modality、Model、Loss、Label 和 Seed，只搜索训练相关超参数：

```yaml
search:
  sampler: tpe
  objective: validation.accuracy
  parameters:
    training.learning_rate: ...
    training.weight_decay: ...
    training.batch_size: ...
    training.eta_min: ...
    model.dropout: ...
```

搜索目标使用 validation 指标，test 指标只用于最终报告，避免用测试集调参。每个 trial 都是一个普通实验目录，完整保存 `config.yaml`、checkpoint、metadata 和预测结果。搜索总目录默认在：

```text
outputs/hparam_search/
```

其中包含：

```text
search_config.yaml
trials.csv
study_summary.json
best_params.json
best_config.yaml
```

Optuna study 默认保存为 SQLite：

```text
outputs/optuna/hparam_alpha_resnet18_tot_a2.db
```

服务器中断后，用同一个搜索配置再次运行即可接着已有 study 继续采样。搜索结束后，可以把 `best_config.yaml` 中的训练超参数整理回后续正式实验配置，作为消融和模型对比的固定训练预算。

## 13. 当前第一阶段已经实现的内容

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
- `resnet18`、`resnet18_no_maxpool`、`resnet18_maxpool`、`resnet18_original`、`shallow_resnet`、`shallow_cnn`、`densenet121`、`efficientnet_b0`、`convnext_tiny`、`vit_tiny` 新接口模型。
- CrossEntropy 和 EMD 损失。
- accuracy、角度 MAE、P90 Error、macro-F1、混淆矩阵。
- 单实验运行、网格实验运行、结果汇总。
- 实验组目录、metadata 实验组记录、按组/全部汇总。
- CUDA AMP 混合精度训练开关、GradScaler checkpoint 恢复、FP32/AMP 对比配置。
- Optuna/TPE 训练超参数搜索入口、搜索配置、trial CSV 和 best config 导出。

## 14. P90 Error 指标

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

## 15. 当前限制

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
  --data-root /root/autodl-tmp/Alpha \
  --set training.epochs=2 \
  --set training.batch_size=32
```

确认能跑通后，再开始跑正式对比实验。

## 16. 进度条和持久化训练

训练时终端会显示每个 epoch 的 train/val batch 进度条，并在每个 epoch 结束后打印：

```text
Epoch summary | train_loss=... val_loss=... val_acc=... val_mae=... val_p90=...
```

配置项：

```yaml
training:
  progress_bar: true
  save_last_checkpoint: true
```

服务器长时间训练推荐使用 `tmux`：

```bash
cd ~/Timepix
tmux new -s timepix
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml --data-root /root/autodl-tmp/Alpha
```

断开 tmux：

```text
Ctrl+b 然后按 d
```

恢复查看：

```bash
tmux attach -t timepix
```

如果进程真的中断，可以用自动保存的 checkpoint 恢复：

```bash
python scripts/train.py \
  --resume outputs/experiments/baseline/<experiment_dir>/last_checkpoint.pth
```

新的 checkpoint 会保存训练配置，因此恢复命令可以不再重复写 `--config`。如果是旧 checkpoint，或者服务器上的数据路径变了，可以显式加上 `--config` 和 `--data-root`。

更完整说明见 `agent/SERVER_TRAINING.md`。
