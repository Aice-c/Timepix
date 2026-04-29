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

实验编号、阶段目的、完成状态和后续安排以 `agent/EXPERIMENT_LOG.md` 的“实验编号与阶段总览”为准。后续新增对比实验时，除了新增或修改 YAML，也要同步在实验日志中记录编号含义、关键决策、运行命令和汇总命令。

## 3. 数据集配置

Alpha_100 数据集：

```yaml
name: Alpha_100
particle: alpha
root: ${TIMEPIX_DATA_ROOT:-Data}/Alpha_100
available_modalities: [ToT, ToA]
default_modalities: [ToT, ToA]
sample_shape: [100, 100]
```

`Alpha_50` 也有独立数据集配置，但当前正式实验主线统一使用 `Alpha_100`。

C/质子 7 分类数据集：

```yaml
name: Proton_C_7
particle: proton
root: ${TIMEPIX_DATA_ROOT:-Data}/Proton_C_7
available_modalities: [ToT]
default_modalities: [ToT]
```

这里最重要的是：

- `Alpha_100` 和 `Alpha_50` 支持 `ToT`、`ToA`、`ToT+ToA`。
- 当前正式 C/质子训练只使用 `Proton_C_7`，且只支持 `ToT`。
- 如果 C/质子误写 `ToA`，或配置里拼错常见字段，新系统会在训练开始前报错。

## 4. 本地和服务器路径如何处理

不要把服务器路径写死进代码。

推荐方法一：命令行覆盖。

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_100
```

推荐方法二：环境变量。

本地 PowerShell：

```powershell
$env:TIMEPIX_DATA_ROOT="D:\Project\Timepix\Data"
```

当前本地实际数据路径：

```text
Alpha_100  -> D:\Project\Timepix\Data\Alpha_100
Proton_C   -> E:\C1Analysis\Proton_C
Proton_C_7 -> E:\C1Analysis\Proton_C_7
```

因为 Alpha 与 Proton 位于不同盘符，不建议在本地长期依赖单个 `TIMEPIX_DATA_ROOT` 覆盖所有任务。更稳妥的方式是按任务显式传 `--data-root`。

训练/评估脚本的 `--data-root` 是具体数据集目录：

```powershell
python scripts\train.py --config configs\experiments\alpha_resnet18_tot.yaml --data-root D:\Project\Timepix\Data\Alpha_100
python scripts\run_grid.py --config configs\experiments\b1_proton_c7_resnet18_tot_lr_batch.yaml --data-root E:\C1Analysis\Proton_C_7 --dry-run
```

数据分析脚本的 `--data-root` 是父目录：

```powershell
python scripts\analyze_datasets.py --data-root D:\Project\Timepix\Data --datasets Alpha_100
python scripts\analyze_datasets.py --data-root E:\C1Analysis --datasets Proton_C
python scripts\analyze_resolution_limit.py --data-root E:\C1Analysis --dataset Proton_C
```

本地验证环境：

```powershell
conda activate timepix-local
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

注意：上面两个是早期简单示例，主要使用旧兼容的 `total_energy`。正式 A5 手工特征实验使用新的 A5a/A5b/A5c/A5d 编号和配置，训练特征实现位于 `timepix/data/features.py`，不复用 `timepix/analysis/` 数据分析特征。

A5a 手工特征筛选不训练 CNN，使用 `RandomForest` 与 one-vs-rest `LogisticRegression` 在 validation 上做特征重要性诊断：

```bash
python scripts/screen_handcrafted_features.py \
  --config configs/experiments/a5a_alpha_handcrafted_screening.yaml \
  --data-root /root/autodl-tmp/Alpha_100 \
  --out-dir outputs/a5_feature_screening \
  --name a5a_alpha_handcrafted_screening \
  --n-repeats 30
```

A5a 的 `LogisticRegression` 显式包装为 `OneVsRestClassifier(LogisticRegression(solver="liblinear"))`，用于兼容较新的 `scikit-learn` 多分类行为。A5a 输出是特征筛选诊断文件，不走 `scripts/summarize.py`；A5b/A5c/A5d 进入 CNN 训练后再使用标准 `run_grid.py`、`summarize.py` 和 `aggregate_seeds.py`。

A5b 低冗余手工特征组 CNN concat 消融：

```bash
python scripts/run_grid.py --config configs/experiments/a5b_alpha_handcrafted_group_ablation.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a5b_alpha_handcrafted_group_ablation.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a5b_alpha_handcrafted_group_ablation --out outputs/a5b_alpha_handcrafted_group_ablation_runs.csv
```

A5b 是 seed42 筛选实验，暂不做 `aggregate_seeds.py`；A5d 三 seed 认证阶段再聚合 mean/std。

A5c 镜像 A5b 的四个低冗余特征组，只把融合方式改为 `gated`。服务器建议使用 `tmux` 持久化运行：

```bash
cd ~/Timepix
tmux new -s a5c_gated
```

进入 `tmux` 后运行：

```bash
python scripts/run_grid.py --config configs/experiments/a5c_alpha_handcrafted_gated_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a5c_alpha_handcrafted_gated_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a5c_alpha_handcrafted_gated_seed42 --out outputs/a5c_alpha_handcrafted_gated_seed42_runs.csv
```

A5c 仍是 seed42 诊断实验，只需要 `summarize.py`；若后续进入 A5d 三 seed 认证，再使用 `scripts/aggregate_seeds.py` 聚合 mean/std。

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

命令记录规范：

- 每个新对比实验都要同时写清楚运行命令和汇总命令，不能只给 `run_grid.py`。
- 单 seed 快速版要给出对应的 summary CSV 输出命令。
- 三 seed 正式版要给出逐 run 汇总命令，以及 mean/std 聚合命令。
- 如果是 A4b 这类 checkpoint 诊断脚本，没有标准 `experiment_group`，也要明确 `--output-summary`、`--output-by-class`、`--output-json` 等输出路径。
- 训练命令默认按 Linux 服务器 bash 书写；本地验证命令才使用 Windows PowerShell 路径。

比较不同损失函数：

```bash
python scripts/run_grid.py --config configs/experiments/compare_losses.yaml
```

比较不同模型：

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml
```

时间紧张时先跑 A3 单 seed 快速版：

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml
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

A3 配置继承 `configs/experiments/alpha_tot_a2_best_base.yaml`，固定 `Alpha_100`、ToT、CE、one-hot、无手工特征、A2 best 训练超参，并显式复用 `outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json`，只切换 `model.name` 和 `training.seed: [42, 43, 44]`。所有新主干都走统一 `FeatureFusion + task head`，因此仍支持手工特征融合、分类/回归任务和现有损失配置。`vit_tiny` 使用原生 `100x100` 输入，A3 使用 `image_size: 100`、`patch_size: 10`。`model.dropout=0.1` 指统一 Timepix task head dropout；torchvision backbone 内部正则保持模型默认，不在 A3 中单独调参。

A4 模态对比：

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml
```

时间紧张时先跑 A4 单 seed 快速版：

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml
```

A4 继承 A2 best base，固定 `Alpha_100`、`resnet18_no_maxpool`、CE、one-hot、无手工特征和 `fusion_mode: none`，只切换 `dataset.modalities`：`[ToT, ToA]`、`[ToT]`、`[ToA]`，以及 `training.seed: [42, 43, 44]`。配置显式使用 `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`。因为 `Alpha_100` 的 ToT/ToA 文件完全一一对应，且 split manifest 使用归一化 sample key，这个 paired split 应从历史 `Alpha_100_ToT` split 复制得到，从而让 A4 与 A1/A2/A3 使用严格一致的数据划分。

A4b ToA 表达方式对比：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --continue-on-error
```

`a4b_toa_transform_seed42.yaml` 先用单 seed 对比 `relative_minmax`、`relative_centered`、`relative_rank` 三种 ToA 相对时间表达，以及是否追加 `hit_mask`。完整三 seed 版本为：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml --continue-on-error
```

A4b 仍固定 `Alpha_100`、同一 paired split、`resnet18_no_maxpool` 和 A2 best 训练超参；第一阶段只改 ToA 输入表达，不新增 dual-stream、GMU、FiLM 或 MMTM。

A4b 第二阶段可以直接评估 A4 已训练完成的 ToT 与 ToA 单模态 checkpoint：

```bash
python scripts/evaluate_logit_fusion.py \
  --group a4_modality_comparison_seed42 \
  --output-csv outputs/a4b_late_logit_fusion_seed42.csv \
  --output-json outputs/a4b_late_logit_fusion_seed42.json
```

该脚本做的是 late logit fusion：

```text
logits = (1 - alpha_toa) * logits_tot + alpha_toa * logits_toa
```

`alpha_toa` 只在 validation set 上选择，test set 只用于最终报告。完整 A4 三 seed 结果存在时，把 `--group` 改为 `a4_modality_comparison`。

A4b-2.5 预测互补性诊断：

```bash
python scripts/analyze_prediction_complementarity.py --seed 42
```

这个脚本只读取已有 `predictions.csv`，不训练、不加载 checkpoint。它会统计 ToT 正确/错误与 ToA 或 relative ToT+ToA 正确/错误的重叠关系，并给出 oracle accuracy 与 oracle MAE，用来判断 ToA 是否存在值得继续挖掘的互补信息。

A4b-3a/b oracle 控制诊断会重新加载 checkpoint，在 validation/test 上做确定性推理，不训练新模型。A4b-3a 的 ToT-vs-ToT 多 seed control 使用 `a2_best_3seed`，因为它是当前已完成且与 A3/A4 主线一致的 ToT 三 seed 基准组：

注意：`a2_best_3seed` 是历史 run，配置中仍记录 `dataset.name: Alpha` 和 `/root/autodl-tmp/Alpha`。如果服务器当前只有 `/root/autodl-tmp/Alpha_100`，先建立旧 split 名称的兼容别名，并在评估命令中传 `--data-root /root/autodl-tmp/Alpha_100`：

```bash
cd /root/Timepix
cp -n outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json
sha256sum outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-seed-control \
  --tot-group a2_best_3seed \
  --splits val,test \
  --seeds 42 43 44 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --output-json outputs/a4b_3a_tot_seed_control.json \
  --output-summary outputs/a4b_3a_tot_seed_control_summary.csv \
  --output-by-class outputs/a4b_3a_tot_seed_control_by_class.csv
```

A4b-3b 先做 seed42 的 ToT vs `relative_minmax/no mask` 复查：

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-vs-candidate \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --splits val,test \
  --seeds 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_3b_tot_vs_relative_minmax.json \
  --output-summary outputs/a4b_3b_tot_vs_relative_minmax_summary.csv \
  --output-by-class outputs/a4b_3b_tot_vs_relative_minmax_by_class.csv
```

A4b-3 当前结果：ToT-vs-ToT seed-control oracle gain 只有 validation +2.33%、test +2.55%，30 deg 上只有 validation +2.55%、test +1.15%；ToT vs `relative_minmax/no mask` 则达到 validation +10.19%、test +11.03%，30 deg 上达到 +27.08% 和 +25.52%。这支持继续做 selector/gate，而不是把互补性解释为普通 seed 波动。

A4b-4 selector fusion 分为三版。旧的泛称 A4b-4 初版结果作废，后续重新按 A4b-4a/4b/4c 运行。

A4b-4a rule-based selector：

```bash
python scripts/evaluate_selector_fusion.py \
  --selector-mode rule \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_4a_rule_selector_seed42.json \
  --output-summary outputs/a4b_4a_rule_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4a_rule_selector_seed42_by_class.csv
```

A4b-4b train-logit selector：

```bash
python scripts/evaluate_selector_fusion.py \
  --selector-mode trained \
  --selector-fit train \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --selector-target lower-error \
  --selector-epochs 500 \
  --selector-lr 0.01 \
  --selector-weight-decay 0.0001 \
  --output-json outputs/a4b_4b_train_logit_selector_seed42.json \
  --output-summary outputs/a4b_4b_train_logit_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4b_train_logit_selector_seed42_by_class.csv
```

A4b-4c validation-CV selector：

```bash
python scripts/evaluate_selector_fusion.py \
  --selector-mode trained \
  --selector-fit val-cv \
  --cv-folds 5 \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --selector-target lower-error \
  --selector-epochs 500 \
  --selector-lr 0.01 \
  --selector-weight-decay 0.0001 \
  --output-json outputs/a4b_4c_val_cv_selector_seed42.json \
  --output-summary outputs/a4b_4c_val_cv_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4c_val_cv_selector_seed42_by_class.csv
```

该脚本不重新训练 ResNet。规则、阈值和是否启用 selector 都只由 validation 决定，test 只做最终报告。若 validation 不支持 selector，脚本会选择 `primary_only`，即退回 ToT baseline。

当前结果：A4b-4a rule 为 70.97% Test Acc，A4b-4b train selector 为 71.17%，A4b-4c val-CV selector 为 70.38%，ToT baseline 为 70.48%，oracle 为 81.51%。因此严格结论是：简单 rule/train selector 可带来小幅改善，但 validation-CV selector 未稳定超过 ToT，可靠学习 oracle 切换仍未完全解决。

B1 Proton/C 训练超参搜索：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --continue-on-error
```

B1-1 固定 `Proton_C_7 + ToT` 与 A1 得到的 ResNet18 stem/variant：`resnet18_no_maxpool`、`conv1_kernel_size=2`、`conv1_stride=1`、`conv1_padding=0`，只搜索 `learning_rate × batch_size`。`dropout=0.1` 是保守训练默认值，不作为 A1 结构结论描述。当前版本使用 `epochs=25`、`early_stopping_patience=5`，输出组为 `b1_proton_c7_resnet18_tot_lr_batch_ep25`，用于和早期 20 epoch 诊断结果分开。

如果已经误用旧 20 epoch 预算跑完 B1-1，可以从 `last_checkpoint.pth` 继续到
25 epoch。推荐复制到 `b1_proton_c7_resnet18_tot_lr_batch_ep25_from20`，
并跳过已经早停的 run：

```bash
python scripts/extend_runs.py \
  --source-group b1_proton_c7_resnet18_tot_lr_batch \
  --target-group b1_proton_c7_resnet18_tot_lr_batch_ep25_from20 \
  --target-epochs 25 \
  --data-root /root/autodl-tmp/Proton_C_7 \
  --skip-completed \
  --skip-early-stopped \
  --resume-target-existing \
  --continue-on-error
```

这类结果需要标注为 `from20` 续跑：它节省算力，但由于前 20 epoch 已按旧
cosine schedule 训练，不完全等价于从头使用 `T_max=25` 的正式重跑。

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

汇总表中包含 `experiment_group`、模型名、`conv1_kernel_size`、`conv1_stride`、`conv1_padding`、`dropout`、`feature_dim`、`hidden_dim`、`image_size`、`patch_size`、`seed`、`split_seed`、`split_manifest_hash`、早停状态、训练超参数、混合精度状态、训练/测试耗时、git commit 和主要验证/测试指标，A1 结构对比、AMP 对比、主干模型对比或多 seed 对比可以直接按这些列筛选。

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

A4b 新增 ToA 输入表达控制：

```yaml
data:
  toa_transform: relative_minmax
  add_hit_mask: false
```

`toa_transform` 支持 `none`、`raw_log1p`、`relative_minmax`、`relative_centered` 和 `relative_rank`。`add_hit_mask: true` 会在图像输入末尾追加一个命中掩码通道，因此 ToT+ToA 输入会从 2 通道变成 3 通道。新系统会在 `metadata.json` / summary CSV 中记录 `input_channels`、`toa_transform` 和 `add_hit_mask`。

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

搜索配置继承 `configs/experiments/alpha_resnet18_tot.yaml`，固定 Dataset、Modality、Model、Loss、Label 和数据划分 seed，只搜索训练相关超参数：

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

A2 当前最佳训练配置已经沉淀为：

```text
configs/experiments/alpha_tot_a2_best_base.yaml
```

后续消融和模型对比优先继承该 base。

## 13. 多 seed 认证

`split.seed` 和 `training.seed` 已经拆开：

```yaml
split:
  seed: 42
  reuse_split: true

grid:
  training.seed: [42, 43, 44]
```

`split.seed` 控制 train/val/test 分层划分；`training.seed` 控制模型初始化、DataLoader shuffle 和训练随机性。旧配置如果不写 `split.seed`，仍会沿用 `training.seed`，保持兼容。

A2 当前最优训练超参的三 seed 认证入口：

```bash
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml
```

结果汇总和 mean/std 聚合：

```bash
python scripts/summarize.py --group a2_best_3seed --out outputs/a2_best_3seed_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a2_best_3seed_runs.csv --out outputs/a2_best_3seed_mean_std.csv
```

报告时建议写 `mean ± std`，不要从三个 seed 中挑最高值作为正式结果。

## 14. 当前第一阶段已经实现的内容

已实现：

- YAML 配置加载。
- 数据集模态校验。
- ToT/ToA 文件配对。
- train/val/test 分层划分。
- split manifest 复用，且支持独立 `split.seed`。
- 配置化读取精度，默认 `float32`。
- ToT/ToA 标准化。
- 90 度旋转增强。
- `total_energy` 手工特征。
- `none` / `concat` / `gated` 三种融合模式。
- `resnet18`、`resnet18_no_maxpool`、`resnet18_maxpool`、`resnet18_original`、`shallow_resnet`、`shallow_cnn`、`densenet121`、`efficientnet_b0`、`convnext_tiny`、`vit_tiny` 新接口模型。
- CrossEntropy 和 EMD 损失。
- accuracy、角度 MAE、P90 Error、macro-F1、混淆矩阵。
- 单实验运行、网格实验运行、结果汇总和多 seed mean/std 聚合。
- 实验组目录、metadata 实验组记录、按组/全部汇总。
- CUDA AMP 混合精度训练开关、GradScaler checkpoint 恢复、FP32/AMP 对比配置。
- Optuna/TPE 训练超参数搜索入口、搜索配置、trial CSV 和 best config 导出。

## 15. P90 Error 指标

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

## 16. 当前限制

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
  --data-root /root/autodl-tmp/Alpha_100 \
  --set training.epochs=2 \
  --set training.batch_size=32
```

确认能跑通后，再开始跑正式对比实验。

## 17. 进度条和持久化训练

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
python scripts/train.py --config configs/experiments/alpha_resnet18_tot.yaml --data-root /root/autodl-tmp/Alpha_100
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

## A4b-4d Selector Switch Diagnostics

Use this after A4b-4a/4b/4c to explain selector behavior. It does not train new
ResNet checkpoints and does not choose a new test-set rule. It applies the fixed
A4b-4a validation-selected rule `entropy_adv_0p03` and writes overall,
per-class, per-sample, and score-distribution diagnostics.

Server command:

```bash
cd /root/Timepix

python scripts/analyze_selector_switches.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --rule entropy_adv_0p03 \
  --output-json outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42.json \
  --output-summary outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_summary.csv \
  --output-by-class outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_by_class.csv \
  --output-samples outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_samples.csv \
  --output-distribution outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_distribution.csv
```

## A4b-4e Three-Seed Selector Confirmation

Train only the missing key candidate seeds:

```bash
python scripts/run_grid.py \
  --config configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml \
  --continue-on-error
```

Then combine the existing seed42 candidate group with the new seed43/44 group
for oracle and selector evaluation:

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-vs-candidate \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
  --seeds 42 43 44 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_4e_oracle_3seed.json \
  --output-summary outputs/a4b_4e_oracle_3seed_summary.csv \
  --output-by-class outputs/a4b_4e_oracle_3seed_by_class.csv
```

```bash
for seed in 42 43 44; do
  python scripts/evaluate_selector_fusion.py \
    --selector-mode rule \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_4e_rule_selector_seed${seed}.json" \
    --output-summary "outputs/a4b_4e_rule_selector_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_4e_rule_selector_seed${seed}_by_class.csv"
done

python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_4e_rule_selector_seed42_summary.csv \
    outputs/a4b_4e_rule_selector_seed43_summary.csv \
    outputs/a4b_4e_rule_selector_seed44_summary.csv \
  --out outputs/a4b_4e_rule_selector_mean_std.csv
```

## A4b-5 Gated Late Fusion

A4b-5 does not retrain ResNet experts. It reloads frozen ToT/candidate logits and
compares sample-wise gate variants.

Seed42:

```bash
python scripts/evaluate_gated_late_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_5_gated_late_fusion_seed42.json \
  --output-summary outputs/a4b_5_gated_late_fusion_seed42_summary.csv \
  --output-by-class outputs/a4b_5_gated_late_fusion_seed42_by_class.csv
```

Three-seed run after A4b-4e:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_gated_late_fusion.py \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_5_gated_late_fusion_seed${seed}.json" \
    --output-summary "outputs/a4b_5_gated_late_fusion_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_5_gated_late_fusion_seed${seed}_by_class.csv"
done
```

Aggregate A4b-5:

```bash
python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_5_gated_late_fusion_seed42_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed43_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed44_summary.csv \
  --out outputs/a4b_5_gated_late_fusion_mean_std.csv
```

## A4b-6 Residual Gated Fusion

A4b-6 does not retrain ResNet experts. It treats the candidate as a constrained
correction to the ToT logits:

```text
logits_final = logits_tot + residual_weight * (logits_candidate - logits_tot)
```

Seed42:

```bash
python scripts/evaluate_residual_gated_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_6_residual_gated_fusion_seed42.json \
  --output-summary outputs/a4b_6_residual_gated_fusion_seed42_summary.csv \
  --output-by-class outputs/a4b_6_residual_gated_fusion_seed42_by_class.csv
```

Three-seed run after A4b-4e:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_residual_gated_fusion.py \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_6_residual_gated_fusion_seed${seed}.json" \
    --output-summary "outputs/a4b_6_residual_gated_fusion_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_6_residual_gated_fusion_seed${seed}_by_class.csv"
done

python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_6_residual_gated_fusion_seed42_summary.csv \
    outputs/a4b_6_residual_gated_fusion_seed43_summary.csv \
    outputs/a4b_6_residual_gated_fusion_seed44_summary.csv \
  --out outputs/a4b_6_residual_gated_fusion_mean_std.csv
```
