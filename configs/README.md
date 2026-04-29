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

本地 Windows 当前数据路径：

```text
Alpha_100  -> D:\Project\Timepix\Data\Alpha_100
Proton_C   -> E:\C1Analysis\Proton_C
Proton_C_7 -> E:\C1Analysis\Proton_C_7
```

训练和 checkpoint 评估脚本的 `--data-root` 覆盖的是具体数据集目录：

```powershell
python scripts\train.py --config configs\experiments\alpha_resnet18_tot.yaml --data-root D:\Project\Timepix\Data\Alpha_100
python scripts\run_grid.py --config configs\experiments\b1_proton_c7_resnet18_tot_lr_batch.yaml --data-root E:\C1Analysis\Proton_C_7 --dry-run
```

论文数据分析脚本的 `--data-root` 是父目录：

```powershell
python scripts\analyze_datasets.py --data-root D:\Project\Timepix\Data --datasets Alpha_100
python scripts\analyze_datasets.py --data-root E:\C1Analysis --datasets Proton_C
python scripts\analyze_resolution_limit.py --data-root E:\C1Analysis --dataset Proton_C
```

由于本地 Alpha 与 Proton 不在同一个父目录下，不要直接用一个本地 `--data-root` 同时分析 `Alpha_100 Proton_C`；如需合并报告，先建立本地链接/镜像父目录。

## 重要模态约束

- 当前正式 Alpha 主线使用 `Alpha_100`，配置文件为 `configs/datasets/alpha_100.yaml`，输入尺寸为 `100x100`。
- `Alpha_50` 保留为对照/历史数据集配置，不用于当前正式 A3/A4 后续实验主线。
- `Alpha_100` 和 `Alpha_50` 均支持 `ToT` 和 `ToA`。
- 当前正式 Proton/C 主线使用 `Proton_C_7`，配置文件为 `configs/datasets/proton_c_7.yaml`，代表 7 分类质子/C 数据集；只支持 `ToT`。
- `configs/datasets/proton_c.yaml` 仅作为兼容入口保留，也指向 `Proton_C_7`，后续训练配置不要再写旧名 `Proton_C`。
- 独立论文数据分析链路不属于训练主线：`scripts/analyze_datasets.py` 和 `scripts/analyze_resolution_limit.py` 默认分析全量 `Proton_C`，用于和训练用的 `Proton_C_7` 区分。
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

汇总 CSV 会包含模型结构超参数列，例如 `conv1_kernel_size`、`conv1_stride`、`conv1_padding`、`dropout`、`feature_dim`、`hidden_dim`、`image_size` 和 `patch_size`，也会记录 `input_channels`、`toa_transform`、`add_hit_mask`、`seed`、`split_seed`、`split_manifest_hash`、`mixed_precision` / `mixed_precision_enabled` 与 `fit_seconds`，方便直接筛选 A1、AMP、主干模型、多 seed 对比或 A4b ToA 表达方式结果。

## 对比实验命令记录规范

新增或修改对比实验配置时，文档必须同步给出完整命令链：

- 服务器运行命令：单 seed 或三 seed，按实验设计说明。
- 汇总命令：标准训练组使用 `scripts/summarize.py --group ... --out ...`。
- 多 seed 聚合命令：标准训练组使用 `scripts/aggregate_seeds.py`；A4b selector/gate 这类后处理诊断使用 `scripts/aggregate_selector_fusion.py`。
- 诊断脚本输出：如果脚本不产生标准训练目录，需要明确 `--output-summary`、`--output-by-class`、`--output-json` 等输出文件。

这条规则是为了保证每个实验不只“能跑”，还可以稳定生成论文表格所需的可追溯 CSV。

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

时间紧张时，可以先运行固定 `training.seed=42` 的快速版。该配置继承完整 A3，只保留 7 个模型主干各跑一次：

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml
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

时间紧张时，可以先运行固定 `training.seed=42` 的快速版。该配置继承完整 A4，只保留 ToT+ToA、ToT、ToA 三个模态各跑一次：

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml
```

跑完后建议计算三 seed 均值和标准差：

```bash
python scripts/summarize.py --group a4_modality_comparison --out outputs/a4_modality_comparison_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4_modality_comparison_runs.csv --out outputs/a4_modality_comparison_mean_std.csv
```

## A4b ToA 表达方式对比

`configs/experiments/a4b_toa_transform.yaml` 用于在 A4 之后检查 ToA 的输入表达是否影响 early fusion 效果。该配置仍继承 A2 best base，固定 `Alpha_100`、`resnet18_no_maxpool`、A2 best 训练超参、CE、one-hot、无手工特征和同一份 paired split，只切换 ToA 变换方式与是否加入 hit mask。

新增数据配置字段：

```yaml
data:
  toa_transform: relative_minmax
  add_hit_mask: false
```

`toa_transform` 支持：

```text
none
raw_log1p
relative_minmax
relative_centered
relative_rank
```

`add_hit_mask: true` 会在图像输入末尾追加一个命中掩码通道，输入从 `[ToT, transformed_ToA]` 变为 `[ToT, transformed_ToA, hit_mask]`。模型输入通道数由 dataloader 记录的 `data_info.input_channels` 决定，因此可以参与 grid 对比。

A4b 第一阶段配置不重复 A4 的 raw/log1p baseline；A4 已经提供 ToT、ToA 和 ToT+ToA raw/log1p 结果。对 relative ToA 变换，配置中关闭 `normalization.ToA.log1p`：

```yaml
normalization:
  ToA:
    enabled: true
    log1p: false
    ignore_zero: true
```

时间紧张时先运行 seed42 快速版：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --continue-on-error
python scripts/summarize.py --group a4b_toa_transform_seed42 --out outputs/a4b_toa_transform_seed42_runs.csv
```

如果 seed42 值得继续，再运行三 seed 版本：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml --continue-on-error
python scripts/summarize.py --group a4b_toa_transform --out outputs/a4b_toa_transform_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4b_toa_transform_runs.csv --out outputs/a4b_toa_transform_mean_std.csv
```

A4b 第二阶段使用 A4 已训练好的 ToT 与 ToA 单模态 checkpoint 做 late logit fusion，不重新训练模型。脚本只用 validation set 选择 `alpha_toa`，再报告 test 指标：

```bash
python scripts/evaluate_logit_fusion.py \
  --group a4_modality_comparison_seed42 \
  --output-csv outputs/a4b_late_logit_fusion_seed42.csv \
  --output-json outputs/a4b_late_logit_fusion_seed42.json
```

默认融合权重为：

```text
alpha_toa = 0, 0.05, 0.10, 0.20, 0.30, 0.50
```

如果完整 A4 三 seed 结果已经存在，可以改用：

```bash
python scripts/evaluate_logit_fusion.py \
  --group a4_modality_comparison \
  --output-csv outputs/a4b_late_logit_fusion_runs.csv \
  --output-json outputs/a4b_late_logit_fusion_runs.json
```

A4b-2.5 使用已有 `predictions.csv` 做预测互补性诊断，不训练、不加载 checkpoint：

```bash
python scripts/analyze_prediction_complementarity.py --seed 42
```

默认输出：

```text
outputs/a4b_prediction_complementarity_seed42.json
outputs/a4b_prediction_complementarity_seed42_summary.csv
outputs/a4b_prediction_complementarity_seed42_by_class.csv
```

这个脚本回答：ToA 或 relative ToT+ToA 是否能在 ToT 出错时预测正确、是否有更小角度误差，以及 oracle fusion 的 accuracy/MAE 上限。

A4b-3a/b 使用 checkpoint 重新在 validation/test 上做确定性推理，用于排查 oracle 提升是否只是 seed 差异，并复查互补性是否也存在于 validation。该脚本不会训练新模型。

旧的 `a2_best_3seed` run 记录中 dataset 名称/路径仍是历史 `Alpha`、`/root/autodl-tmp/Alpha`，但它实际对应当前正式主线 `Alpha_100`。服务器重放 A4b-3a/b 时不要修改历史 run 文件；先准备 split 兼容别名，并在脚本命令中显式覆盖数据目录：

```bash
cd /root/Timepix
test -d /root/autodl-tmp/Alpha_100
test -f outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json
test -f outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
cp -n outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json
sha256sum outputs/splits/Alpha_100_ToT_seed42_0.8_0.1_0.1.json outputs/splits/Alpha_ToT_seed42_0.8_0.1_0.1.json outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

A4b-3a 的纯 ToT seed control 使用 `a2_best_3seed`，因为它是当前已完成的 `Alpha_100 + ToT + resnet18_no_maxpool + A2 best` 三 seed 基准组：

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

A4b-3b 先做 seed42 的 `ToT` vs `relative_minmax/no mask` 复查；ToT 侧同样优先来自 `a2_best_3seed`，candidate 侧来自 `a4b_toa_transform_seed42`。选择 `relative_minmax/no mask` 的依据是 A4b-2.5：它虽然不是 standalone Test Acc 最高的候选，但与 ToT 的 oracle Test Acc 最高、30 deg oracle 改善最明显：

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

A4b-3 当前结果显示，ToT-vs-ToT 随机 seed control 的 oracle gain 很小：validation/test 分别约为 +2.33% 和 +2.55%，30 deg 上仅约 +2.55% 和 +1.15%。而 ToT vs `relative_minmax/no mask` 的 oracle gain 在 validation/test 分别为 +10.19% 和 +11.03%，30 deg 上达到 +27.08% 和 +25.52%。因此该互补性不能简单归因于随机 seed 差异，后续应进入 selector/gate 融合验证。

A4b-4 使用 selector 验证互补性是否可学习。该脚本不训练新的 ResNet，只重新加载 ToT 与 `relative_minmax/no mask` checkpoint。旧的泛称 A4b-4 初版结果作废，后续按三个编号重新运行：

- A4b-4a：`--selector-mode rule`，不训练模型，只在 validation 上选择简单规则。
- A4b-4b：`--selector-mode trained --selector-fit train`，在 train logits 上训练 logistic selector，作为探索性对照。
- A4b-4c：`--selector-mode trained --selector-fit val-cv`，在 validation 内做 cross-fitting 并选择 threshold，是更严格的 selector 版本。

A4b-4a：

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

A4b-4b：

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

A4b-4c：

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

脚本会同时输出 `primary_only`、`candidate_only`、规则/selector 候选和 `oracle`。`primary_only` 也参与 validation 策略选择，因此 selector 无效时会退回 ToT baseline。

当前 A4b-4 结果显示：A4b-4a 规则 `entropy_adv_0p03` 获得 Test Acc 70.97%，相对 ToT +0.50%；A4b-4b train-logit selector 获得 71.17%，相对 ToT +0.70%；更严格的 A4b-4c validation-CV selector 为 70.38%，相对 ToT -0.10%。因此只能谨慎说明规则/训练集 selector 有小幅真实收益，但严格 validation-CV selector 未能稳定超过 ToT，仍与 oracle 81.51% 存在很大差距。

## B1 Proton/C 训练超参搜索

`configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml` 是质子/C 7 分类数据集的第一轮训练超参搜索。它固定 alpha A1 得到的 ResNet18 stem/variant：

```yaml
model:
  name: resnet18_no_maxpool
  conv1_kernel_size: 2
  conv1_stride: 1
  conv1_padding: 0
```

同时固定 `Proton_C_7`、`ToT`、CE、one-hot、无手工特征、`fusion_mode: none`、cosine scheduler、`eta_min=1e-7`、`weight_decay=1e-4`、`dropout=0.1`、`epochs=25` 和 `early_stopping_patience=5`。这里的 `dropout=0.1` 是沿用 A2 风格的保守训练默认值，不表述为 A1 结构参数。

原 B1-1 使用 `epochs=20`，部分组合停止时准确率仍在上升，因此当前配置提升到 25 epoch。为了不混入旧结果，配置的 `experiment_group` 为 `b1_proton_c7_resnet18_tot_lr_batch_ep25`。

B1-1 只搜索：

```yaml
grid:
  training.learning_rate:
    - 0.0001
    - 0.0003
    - 0.001
  training.batch_size:
    - 64
    - 128
    - 256
```

运行：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_lr_batch_ep25 --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_runs.csv
```

如果 `batch_size=256` 显存不足，`--continue-on-error` 会继续后面的组合；后续 B1-2 将基于 B1-1 最佳 `learning_rate + batch_size` 搜索 `weight_decay`。

当前 B1-1 结果结论：

- 20 epoch 旧结果中，validation-selected 最佳组合为 `learning_rate=3e-4`、`batch_size=128`，同时 Test Acc、Test MAE 和 Test F1 也最优。
- from20 中继到 25 epoch 后，4 组未早停 run 被继续训练；`1e-4` 系列略有改善，但按 validation 选择的最佳组合仍然是 `learning_rate=3e-4`、`batch_size=128`。
- 因此 B1-2 固定：

```text
learning_rate = 3e-4
batch_size    = 128
dropout       = 0.1
scheduler     = cosine
eta_min       = 1e-7
```

并继续搜索：

```text
weight_decay = [0, 1e-5, 1e-4]
```

### B1-2 weight decay 搜索

`configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml` 是 B1-2 配置。它继承 B1-1 的 `Proton_C_7 + ToT + resnet18_no_maxpool + conv1 2/1/0 + CE one-hot + AMP + 25 epochs` 设置，固定 B1-1 最佳组合：

```text
learning_rate = 3e-4
batch_size    = 128
```

只搜索：

```yaml
grid:
  training.learning_rate:
    - 0.0003
  training.batch_size:
    - 128
  training.weight_decay:
    - 0.0
    - 0.00001
    - 0.0001
```

注意：因为该配置继承 B1-1，必须在 `grid` 中显式写入单值 `training.learning_rate` 和 `training.batch_size`，否则会继承父配置的 `learning_rate × batch_size` 网格，错误扩展为 27 组。

运行与汇总：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml --continue-on-error
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_weight_decay_ep25 --out outputs/b1_proton_c7_resnet18_tot_weight_decay_ep25_runs.csv
```

B1-2 当前结果结论：

| `weight_decay` | Best epoch | Early stop | Val Acc | Test Acc | Test MAE | Test F1 |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 | 13 | 是 | 93.57% | 93.90% | 0.578 | 0.9561 |
| 1e-5 | 7 | 是 | 92.56% | 92.42% | 0.715 | 0.9446 |
| 1e-4 | 17 | 是 | 93.84% | 93.97% | 0.574 | 0.9563 |

按 `val_accuracy` 选择，B1-2 最佳仍为：

```text
learning_rate = 3e-4
batch_size    = 128
weight_decay  = 1e-4
dropout       = 0.1
scheduler     = cosine
eta_min       = 1e-7
```

`weight_decay=0` 与最佳组很接近，但 validation、test 和 MAE 均略低；`weight_decay=1e-5` 明显更差。下一步进入 `B1-best` 三 seed 认证，固定上述组合并运行 `training.seed=42/43/44`。

### B1-best 三 seed 认证

`configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml` 固定 B1-2 最佳组合，只展开训练随机种子：

```yaml
training:
  learning_rate: 0.0003
  batch_size: 128
  weight_decay: 0.0001
  scheduler: cosine
  eta_min: 0.0000001
  epochs: 25
  early_stopping_patience: 5
  mixed_precision: true

grid:
  training.seed: [42, 43, 44]
```

注意：B1-best 不继承 B1-2 配置文件，因为 B1-2 配置含有 `weight_decay` 搜索 grid；为了避免深度合并后误跑旧搜索项，B1-best 独立写出固定配置。

服务器 `tmux` 持久化运行：

```bash
cd ~/Timepix
tmux new -s b1_best
```

进入 `tmux` 后一次性运行训练、逐 run 汇总和三 seed 聚合：

```bash
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_best_3seed --out outputs/b1_proton_c7_resnet18_tot_best_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b1_proton_c7_resnet18_tot_best_3seed_runs.csv --out outputs/b1_proton_c7_resnet18_tot_best_3seed_mean_std.csv
```

### B1-1 20 epoch 结果续跑到 25 epoch

如果 B1-1 已经用旧的 20 epoch 预算跑完，且每个 run 都保留了
`last_checkpoint.pth`，可以用 `scripts/extend_runs.py` 继续到 25 epoch。
推荐复制到新组 `b1_proton_c7_resnet18_tot_lr_batch_ep25_from20`，不要覆盖旧
20 epoch 结果。

注意：这类结果是 `from20` 续跑结果，不完全等价于从一开始就用
`CosineAnnealingLR(T_max=25)` 训练，因为前 20 个 epoch 已经按旧的 cosine
schedule 跑完。它适合用作节省算力的 B1 epoch-budget rescue，并应在实验日志中
标注。

先 dry-run：

```bash
python scripts/extend_runs.py \
  --source-group b1_proton_c7_resnet18_tot_lr_batch \
  --target-group b1_proton_c7_resnet18_tot_lr_batch_ep25_from20 \
  --target-epochs 25 \
  --data-root /root/autodl-tmp/Proton_C_7 \
  --skip-completed \
  --skip-early-stopped \
  --resume-target-existing \
  --dry-run
```

确认计划无误后执行：

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

汇总：

```bash
python scripts/summarize.py \
  --group b1_proton_c7_resnet18_tot_lr_batch_ep25_from20 \
  --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_from20_runs.csv
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

## A4b-4d selector switch diagnostics

A4b-4d is a no-training diagnostic for the A4b-4a selected rule
`entropy_adv_0p03`. It reloads the frozen ToT expert and the
`relative_minmax/no mask` candidate, applies the fixed rule, and reports switch
precision/recall, harmful switches, per-class switch behavior, per-sample
outcomes, and score distributions.

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

## A4b-4e three-seed selector confirmation

A4b-4e checks whether the A4b-4a rule-selector result is stable across seeds.
It does not rerun the whole A4b transform grid. It trains only the key
candidate `ToT + relative_minmax ToA, no mask` for seeds 43 and 44, then reuses
seed42 from `a4b_toa_transform_seed42`.

Candidate config:

```text
configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml
```

Training:

```bash
python scripts/run_grid.py \
  --config configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml \
  --dry-run

python scripts/run_grid.py \
  --config configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml \
  --continue-on-error
```

Oracle across three seeds:

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

Rule selector across three seeds:

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

## A4b-5 gated late fusion

A4b-5 uses frozen ToT/candidate experts and trains or calibrates only a
sample-wise gate. It compares entropy soft gate, learned scalar probability
gate, learned scalar logit gate, class-aware gate, and conservative gate in one
script.

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

Three seeds, after A4b-4e candidate seeds finish:

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

python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_5_gated_late_fusion_seed42_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed43_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed44_summary.csv \
  --out outputs/a4b_5_gated_late_fusion_mean_std.csv
```

## A4b-6 residual gated fusion

A4b-6 keeps ToT as the primary expert and uses the `relative_minmax/no mask`
candidate only as a residual correction.

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

Three seeds, after A4b-4e candidate seeds finish:

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

## A4c end-to-end full bimodal fusion

`configs/experiments/a4c_end_to_end_bimodal_fusion.yaml` 是 A4c 第一批完整端到端双模态模型配置。它与 A4b-5/6 区分开：A4b-5/6 使用 frozen expert 后处理融合；A4c 重新训练 ToT/ToA 图像分支。

固定输入与训练设置：

```yaml
dataset:
  modalities: [ToT, ToA]

data:
  toa_transform: relative_minmax
  add_hit_mask: false

split:
  path: outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json
```

第一批模型：

```text
dual_stream_concat_aux
dual_stream_gmu_aux
toa_conditioned_film
```

其中 `dual_stream_concat_aux` 和 `dual_stream_gmu_aux` 会返回 auxiliary logits，默认 auxiliary loss 为：

```yaml
model:
  aux_loss:
    enabled: true
    weight_tot: 0.3
    weight_toa: 0.1
```

`dual_stream_gmu_aux` 会在 `metrics.json` / summary CSV 中记录 `gate_tot`、`gate_toa` 诊断均值；`toa_conditioned_film` 会记录 `film_gamma_abs`、`film_beta_abs` 诊断均值。

seed42 快速验证：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml --continue-on-error
python scripts/summarize.py --group a4c_end_to_end_bimodal_fusion_seed42 --out outputs/a4c_end_to_end_bimodal_fusion_seed42_runs.csv
```

正式三 seed：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion.yaml --continue-on-error
python scripts/summarize.py --group a4c_end_to_end_bimodal_fusion --out outputs/a4c_end_to_end_bimodal_fusion_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4c_end_to_end_bimodal_fusion_runs.csv --out outputs/a4c_end_to_end_bimodal_fusion_mean_std.csv
```

A4c 第一阶段三 seed 结果已经完成：

| 模型 | Val Acc | Test Acc | Test MAE | Test P90 | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dual_stream_concat_aux` | 69.90±1.34% | **72.10±1.35%** | **5.631±0.333** | 15.000±0.000 | 0.686±0.017 |
| `dual_stream_gmu_aux` | 70.20±0.67% | 71.94±0.51% | 5.721±0.009 | 15.000±0.000 | **0.691±0.009** |
| `toa_conditioned_film` | **70.43±0.95%** | 71.60±1.21% | 5.775±0.236 | 15.000±0.000 | 0.678±0.021 |

阶段结论：A4c 第一阶段没有显著刷新 A4b-5 的最高 Test Acc，但已经达到同一水平；更重要的是三个 A4c 模型均明显提升 Macro-F1，其中 `dual_stream_gmu_aux` 的类别均衡性最好。GMU gate 约 `77.6%` 偏向 ToT、`22.4%` 使用 ToA，支持 “ToT 为主模态，relative ToA 为辅助模态” 的解释。A4c 内部若按 Test Acc/MAE 选代表模型可看 `dual_stream_concat_aux`，若按机制解释和 Macro-F1 选代表模型优先看 `dual_stream_gmu_aux`。

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

### A4c-4 warm-started expert gate

`configs/experiments/a4c_warm_started_expert_gate.yaml` 是 A4c 第二批配置。它加载已有 expert checkpoint：

```text
Primary expert: A2 best ToT ResNet18 no-maxpool
Candidate expert: ToT + relative_minmax ToA, no mask ResNet18 no-maxpool
```

runner 会根据 `outputs/experiments/*/metadata.json` 自动按 `training.seed` 查找 checkpoint，因此配置中不需要写死时间戳目录。该配置比较 `freeze_experts=true` 与 `freeze_experts=false` 两个受控变体。

注意：A4c-4 里的 `gate_candidate` 是 candidate expert 的权重，不是单独 ToA 通道权重。candidate expert 的输入是 `ToT + relative_minmax ToA, no mask`。

seed42 快速验证：

```bash
cd ~/Timepix
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate_seed42.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate_seed42.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4c_warm_started_expert_gate_seed42 --out outputs/a4c_warm_started_expert_gate_seed42_runs.csv
```

正式三 seed：

```bash
cd ~/Timepix
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --dry-run
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --skip-existing --continue-on-error
python scripts/summarize.py --group a4c_warm_started_expert_gate --out outputs/a4c_warm_started_expert_gate_runs.csv
python scripts/aggregate_seeds.py --summary outputs/a4c_warm_started_expert_gate_runs.csv --out outputs/a4c_warm_started_expert_gate_mean_std.csv
```

A4c-4 三 seed 结果已经完成：

| 设置 | Val Acc | Test Acc | Test MAE | Test F1 | Candidate Gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `freeze_experts=true` | 70.50±1.00% | 71.84±1.88% | 5.905±0.332 | 0.660±0.015 | 65.11±24.63% |
| `freeze_experts=false` | 68.80±1.36% | 70.15±2.34% | 6.123±0.554 | 0.643±0.049 | 56.53±12.19% |

阶段结论：`freeze_experts=true` 优于 ToT primary，但不超过 A4b-5 gated late fusion，也不超过 A4c 第一阶段的 concat/GMU。`freeze_experts=false` 不稳定，说明加载已有 expert 后继续端到端微调容易破坏原有决策边界。A4c-4 适合作为 warm-start expert gate 对照，而不是当前最佳融合方案。

服务器 `tmux` 持久化运行：

```bash
cd ~/Timepix
tmux new -s a4c_warm_gate
```

进入 `tmux` 后运行：

```bash
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4c_warm_started_expert_gate --out outputs/a4c_warm_started_expert_gate_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a4c_warm_started_expert_gate_runs.csv --out outputs/a4c_warm_started_expert_gate_mean_std.csv
```

重新进入会话：

```bash
tmux attach -t a4c_warm_gate
```
