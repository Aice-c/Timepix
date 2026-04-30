# Timepix 配置与命令索引

本文档是 `configs/` 目录的当前权威使用说明，重点记录配置层级、路径规则、正式训练命令与汇总命令。旧版长文档已归档为 `configs/README.old.md`；旧文档保留历史命令细节和阶段性结果，但不再作为当前实验状态判断依据。

实验目的、关键结果和设计决策的权威记录见：

```text
agent/EXPERIMENT_LOG.md
```

## 一、目录结构

```text
configs/datasets/      数据集事实：名称、路径、可用模态、输入尺寸
configs/experiments/   训练/对比实验配置
configs/search/        Optuna/TPE 超参数搜索配置
```

训练代码主链路为：

```text
configs/ -> scripts/ -> timepix/ -> outputs/
```

legacy `Program/` 不再用于新实验。

## 二、数据路径规则

推荐在服务器运行时显式传入具体数据集路径：

```bash
python scripts/train.py --config <config.yaml> --data-root /root/autodl-tmp/Alpha_100
python scripts/run_grid.py --config <config.yaml> --data-root /root/autodl-tmp/Proton_C_7
```

本地 Windows 路径仅用于 dry-run、checkpoint 评估或论文分析：

```text
Alpha_100  -> D:\Project\Timepix\Data\Alpha_100
Proton_C   -> E:\C1Analysis\Proton_C
Proton_C_7 -> E:\C1Analysis\Proton_C_7
```

训练/评估脚本的 `--data-root` 指向具体数据集目录；数据分析脚本的 `--data-root` 指向包含数据集文件夹的父目录。

## 三、正式数据集配置

| 配置 | 当前用途 |
| --- | --- |
| `configs/datasets/alpha_100.yaml` | 当前 Alpha 正式训练主线，支持 `ToT` 与 `ToA`。 |
| `configs/datasets/alpha_50.yaml` | 历史/对照数据集，不作为当前正式主线。 |
| `configs/datasets/alpha.yaml` | 兼容别名，指向 `Alpha_100`。 |
| `configs/datasets/proton_c_7.yaml` | 当前 Proton/C 七分类训练主线，只支持 `ToT`。 |
| `configs/datasets/proton_c.yaml` | 兼容入口，指向 `Proton_C_7`；新训练配置不再使用旧名。 |

论文数据分析默认使用全量 `Proton_C`，训练默认使用 `Proton_C_7`，二者不能混淆。

## 四、通用运行规范

### 4.1 单个训练

```bash
python scripts/train.py \
  --config configs/experiments/alpha_resnet18_tot.yaml \
  --data-root /root/autodl-tmp/Alpha_100
```

### 4.2 网格训练

```bash
python scripts/run_grid.py --config <config.yaml> --data-root <dataset_root> --dry-run
python scripts/run_grid.py --config <config.yaml> --data-root <dataset_root> --skip-existing --continue-on-error
```

### 4.3 标准汇总

```bash
python scripts/summarize.py --group <experiment_group> --out outputs/<name>_runs.csv
python scripts/aggregate_seeds.py --summary outputs/<name>_runs.csv --out outputs/<name>_mean_std.csv
```

单 seed screening 只需要 `summarize.py`；three-seed 正式验证需要同时运行 `aggregate_seeds.py`。

### 4.4 服务器持久化

正式服务器训练建议使用 `tmux`：

```bash
tmux new -s <session_name>
cd /root/Timepix
```

断开但保持运行：

```text
Ctrl+b 然后按 d
```

重新进入：

```bash
tmux attach -t <session_name>
```

## 五、Alpha 主线配置与命令

### 5.1 A2 超参数搜索

```bash
cd /root/Timepix
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml --dry-run
python scripts/search_hparams.py --config configs/search/a2_alpha_resnet18_tot_training.yaml
```

A2 best base：

```text
configs/experiments/alpha_tot_a2_best_base.yaml
```

### 5.2 A2-best three-seed baseline

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a2_best_3seed --out outputs/a2_best_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a2_best_3seed_runs.csv --out outputs/a2_best_3seed_mean_std.csv
```

### 5.3 A3 backbone comparison

正式 three-seed：

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a3_backbone_comparison --out outputs/a3_backbone_comparison_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a3_backbone_comparison_runs.csv --out outputs/a3_backbone_comparison_mean_std.csv
```

快速 seed42：

```bash
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run
python scripts/run_grid.py --config configs/experiments/a3_backbone_comparison_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error
python scripts/summarize.py --group a3_backbone_comparison_seed42 --out outputs/a3_backbone_comparison_seed42_runs.csv
```

### 5.4 A4 modality comparison

正式 three-seed：

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4_modality_comparison --out outputs/a4_modality_comparison_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a4_modality_comparison_runs.csv --out outputs/a4_modality_comparison_mean_std.csv
```

快速 seed42：

```bash
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run
python scripts/run_grid.py --config configs/experiments/a4_modality_comparison_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error
python scripts/summarize.py --group a4_modality_comparison_seed42 --out outputs/a4_modality_comparison_seed42_runs.csv
```

### 5.5 A4b ToA transform

seed42 screening：

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4b_toa_transform_seed42 --out outputs/a4b_toa_transform_seed42_runs.csv
```

three-seed version：

```bash
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4b_toa_transform.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4b_toa_transform --out outputs/a4b_toa_transform_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a4b_toa_transform_runs.csv --out outputs/a4b_toa_transform_mean_std.csv
```

### 5.6 A4b selector/gate diagnostic scripts

A4b-3a ToT seed-control oracle：

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

A4b-3b ToT vs `relative_minmax/no mask`：

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

A4b-5 three-seed gated late fusion：

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

A4b-6 three-seed residual gated fusion：

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

### 5.7 A4c end-to-end bimodal fusion

A4c-1/2/3：

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4c_end_to_end_bimodal_fusion.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4c_end_to_end_bimodal_fusion --out outputs/a4c_end_to_end_bimodal_fusion_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a4c_end_to_end_bimodal_fusion_runs.csv --out outputs/a4c_end_to_end_bimodal_fusion_mean_std.csv
```

A4c-4 warm-started expert gate：

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a4c_warm_started_expert_gate.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a4c_warm_started_expert_gate --out outputs/a4c_warm_started_expert_gate_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a4c_warm_started_expert_gate_runs.csv --out outputs/a4c_warm_started_expert_gate_mean_std.csv
```

### 5.8 A5 handcrafted scalar features

A5a screening：

```bash
cd /root/Timepix
python scripts/screen_handcrafted_features.py \
  --config configs/experiments/a5a_alpha_handcrafted_screening.yaml \
  --data-root /root/autodl-tmp/Alpha_100 \
  --out-dir outputs/a5a_alpha_handcrafted_screening \
  --name a5a_alpha_handcrafted_screening \
  --n-repeats 30
```

A5b concat pilot：

```bash
python scripts/run_grid.py --config configs/experiments/a5b_alpha_handcrafted_group_ablation.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a5b_alpha_handcrafted_group_ablation.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a5b_alpha_handcrafted_group_ablation --out outputs/a5b_alpha_handcrafted_group_ablation_runs.csv
```

A5c gated diagnostic：

```bash
python scripts/run_grid.py --config configs/experiments/a5c_alpha_handcrafted_gated_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a5c_alpha_handcrafted_gated_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a5c_alpha_handcrafted_gated_seed42 --out outputs/a5c_alpha_handcrafted_gated_seed42_runs.csv
```

A5d three-seed verification：

```bash
python scripts/run_grid.py --config configs/experiments/a5d_alpha_handcrafted_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a5d_alpha_handcrafted_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a5d_alpha_handcrafted_gated_3seed --out outputs/a5d_alpha_handcrafted_gated_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a5d_alpha_handcrafted_gated_3seed_runs.csv --out outputs/a5d_alpha_handcrafted_gated_3seed_mean_std.csv
```

### 5.9 A6 ordinal loss screening

A6a 已完成。CE one-hot baseline 复用 A2-best，不在 A6a 中重跑。

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a6a_alpha_tot_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a6a_alpha_tot_ordinal_loss_seed42 --out outputs/a6a_alpha_tot_ordinal_loss_seed42_runs.csv
```

A6a 主结果：

| 方法 | Val Acc | Val MAE | Val Macro-F1 | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 CE onehot baseline | **69.53%** | 6.279 | 0.630 | 70.48% | 5.964 | 0.646 |
| Gaussian sigma=5 | 68.83% | 6.414 | 0.620 | 69.38% | 6.054 | 0.628 |
| Gaussian sigma=7.5 | 68.33% | 6.533 | 0.615 | 70.68% | 5.949 | 0.643 |
| Gaussian sigma=10 | 68.63% | 6.414 | 0.612 | **71.17%** | **5.755** | 0.642 |
| CE+ExpectedMAE lambda=0.02 | 69.03% | 6.399 | 0.629 | 69.98% | 5.934 | **0.648** |
| CE+ExpectedMAE lambda=0.05 | 69.03% | 6.489 | 0.614 | 70.58% | 5.994 | 0.636 |
| CE+ExpectedMAE lambda=0.10 | 68.53% | 6.489 | 0.622 | 69.09% | 6.113 | 0.629 |
| CE+EMD lambda=0.02 | **69.53%** | **6.264** | **0.636** | 69.68% | 5.964 | 0.641 |
| CE+EMD lambda=0.05 | 69.13% | 6.444 | 0.605 | 70.38% | 5.964 | 0.621 |
| CE+EMD lambda=0.10 | 68.53% | 6.533 | 0.586 | 70.68% | 5.994 | 0.621 |

阶段判断：

- CE one-hot baseline 复用 A2-best，不在 A6a 中重跑。
- 按 validation selection，A6a-main 候选是 `CE+EMD lambda=0.02`。它与 A2 baseline Val Acc 持平，Val MAE 与 Val Macro-F1 更好。
- 该收益很弱，属于 tie-break 级别，不是 Proton B3b 那样的强正结果。
- `CE+ExpectedMAE lambda=0.02` 仅作为 A6a 结果解释中的诊断点；它对 test Macro-F1 和 30 deg test recall/F1 更好，但不能作为 validation-selected 主模型，因此不进入 A6b。
- A6b 收窄为只验证 validation-selected `CE+EMD lambda=0.02`。
- Gaussian soft label 不进入 A6b。`sigma=10` 虽然 test accuracy 最高，但 validation 不支持，不能用 test 反选。
- A6b 已完成；CE baseline 继续复用 A2-best three-seed。
- A6b 结果显示 `CE+EMD lambda=0.02` 不稳定且弱于 A2 CE baseline，因此 A6c 不推进，不把该 loss 迁移到 `dual_stream_gmu_aux`。

A6b `CE+EMD lambda=0.02` three-seed：

```bash
tmux new -s a6b
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a6b_alpha_tot_ce_emd_0p02_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a6b_alpha_tot_ce_emd_0p02_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a6b_alpha_tot_ce_emd_0p02_3seed --out outputs/a6b_alpha_tot_ce_emd_0p02_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a6b_alpha_tot_ce_emd_0p02_3seed_runs.csv --out outputs/a6b_alpha_tot_ce_emd_0p02_3seed_mean_std.csv
```

A6b 三 seed 汇总：

| 方法 | Val Acc | Val MAE | Val P90 | Val Macro-F1 | Test Acc | Test MAE | Test P90 | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A2 CE baseline | **69.03±0.46%** | **6.424±0.127** | 25.0±8.66 | **0.622±0.007** | **70.44±0.15%** | **5.949±0.068** | **15.0±0.0** | **0.636±0.009** |
| A6b CE+EMD lambda=0.02 | 68.33±1.15% | 6.618±0.424 | 25.0±8.66 | 0.609±0.034 | 69.62±0.80% | 6.143±0.336 | 20.0±8.66 | 0.623±0.034 |

A6b 结论：

- A6a 中 `CE+EMD lambda=0.02` 的 seed42 tie-break 优势不稳定；seed43 明显退化。
- 按 validation 指标，A2 CE baseline 全面优于 A6b。
- `CE+EMD lambda=0.02` 未改善 30 deg 困难类别，且拉低 60 deg 类别表现。
- Alpha-ToT 后续继续使用 A2 CE one-hot；A6c 不推进。

### 5.10 A7 final multimodal + handcrafted confirmation

A7 只回答一个问题：在最终端到端多模态架构 `A4c-2 dual_stream_gmu_aux` 上，A5 选出的五维物理标量 `main_5feat` 是否还有额外补充价值。A7 loss 固定为 `CE one-hot`，不再运行 `GMU + CE+EMD` 或其他 loss/feature/architecture 网格。

对照关系：

| 编号 | 设置 | 状态 |
| --- | --- | --- |
| A7-0 | `dual_stream_gmu_aux + CE one-hot + no handcrafted` | 复用 A4c GMU three-seed |
| A7-1 | `dual_stream_gmu_aux + CE one-hot + main_5feat gated` | 新运行 three-seed |

`main_5feat`：

```text
active_pixel_count
bbox_fill_ratio
ToT_density
ToA_span
ToA_major_axis_corr_abs
```

服务器命令：

```bash
tmux new -s a7
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/a7_final_gmu_main5feat_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --dry-run && \
python scripts/run_grid.py --config configs/experiments/a7_final_gmu_main5feat_gated_3seed.yaml --data-root /root/autodl-tmp/Alpha_100 --skip-existing --continue-on-error && \
python scripts/summarize.py --group a7_final_gmu_main5feat_gated_3seed --out outputs/a7_final_gmu_main5feat_gated_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/a7_final_gmu_main5feat_gated_3seed_runs.csv --out outputs/a7_final_gmu_main5feat_gated_3seed_mean_std.csv
```

选择规则：

- 只用 validation 判断是否把 `main_5feat` 纳入最终模型。
- Primary: Val Acc。
- Tie-break: Val MAE 更低，其次 Val Macro-F1 更高。
- Test 只用于最终泛化报告，不能反向选择 A7-1。

## 六、Proton_C_7 主线配置与命令

### 6.1 B1-1 learning rate × batch size

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_lr_batch_ep25 --out outputs/b1_proton_c7_resnet18_tot_lr_batch_ep25_runs.csv
```

### 6.2 B1-2 weight decay

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_weight_decay_ep25 --out outputs/b1_proton_c7_resnet18_tot_weight_decay_ep25_runs.csv
```

### 6.3 B1-best patience=8

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b1_proton_c7_resnet18_tot_best_patience8_3seed --out outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_runs.csv --out outputs/b1_proton_c7_resnet18_tot_best_patience8_3seed_mean_std.csv
```

### 6.4 B2 handcrafted diagnostics

B2a concat：

```bash
python scripts/run_grid.py --config configs/experiments/b2_proton_c7_handcrafted_lowcorr_seed42.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b2_proton_c7_handcrafted_lowcorr_seed42.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b2_proton_c7_handcrafted_lowcorr_seed42 --out outputs/b2_proton_c7_handcrafted_lowcorr_seed42_runs.csv
```

B2b gated：

```bash
python scripts/run_grid.py --config configs/experiments/b2_proton_c7_handcrafted_gated_seed42.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b2_proton_c7_handcrafted_gated_seed42.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b2_proton_c7_handcrafted_gated_seed42 --out outputs/b2_proton_c7_handcrafted_gated_seed42_runs.csv
```

B2c 暂不优先推进；若后期需要三 seed 确认，使用：

```text
configs/experiments/b2_proton_c7_handcrafted_transfer_TEMPLATE.yaml
```

### 6.5 B3 ordinal loss

B3a seed42 screening：

```bash
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/b3a_proton_c7_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b3a_proton_c7_ordinal_loss_seed42.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b3a_proton_c7_ordinal_loss_seed42 --out outputs/b3a_proton_c7_ordinal_loss_seed42_runs.csv
```

B3b-main `CE+ExpectedMAE lambda=0.05`：

```bash
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_expected_mae_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b3b_proton_c7_expected_mae_3seed --out outputs/b3b_proton_c7_expected_mae_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b3b_proton_c7_expected_mae_3seed_runs.csv --out outputs/b3b_proton_c7_expected_mae_3seed_mean_std.csv
```

B3b optional `CE+EMD lambda=0.05`：

```bash
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_ce_emd_optional_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b3b_proton_c7_ce_emd_optional_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b3b_proton_c7_ce_emd_optional_3seed --out outputs/b3b_proton_c7_ce_emd_optional_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b3b_proton_c7_ce_emd_optional_3seed_runs.csv --out outputs/b3b_proton_c7_ce_emd_optional_3seed_mean_std.csv
```

## 七、数据分析命令

数据分析链路独立于训练实验。

```bash
python scripts/analyze_datasets.py \
  --data-root Data \
  --output-root outputs/data_analysis \
  --datasets Alpha_100 Proton_C \
  --sample-cap-plot 5000 \
  --seed 42

python scripts/analyze_resolution_limit.py \
  --data-root Data \
  --dataset Proton_C \
  --angles 80 82 84 86 88 90 \
  --modality ToT \
  --output-root outputs/resolution_limit \
  --sample-cap-plot 5000 \
  --sample-cap-ml 10000 \
  --seeds 42 43 44 45 46

python scripts/make_analysis_report.py \
  --data-analysis-root outputs/data_analysis \
  --resolution-root outputs/resolution_limit \
  --out outputs/analysis_report.md
```

## 八、历史与模板配置

以下文件保留可追溯性或作为模板，不代表当前正式实验结果：

```text
configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml
configs/experiments/a5b_alpha_handcrafted_group_ablation_TEMPLATE.yaml
configs/experiments/a5c_alpha_handcrafted_fusion_mode_TEMPLATE.yaml
configs/experiments/a5c_alpha_handcrafted_only_TEMPLATE.yaml
configs/experiments/a5d_alpha_handcrafted_best_3seed_TEMPLATE.yaml
configs/experiments/b2_proton_c7_handcrafted_transfer_TEMPLATE.yaml
```

其中 `b1_proton_c7_resnet18_tot_best_3seed.yaml` 使用 `early_stopping_patience=5`，只作为 Proton_C_7 早停过激诊断；正式 B1-best 使用 patience=8 版本。
