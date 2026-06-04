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
Particle_Source_3 -> E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1\dataset
ptype_stage1_full_am_co_sr_p_v1 -> E:\TimepixData\particle\datasets\particle_type_stage1_full_am_co_sr_p_v1
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
| `configs/datasets/particle_source_3.yaml` | 当前 Timepix3 放射源/粒子类别识别数据集，支持 `ToT` 与 `ToA`，类别名从文件夹自动提取。 |
| `configs/datasets/particle_ps3_totgmmk2_v1.yaml` | Particle/source 新 P 系列主线数据集，来自 `particle_source_label_cleaned_tot_toa_tot_gmm_k2_selected_v1`，支持 paired `ToT`/`ToA`，类别名从文件夹自动提取。 |
| `configs/datasets/particle_type_stage1_full_am_co_sr_p_v1.yaml` | Particle type/source 四分类数据集，来自 `particle_type_stage1_full_am_co_sr_p_v1`，当前类别为 `Am`、`Co`、`P`、`Sr`，类别名仍从文件夹自动提取。 |

论文数据分析默认使用全量 `Proton_C`，训练默认使用 `Proton_C_7`，二者不能混淆。

### 3.1 标签类型

`dataset.label_type` 控制标签解释方式：

| `label_type` | 用途 | 标签来源 | 指标 |
| --- | --- | --- | --- |
| `angle_folder` | Alpha / Proton 角度识别 | 数值角度文件夹，例如 `15`、`30`、`45`、`60` | accuracy、MAE、P90、Macro-F1、角度混淆等 |
| `categorical_folder` | Particle source / particle type 分类 | 普通类别文件夹，例如 `Am`、`Co60`、`Sr`，未来也可为 `Alpha`、`Beta`、`Gamma` | accuracy、balanced accuracy、Macro-F1、weighted-F1、per-class 指标 |

`categorical_folder` 默认自动按文件夹名生成 `class_names`。若后续需要固定类别顺序，可在 dataset config 中显式写：

```yaml
class_names: [Am, Co60, Sr]
```

普通类别任务不计算角度 `MAE`、`P90`，也不能使用 `gaussian` soft label、`emd`、`ce_expected_mae`、`ce_emd` 等角度有序损失。

### 3.2 P-series Particle/source experiments

P 系列是 Particle/source 分类的新主线。旧 `C1/C2` 使用早期 source-label 数据集，当前全部作为 deprecated diagnostic，不进入后续主结果。P 系列把数据集版本显式写入 dataset id、experiment group 和 split path，避免多版 particle 数据集混用。

当前数据集短名：

```text
ps3_totgmmk2_v1 = particle_source_label_cleaned_tot_toa_tot_gmm_k2_selected_v1
```

P1 是新数据集的模态诊断阶段。先只运行 `P1a ToT-only`，用于确认 ToT 单模态在提纯数据集上的基本可分性。

| 编号 | 配置文件 | 输入 | 模型 | 定位 |
| --- | --- | --- | --- | --- |
| P1a | `configs/experiments/p1a_ps3_totgmmk2_v1_tot_seed42.yaml` | `ToT` | `resnet18_no_maxpool` | 新数据集 ToT-only 诊断 |

P1a 服务器运行与汇总命令：

```bash
tmux new -s p1a_ps3_totgmmk2
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_source_label_cleaned_tot_toa_tot_gmm_k2_selected_v1/dataset
LOG=outputs/p1_ps3_totgmmk2_v1_modality_seed42_tmux.log

{
  echo "[P1a] start $(date)"
  $PY scripts/train.py --config configs/experiments/p1a_ps3_totgmmk2_v1_tot_seed42.yaml --data-root "$DATA"
  $PY scripts/summarize.py --group p1_ps3_totgmmk2_v1_modality_seed42 --out outputs/p1_ps3_totgmmk2_v1_modality_seed42_runs.csv
  echo "[P1a] done $(date)"
} 2>&1 | tee "$LOG"
```

P1a seed42 已完成。`ToT-only` 在 `ps3_totgmmk2_v1` 上取得 `Val Macro-F1=0.977`、`Test Macro-F1=0.978`，主要错误为 `Sr -> Co60`。该结果说明新提纯数据集上 ToT 单模态已经很强，但训练过程中 validation 多次塌陷后恢复，后续 P1/P2 仍需要关注训练稳定性。

#### P1lr ToT-only learning-rate stability diagnostic

P1lr 是 P1a 后的学习率稳定性诊断，不改变输入模态、模型、loss、split 或 seed，只比较较小学习率是否缓解 validation collapse。该阶段暂不加入 class weight / sampler，也不展开 `ToA`、concat 或 GMU。

| 编号 | 配置文件 | 搜索项 | 固定设置 |
| --- | --- | --- | --- |
| P1lr | `configs/experiments/p1lr_ps3_totgmmk2_v1_tot_lr_stability_seed42.yaml` | `learning_rate=[1e-4,5e-5,3e-5,1e-5]` | `ToT-only`, `epochs=40`, `patience=12`, `seed=42` |

服务器运行与汇总命令：

```bash
tmux new -s p1lr_ps3_totgmmk2
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_source_label_cleaned_tot_toa_tot_gmm_k2_selected_v1/dataset
LOG=outputs/p1lr_ps3_totgmmk2_v1_tot_lr_stability_seed42_tmux.log

{
  echo "[P1lr] start $(date)"
  $PY scripts/run_grid.py --config configs/experiments/p1lr_ps3_totgmmk2_v1_tot_lr_stability_seed42.yaml --data-root "$DATA" --continue-on-error
  $PY scripts/summarize.py --group p1lr_ps3_totgmmk2_v1_tot_lr_stability_seed42 --out outputs/p1lr_ps3_totgmmk2_v1_tot_lr_stability_seed42_runs.csv
  echo "[P1lr] done $(date)"
} 2>&1 | tee "$LOG"
```

分析时除 `Val/Test Macro-F1` 外，还要检查每个 run 的 `training_log.csv`：`val_macro_f1 < 0.8` 的 collapse epoch 数、best epoch 后是否持续塌陷、final 与 best 的 gap，以及 `Sr -> Co60` 混淆是否减少。

P1lr 已完成。`5e-5` 的 best validation/test 指标最高，但仍有 `8` 个 collapse epoch；`3e-5` 的 `Val/Test Macro-F1=0.980/0.979`，collapse 仅 `1` 次，best 后没有再次塌陷，且 `Sr -> Co60` 最少。因此后续 P1 模态诊断默认使用：

```yaml
training:
  learning_rate: 0.00003
  epochs: 40
  early_stopping_patience: 12
```

#### P1b four-class particle type/source modality diagnostic

P1b 是新四分类数据集 `particle_type_stage1_full_am_co_sr_p_v1` 的单 seed 模态/融合完整诊断。它不复用旧 `ps3_totgmmk2_v1` 的 split 或结果口径；旧 P1a/P1lr 只提供学习率稳定性经验。服务器目录已确认四个类别均有 paired `ToT` 和 `ToA`：

```text
Am: ToT=3683, ToA=3683
Co: ToT=36182, ToA=36182
P : ToT=25163, ToA=25163
Sr: ToT=12854, ToA=12854
```

固定设置：

| 项目 | 设置 |
| --- | --- |
| Dataset | `ptype_stage1_full_am_co_sr_p_v1` |
| Server data root | `/root/autodl-tmp/particle_type_stage1_full_am_co_sr_p_v1` |
| Task | `categorical_folder` classification |
| Primary metric | `val_macro_f1` |
| Loss | `cross_entropy + onehot + class_weight=balanced` |
| Backbone/stem | `resnet18_no_maxpool`, `conv1=2/1/0`, `dropout=0.1` |
| Training | `lr=3e-5`, `epochs=40`, `patience=12`, `seed=42` |
| Split | `outputs/splits/ptype_stage1_full_am_co_sr_p_v1_ToT-ToA_seed42_0.8_0.1_0.1.json` |

P1b 实验矩阵：

| 编号 | 配置文件 | 输入 | 模型 | 定位 |
| --- | --- | --- | --- | --- |
| P1b-a | `configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_tot_seed42.yaml` | `ToT` | `resnet18_no_maxpool` | ToT 单模态诊断 |
| P1b-b | `configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_rtoa_seed42.yaml` | `RToA` | `resnet18_no_maxpool` | ToA 相对时间单模态诊断 |
| P1b-c | `configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_input_concat_seed42.yaml` | `[ToT, RToA]` | `resnet18_no_maxpool` | 输入层 concat |
| P1b-d | `configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_dual_concat_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_concat_aux` | 双分支特征 concat |
| P1b-e | `configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_gmu_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_gmu_aux` | 双分支 GMU，目标模型诊断 |

服务器训练与汇总命令：

```bash
tmux new -s p1b_ptype_stage1
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_type_stage1_full_am_co_sr_p_v1
GROUP=p1b_ptype_stage1_full_am_co_sr_p_v1_modality_seed42
LOG=outputs/${GROUP}_tmux.log

{
  echo "[P1b] start $(date)"
  $PY scripts/train.py --config configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_tot_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_rtoa_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_input_concat_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_dual_concat_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1b_ptype_stage1_full_am_co_sr_p_v1_gmu_seed42.yaml --data-root "$DATA"
  $PY scripts/summarize.py --group "$GROUP" --out outputs/${GROUP}_runs.csv
  echo "[P1b] done $(date)"
} 2>&1 | tee "$LOG"
```

P1b 运行后在 `lr=3e-5` 下仍出现验证集准确率崩塌和明显震荡，实验已人工终止。该批次不生成正式 summary，不进入结果表；服务器中旧 group 下的 partial checkpoint 仅作为中断痕迹保留。后续使用全新 group `p1c_ptype_stage1_full_am_co_sr_p_v1_modality_lr1e5_seed42`，避免和半截 run 混合。

#### P1c four-class particle type/source modality diagnostic, lr=1e-5

P1c 是 P1b 的低学习率重跑。唯一训练策略变化是：

```yaml
training:
  learning_rate: 0.00001
```

其余设置保持 P1b 一致：`particle_type_stage1_full_am_co_sr_p_v1`、`categorical_folder`、balanced CE、`val_macro_f1`、`epochs=40`、`patience=12`、`seed=42`、shared `ToT-ToA` split。实验目的不是改变数据或模型定义，而是诊断 `lr=1e-5` 是否能缓解新四分类数据集上的验证集崩塌。

P1c 实验矩阵：

| 编号 | 配置文件 | 输入 | 模型 | 定位 |
| --- | --- | --- | --- | --- |
| P1c-a | `configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_tot_seed42.yaml` | `ToT` | `resnet18_no_maxpool` | ToT 单模态低学习率诊断 |
| P1c-b | `configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_rtoa_seed42.yaml` | `RToA` | `resnet18_no_maxpool` | ToA 相对时间单模态低学习率诊断 |
| P1c-c | `configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_input_concat_seed42.yaml` | `[ToT, RToA]` | `resnet18_no_maxpool` | 输入层 concat 低学习率诊断 |
| P1c-d | `configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_dual_concat_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_concat_aux` | 双分支特征 concat 低学习率诊断 |
| P1c-e | `configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_gmu_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_gmu_aux` | 双分支 GMU 低学习率诊断 |

服务器训练与汇总命令：

```bash
tmux new -s p1c_ptype_stage1_lr1e5
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_type_stage1_full_am_co_sr_p_v1
GROUP=p1c_ptype_stage1_full_am_co_sr_p_v1_modality_lr1e5_seed42
LOG=outputs/${GROUP}_tmux.log

{
  echo "[P1c] start $(date)"
  $PY scripts/train.py --config configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_tot_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_rtoa_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_input_concat_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_dual_concat_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1c_ptype_stage1_full_am_co_sr_p_v1_gmu_seed42.yaml --data-root "$DATA"
  $PY scripts/summarize.py --group "$GROUP" --out outputs/${GROUP}_runs.csv
  echo "[P1c] done $(date)"
} 2>&1 | tee "$LOG"
```

本地结果拉取命令：

```powershell
rclone copy autodl35610:/root/Timepix/outputs/ D:/Project/Timepix/outputs/ `
  --progress `
  --transfers 8 `
  --checkers 16 `
  --create-empty-src-dirs `
  --exclude "**/best_model.pth" `
  --exclude "**/last_checkpoint.pth" `
  --exclude "**/*.pt" `
  --log-file D:/Project/Timepix/outputs/rclone_autodl35610_pull.log `
  --log-level INFO
```

P1c seed42 已完成，结果已拉回本地。按 `val_macro_f1` 排序：

| 排名 | 模型/输入 | Val Acc | Val Macro-F1 | Test Acc | Test Macro-F1 | Best/Stop |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `dual_stream_concat_aux` | 96.11% | **0.9573** | **95.96%** | **0.9553** | 12/24 |
| 2 | `dual_stream_gmu_aux` | 95.90% | 0.9552 | 95.73% | 0.9532 | 9/21 |
| 3 | `ToT` | 95.70% | 0.9522 | 95.85% | 0.9541 | 10/22 |
| 4 | input concat `[ToT,RToA]` | 95.67% | 0.9522 | 95.53% | 0.9506 | 19/31 |
| 5 | `RToA` | 94.68% | 0.9374 | 94.65% | 0.9373 | 32/40 |

阶段判断：`lr=1e-5` 相比被终止的 P1b `lr=3e-5` 明显缓解 validation collapse，但 dual concat 和 GMU 在 epoch 20 附近仍有低谷。`dual_stream_concat_aux` 是当前 single-seed validation-selected 最强候选；`dual_stream_gmu_aux` 是接近的目标门控架构候选。后续若进入正式认证，优先只对这两组做 3 seed，不再扩大 RToA-only 或 input concat 网格。

#### P1d GMU learning-rate stability scan

P1d 只针对 `dual_stream_gmu_aux` 做学习率稳定性诊断。P1c 显示 `lr=1e-5` 已明显缓解 P1b 的严重崩塌，但 GMU 在 epoch 20 附近仍出现明显 validation 低谷；因此 P1d 不再扩大模型网格，只检查更小学习率是否能让 GMU 曲线更平滑。

固定设置：

| 项目 | 设置 |
| --- | --- |
| Dataset | `particle_type_stage1_full_am_co_sr_p_v1` |
| Input | `ToT + relative_minmax ToA` |
| Model | `dual_stream_gmu_aux` |
| Loss | `cross_entropy + class_weight=balanced` |
| Primary metric | `val_macro_f1` |
| Epochs / patience | `60 / 20` |
| Seed | `42` |
| Group | `p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr_stability_seed42` |

P1d 实验矩阵：

| 编号 | 配置文件 | learning rate | 目的 |
| --- | --- | ---: | --- |
| P1d-a | `configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr1e5_seed42.yaml` | `1e-5` | P1c GMU 设置复核，预算拉长 |
| P1d-b | `configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr5e6_seed42.yaml` | `5e-6` | 检查轻微降 lr 是否减少低谷 |
| P1d-c | `configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr3e6_seed42.yaml` | `3e-6` | P1lr 风格保守候选 |
| P1d-d | `configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr1e6_seed42.yaml` | `1e-6` | 极低 lr 欠拟合/稳定性下界 |

选择规则：

```text
1. 先排除有严重 collapse 的 lr。
2. 在稳定 lr 中选择 best Val Macro-F1 最高者。
3. 若 best Val Macro-F1 差距 < 0.005，优先选择 post-best drop 更小、collapse epochs 更少者。
4. 若稳定性和指标接近，选择较大的 lr，避免训练过慢或欠拟合。
```

服务器训练与汇总命令：

```bash
tmux new -s p1d_gmu_lr_stability
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_type_stage1_full_am_co_sr_p_v1
GROUP=p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr_stability_seed42
LOG=outputs/${GROUP}_tmux.log

{
  echo "[P1d] start $(date)"
  $PY scripts/train.py --config configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr1e5_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr5e6_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr3e6_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/p1d_ptype_stage1_full_am_co_sr_p_v1_gmu_lr1e6_seed42.yaml --data-root "$DATA"
  $PY scripts/summarize.py --group "$GROUP" --out outputs/${GROUP}_runs.csv
  echo "[P1d] done $(date)"
} 2>&1 | tee "$LOG"
```

本地结果拉取命令：

```powershell
rclone copy autodl35610:/root/Timepix/outputs/ D:/Project/Timepix/outputs/ `
  --progress `
  --transfers 8 `
  --checkers 16 `
  --create-empty-src-dirs `
  --exclude "**/best_model.pth" `
  --exclude "**/last_checkpoint.pth" `
  --exclude "**/*.pt" `
  --log-file D:/Project/Timepix/outputs/rclone_autodl35610_pull.log `
  --log-level INFO
```

### 3.3 Deprecated C-series Particle/source diagnostics

C1/C2 是早期 `Particle_Source_3` 数据集上的诊断实验。由于后续 particle 数据集经过多轮清洗和 GMM 选择，C1/C2 不再作为 P 系列主线结论，只保留其对 ToA/RToA 重要性和类别不均衡风险的诊断价值。

#### 3.3.1 C1 Particle source baseline

C1 是 `Particle_Source_3` 的首个 source-label 分类训练实验，使用 `categorical_folder`，类别名从顶层文件夹自动提取。当前类别为 `Am`、`Co60`、`Sr`；未来若数据集改为 `Alpha`、`Beta`、`Gamma`，同一框架仍按文件夹自动生成类别名。

C1 是 single-seed screening，固定 `training.seed=42` 和 shared split：

```text
outputs/splits/Particle_Source_3_ToT-ToA_seed42_0.8_0.1_0.1.json
```

由于 `Co60` 样本量明显高于 `Am` 和 `Sr`，C1 的 `task.primary_metric` 固定为 `val_macro_f1`，并同时报告 `balanced_accuracy`、`weighted_f1`、per-class F1 和 confusion matrix。

配置文件：

| 编号 | 配置文件 | 输入 | 模型 |
| --- | --- | --- | --- |
| C1a | `configs/experiments/c1a_particle_source_tot_seed42.yaml` | `ToT` | `resnet18_no_maxpool` |
| C1b | `configs/experiments/c1b_particle_source_rtoa_seed42.yaml` | `RToA` | `resnet18_no_maxpool` |
| C1c | `configs/experiments/c1c_particle_source_tot_rtoa_input_concat_seed42.yaml` | `[ToT, RToA]` | `resnet18_no_maxpool` |
| C1d | `configs/experiments/c1d_particle_source_tot_rtoa_dual_concat_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_concat_aux` |
| C1e | `configs/experiments/c1e_particle_source_tot_rtoa_gmu_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_gmu_aux` |

`RToA` 表示：

```yaml
data:
  toa_transform: relative_minmax
  add_hit_mask: false
```

服务器一键运行：

```bash
tmux new -s c1_particle
cd /root/Timepix
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_source_label_cleaned_tot_toa_v1/dataset

$PY scripts/train.py --config configs/experiments/c1a_particle_source_tot_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1b_particle_source_rtoa_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1c_particle_source_tot_rtoa_input_concat_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1d_particle_source_tot_rtoa_dual_concat_seed42.yaml --data-root $DATA && \
$PY scripts/train.py --config configs/experiments/c1e_particle_source_tot_rtoa_gmu_seed42.yaml --data-root $DATA && \
$PY scripts/summarize.py --group c1_particle_source_baseline_seed42 --out outputs/c1_particle_source_baseline_seed42_runs.csv
```

本地增量拉取结果：

```powershell
rclone copy autodl37655:/root/Timepix/outputs/ D:/Project/Timepix/outputs/ `
  --progress `
  --transfers 8 `
  --checkers 16 `
  --create-empty-src-dirs `
  --exclude "**/best_model.pth" `
  --exclude "**/last_checkpoint.pth" `
  --exclude "**/*.pt" `
  --log-file D:/Project/Timepix/outputs/rclone_autodl37655_pull.log `
  --log-level INFO
```

C1 seed42 阶段结果已经完成。按预设主指标 `val_macro_f1`，当前排序为：

| 编号 | 输入/模型 | Val Macro-F1 | Test Macro-F1 | 阶段判断 |
| --- | --- | ---: | ---: | --- |
| C1d | dual-stream concat aux | **0.820** | **0.824** | 当前最强候选 |
| C1c | input concat | 0.812 | 0.814 | 轻量融合备选 |
| C1b | RToA single modality | 0.799 | 0.804 | 证明 ToA/RToA 具有 source 判别力 |
| C1a | ToT single modality | 0.635 | 0.634 | `Sr` 基本失败 |
| C1e | GMU aux | 0.413 | 0.410 | 本轮训练异常，不进入主候选 |

C1 不作为最终结论。后续 C2 应优先处理类别不均衡和 `Sr` 少数类召回，而不是继续扩大架构网格。

#### 3.3.2 C2 Particle source weighted-CE stability run

C2 是 C1 后的稳定性复跑。它保留 C1 的五组模态/融合结构，但加入 train split 自动类别权重，降低学习率，并延长训练周期。C2 暂不引入 weighted sampler，避免一次改变过多变量。

固定变化：

| 项目 | C1 | C2 |
| --- | --- | --- |
| Loss | CE one-hot | CE one-hot + `class_weight: balanced` |
| Learning rate | `3e-4` | `1e-4` |
| Epochs | `15` | `30` |
| Early stopping patience | `5` | `8` |
| Primary metric | `val_macro_f1` | `val_macro_f1` |

配置文件：

| 编号 | 配置文件 | 输入 | 模型 |
| --- | --- | --- | --- |
| C2a | `configs/experiments/c2a_particle_source_tot_weighted_seed42.yaml` | `ToT` | `resnet18_no_maxpool` |
| C2b | `configs/experiments/c2b_particle_source_rtoa_weighted_seed42.yaml` | `RToA` | `resnet18_no_maxpool` |
| C2c | `configs/experiments/c2c_particle_source_tot_rtoa_input_concat_weighted_seed42.yaml` | `[ToT, RToA]` | `resnet18_no_maxpool` |
| C2d | `configs/experiments/c2d_particle_source_tot_rtoa_dual_concat_weighted_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_concat_aux` |
| C2e | `configs/experiments/c2e_particle_source_tot_rtoa_gmu_weighted_seed42.yaml` | `ToT branch + RToA branch` | `dual_stream_gmu_aux` |

服务器一键运行：

```bash
tmux new -s c2_particle
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_source_label_cleaned_tot_toa_v1/dataset
LOG=outputs/c2_particle_source_weighted_ce_seed42_tmux.log

{
  echo "[C2] start $(date)"
  $PY scripts/train.py --config configs/experiments/c2a_particle_source_tot_weighted_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/c2b_particle_source_rtoa_weighted_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/c2c_particle_source_tot_rtoa_input_concat_weighted_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/c2d_particle_source_tot_rtoa_dual_concat_weighted_seed42.yaml --data-root "$DATA"
  $PY scripts/train.py --config configs/experiments/c2e_particle_source_tot_rtoa_gmu_weighted_seed42.yaml --data-root "$DATA"
  $PY scripts/summarize.py --group c2_particle_source_weighted_ce_seed42 --out outputs/c2_particle_source_weighted_ce_seed42_runs.csv
  echo "[C2] done $(date)"
} 2>&1 | tee "$LOG"
```

本地增量拉取结果仍使用：

```powershell
rclone copy autodl37655:/root/Timepix/outputs/ D:/Project/Timepix/outputs/ `
  --progress `
  --transfers 8 `
  --checkers 16 `
  --create-empty-src-dirs `
  --exclude "**/best_model.pth" `
  --exclude "**/last_checkpoint.pth" `
  --exclude "**/*.pt" `
  --log-file D:/Project/Timepix/outputs/rclone_autodl37655_pull.log `
  --log-level INFO
```

C2 seed42 结果已完成。按主指标 `val_macro_f1`，C2 内部排序为：

| 编号 | 模型 / 输入 | Val Macro-F1 | Test Macro-F1 | 阶段判断 |
| --- | --- | ---: | ---: | --- |
| C2e | GMU aux / `ToT+RToA` | **0.797** | **0.801** | balanced CE 将 GMU 从 C1 塌缩中救回，但仍低于 C1c/C1d。 |
| C2c | input concat / `ToT+RToA` | 0.792 | 0.799 | C2 中轻量强候选，但低于 C1c。 |
| C2b | `RToA` single modality | 0.768 | 0.776 | 继续证明 RToA 有 source 判别力，但 balanced CE 后不如 C1b。 |
| C2d | dual-stream concat aux | 0.760 | 0.767 | 相比 C1d 明显下降。 |
| C2a | `ToT` single modality | 0.688 | 0.691 | `Sr` recall 被拉高，但 `Co60` 被明显牺牲。 |

C2 结论：`class_weight: balanced` 能缓解少数类与 GMU 塌缩问题，但对 `Co60` 主类惩罚过强，整体未超过 C1 的强候选。后续若继续 C3，应优先尝试更温和的类别权重，而不是叠加 weighted sampler。

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
| A7-1 | `dual_stream_gmu_aux + CE one-hot + main_5feat gated` | 已完成 three-seed |

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

A7 三 seed 汇总：

| 方案 | Val Acc | Val MAE | Val Macro-F1 | Test Acc | Test MAE | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A7-0 GMU_aux，复用 A4c | **70.20±0.67%** | **6.274±0.129** | 0.668±0.007 | **71.94±0.51%** | 5.721±0.009 | **0.691±0.009** |
| A7-1 GMU_aux + main_5feat gated | **70.20±0.61%** | 6.359±0.277 | **0.669±0.019** | 71.80±1.09% | **5.706±0.145** | 0.687±0.008 |

A7 结论：

- A7-1 与 A7-0 的 Val Acc 持平，但 Val MAE 更差，Val Macro-F1 仅有极小变化。
- A7-1 没有改善 30 deg 类别，反而降低 30 deg Val/Test F1；它对 45 deg/60 deg 有局部帮助但不稳定。
- 按预先 validation 规则，`main_5feat` 不进入 Alpha 最终多模态主模型。
- Alpha 最终端到端多模态主模型保持 `dual_stream_gmu_aux + ToT/relative_minmax ToA + CE one-hot + no handcrafted`。

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

B2c geometry three-seed confirmation：

```bash
tmux new -s b2c
cd /root/Timepix
python scripts/run_grid.py --config configs/experiments/b2c_proton_c7_geometry_handcrafted_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --dry-run && \
python scripts/run_grid.py --config configs/experiments/b2c_proton_c7_geometry_handcrafted_3seed.yaml --data-root /root/autodl-tmp/Proton_C_7 --skip-existing --continue-on-error && \
python scripts/summarize.py --group b2c_proton_c7_geometry_handcrafted_3seed --out outputs/b2c_proton_c7_geometry_handcrafted_3seed_runs.csv && \
python scripts/aggregate_seeds.py --summary outputs/b2c_proton_c7_geometry_handcrafted_3seed_runs.csv --out outputs/b2c_proton_c7_geometry_handcrafted_3seed_mean_std.csv
```

B2c 只验证 `active_pixel_count + bbox_fill_ratio`，并比较 `concat` 与 `gated`。不再加入 `ToT_density`，因为它在 B2a concat 中明显伤害结果，B2b gated 只证明能抑制坏特征，并未证明其有稳定增益。

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
configs/experiments/a1_resnet18_original_baseline.yaml
configs/experiments/a1_structure_adaptation.yaml
configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml
configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml
configs/experiments/a4c_warm_started_expert_gate_seed42.yaml
configs/experiments/b1_proton_c7_resnet18_tot_best_3seed.yaml
configs/experiments/b1_proton_resnet18_tot_lr_batch.yaml
configs/experiments/a5b_alpha_handcrafted_group_ablation_TEMPLATE.yaml
configs/experiments/a5c_alpha_handcrafted_fusion_mode_TEMPLATE.yaml
configs/experiments/a5c_alpha_handcrafted_only_TEMPLATE.yaml
configs/experiments/a5d_alpha_handcrafted_best_3seed_TEMPLATE.yaml
configs/experiments/b2_proton_c7_handcrafted_transfer_TEMPLATE.yaml
configs/experiments/alpha_resnet18_tot_toa.yaml
configs/experiments/alpha_resnet18_tot_handcrafted_concat.yaml
configs/experiments/alpha_resnet18_tot_handcrafted_gated.yaml
configs/experiments/proton_resnet18_tot.yaml
configs/experiments/compare_models.yaml
configs/experiments/compare_losses.yaml
configs/experiments/compare_mixed_precision.yaml
```

说明：

- `b1_proton_c7_resnet18_tot_best_3seed.yaml` 使用 `early_stopping_patience=5`，只作为 Proton_C_7 早停过激诊断；正式 B1-best 使用 `b1_proton_c7_resnet18_tot_best_patience8_3seed.yaml`。
- `a1_*`、`compare_mixed_precision.yaml` 是已完成的结构/训练策略验证配置，当前不再重复运行。
- `alpha_resnet18_tot_toa.yaml`、`alpha_resnet18_tot_handcrafted_*.yaml`、`proton_resnet18_tot.yaml` 是早期过渡配置，正式实验分别使用 A4/A5/B 系列配置。
- `compare_models.yaml`、`compare_losses.yaml` 是早期通用网格配置；正式模型和 loss 实验分别使用 A3、A6、B3 的编号配置。
