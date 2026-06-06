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

P1d seed42 已完成，结果已拉回本地。主结果如下：

| Learning rate | Val Macro-F1 | Post-best drop | Collapse `<0.85` | Last10 median | Test Macro-F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-5` | **0.9576** | 0.3586 | 5 | 0.9479 | 0.9549 |
| `5e-6` | 0.9567 | 0.3488 | 3 | 0.9441 | 0.9554 |
| `3e-6` | 0.9568 | 0.0513 | 2 | 0.9502 | **0.9556** |
| `1e-6` | 0.9553 | **0.0016** | 1 | **0.9538** | 0.9523 |

阶段判断：`1e-5` / `5e-6` best 指标略高或接近，但 best 后仍有严重 collapse；`1e-6` 最稳但收敛很晚且指标略低。`3e-6` 在 best Val Macro-F1 和稳定性之间最平衡，建议作为后续 P 系列 GMU 多 seed 默认学习率。

#### P2a new particle type/source dataset, five-model 3-seed comparison

P2a 使用新的四分类 particle/source 数据集 `particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3`。该数据集不同于 P1b/P1c/P1d 的 `particle_type_stage1_full_am_co_sr_p_v1`，因此单独编号为 P2a，不与旧数据集结果混合汇总。

固定设置：

| 项目 | 设置 |
| --- | --- |
| Dataset config | `configs/datasets/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3.yaml` |
| Server data root | `/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3` |
| Label type | `categorical_folder`，类别名从文件夹自动提取 |
| Loss | `cross_entropy + class_weight=balanced` |
| Primary metric | `val_macro_f1` |
| Learning rate | `3e-6` |
| Epochs / patience | `50 / 15` |
| Seeds | `42, 43, 44` |
| Group | `p2a_ptype_stage1_gmm02_p_v3_modality_lr3e6_3seed` |

P2a 实验矩阵：

| 编号 | 配置文件 | 输入 | 模型 |
| --- | --- | --- | --- |
| P2a-a | `configs/experiments/p2a_ptype_stage1_gmm02_p_v3_tot_3seed.yaml` | `ToT` | `resnet18_no_maxpool` |
| P2a-b | `configs/experiments/p2a_ptype_stage1_gmm02_p_v3_rtoa_3seed.yaml` | `RToA` | `resnet18_no_maxpool` |
| P2a-c | `configs/experiments/p2a_ptype_stage1_gmm02_p_v3_input_concat_3seed.yaml` | `[ToT, RToA]` | `resnet18_no_maxpool` |
| P2a-d | `configs/experiments/p2a_ptype_stage1_gmm02_p_v3_dual_concat_3seed.yaml` | `ToT branch + RToA branch` | `dual_stream_concat_aux` |
| P2a-e | `configs/experiments/p2a_ptype_stage1_gmm02_p_v3_gmu_3seed.yaml` | `ToT branch + RToA branch` | `dual_stream_gmu_aux` |

服务器训练与汇总命令：

```bash
tmux new -s p2a_ptype_stage1_gmm02_p_v3_3seed
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3
GROUP=p2a_ptype_stage1_gmm02_p_v3_modality_lr3e6_3seed
LOG=outputs/${GROUP}_tmux.log
CONFIGS=(
  configs/experiments/p2a_ptype_stage1_gmm02_p_v3_tot_3seed.yaml
  configs/experiments/p2a_ptype_stage1_gmm02_p_v3_rtoa_3seed.yaml
  configs/experiments/p2a_ptype_stage1_gmm02_p_v3_input_concat_3seed.yaml
  configs/experiments/p2a_ptype_stage1_gmm02_p_v3_dual_concat_3seed.yaml
  configs/experiments/p2a_ptype_stage1_gmm02_p_v3_gmu_3seed.yaml
)

{
  echo "[P2a] start $(date)"
  test -d "$DATA"
  find "$DATA" -maxdepth 2 -type d | sort | head -40
  for cfg in "${CONFIGS[@]}"; do
    $PY scripts/run_grid.py --config "$cfg" --data-root "$DATA" --dry-run
  done
  for cfg in "${CONFIGS[@]}"; do
    $PY scripts/run_grid.py --config "$cfg" --data-root "$DATA" --skip-existing --continue-on-error
  done
  $PY scripts/summarize.py --group "$GROUP" --out outputs/${GROUP}_runs.csv
  $PY scripts/aggregate_seeds.py --summary outputs/${GROUP}_runs.csv --out outputs/${GROUP}_mean_std.csv
  echo "[P2a] done $(date)"
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

P2a 执行备注与结果：

- 首次启动时服务器数据仍为 `dataset/<class>/<modality>` 形式，导致 `DATA` 指向总目录时无法找到 `ToT`/`ToA`。该错误未进入有效训练，只生成 0-row summary。
- 用户修正数据集布局后，主控验证 `ToT`、`ToA`、`ToT+ToA` 均可配对出 `75113` 个样本，类别数量为 `Am=3683`、`Co=36182`、`P=25163`、`Sr=10085`。
- 正式重启后 `15/15` 个 run 成功完成，`runs.csv` 为 `15` 行，`mean_std.csv` 为 `5` 行。

P2a 三 seed 主结果：

| Model | Val Macro-F1 | Val Acc | Val Bal Acc | Test Macro-F1 | Test Acc | Test Bal Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dual_stream_gmu_aux` | **0.9880±0.0005** | **99.01±0.04%** | **0.9845±0.0010** | 0.9825±0.0004 | 98.56±0.04% | **0.9774±0.0004** |
| `dual_stream_concat_aux` | **0.9880±0.0006** | **99.01±0.05%** | 0.9842±0.0015 | **0.9828±0.0007** | **98.60±0.05%** | 0.9770±0.0019 |
| `ToT only` | 0.9870±0.0003 | 98.93±0.03% | 0.9839±0.0011 | 0.9818±0.0015 | 98.54±0.10% | 0.9762±0.0022 |
| `[ToT, RToA] input concat` | 0.9802±0.0002 | 98.41±0.02% | 0.9784±0.0001 | 0.9788±0.0024 | 98.31±0.19% | 0.9743±0.0024 |
| `RToA only` | 0.9690±0.0009 | 97.77±0.04% | 0.9641±0.0007 | 0.9633±0.0013 | 97.36±0.04% | 0.9565±0.0019 |

稳定性摘要：

| Model | Best-after drop mean/max | Best-after min Val Macro-F1 | Collapse `<0.90` |
| --- | ---: | ---: | ---: |
| `dual_stream_gmu_aux` | 0.0303 / 0.0515 | 0.9363 | 0 |
| `dual_stream_concat_aux` | 0.0268 / 0.0336 | 0.9539 | 0 |
| `[ToT, RToA] input concat` | 0.0124 / 0.0150 | 0.9650 | 0 |
| `ToT only` | 0.1233 / 0.1861 | 0.8013 | 13 |
| `RToA only` | 0.0170 / 0.0227 | 0.9466 | 0 |

阶段判断：

- `dual_stream_gmu_aux` 与 `dual_stream_concat_aux` 是 P2a 最强双模态候选；二者 `Val Macro-F1` 几乎并列。
- `ToT only` 指标接近，但 validation collapse 明显，不宜只看 best 指标。
- `RToA only` 明显弱于 ToT 和双模态，说明 RToA 单独不是主模型。
- `[ToT, RToA] input concat` 稳定但上限低于双分支，继续支持双分支结构优先。
- 若推进 P2b，建议聚焦 `dual_stream_gmu_aux` 与 `dual_stream_concat_aux`，并重点分析 `Sr<->Co` 混淆。

#### P2b GMU hyperparameter screening

P2b 只针对 P2a 中的目标模型 `dual_stream_gmu_aux` 做小范围超参数筛选。P2a 已证明 GMU 与 dual concat 是最强双模态候选；本轮不再扩大输入模态或学习率网格，只检查 GMU 自身的 gate bias、auxiliary loss 和 dropout。P2a GMU 作为 baseline 复用，不重跑。

固定设置：

| 项目 | 设置 |
| --- | --- |
| Dataset | `particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3` |
| Server data root | `/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3` |
| Input | `ToT + relative_minmax ToA` |
| Model | `dual_stream_gmu_aux` |
| Loss | `cross_entropy`, `label_encoding=onehot`, `class_weight=balanced` |
| Primary metric | `val_macro_f1` |
| LR / Epoch / Patience | `3e-6` / `50` / `15` |
| Batch / WD / Scheduler | `64` / `1e-4` / `cosine`, `eta_min=1e-7` |
| Seed | `42` screening only |
| Baseline | P2a GMU: `bias=2.0`, `aux_tot=0.3`, `aux_toa=0.1`, `dropout=0.1` |

P2b 实验矩阵：

| 编号 | Config | 变量 | 设置 |
| --- | --- | --- | --- |
| P2b-0 | reuse P2a GMU | baseline | 不重跑 |
| P2b-1a | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_bias1_seed42.yaml` | gate bias | `init_bias_to_tot=1.0` |
| P2b-1b | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_bias3_seed42.yaml` | gate bias | `init_bias_to_tot=3.0` |
| P2b-2a | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_none_seed42.yaml` | aux loss | disabled |
| P2b-2b | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_light_seed42.yaml` | aux loss | `weight_tot=0.1`, `weight_toa=0.05` |
| P2b-2c | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_balanced_seed42.yaml` | aux loss | `weight_tot=0.3`, `weight_toa=0.3` |
| P2b-2d | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_totstrong_seed42.yaml` | aux loss | `weight_tot=0.5`, `weight_toa=0.1` |
| P2b-3a | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_dropout005_seed42.yaml` | dropout | `0.05` |
| P2b-3b | `configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_dropout02_seed42.yaml` | dropout | `0.2` |

服务器训练与汇总命令：

```bash
tmux new -s p2b_ptype_stage1_gmm02_p_v3_gmu_hparam_seed42
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3
GROUP=p2b_ptype_stage1_gmm02_p_v3_gmu_hparam_seed42
LOG=outputs/${GROUP}_tmux.log
CONFIGS=(
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_bias1_seed42.yaml
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_bias3_seed42.yaml
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_none_seed42.yaml
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_light_seed42.yaml
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_balanced_seed42.yaml
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_aux_totstrong_seed42.yaml
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_dropout005_seed42.yaml
  configs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_dropout02_seed42.yaml
)

{
  echo "[P2b] start $(date)"
  git rev-parse --short HEAD
  test -d "$DATA"
  for cfg in "${CONFIGS[@]}"; do
    echo "[P2b] dry-run $cfg"
    $PY scripts/run_grid.py --config "$cfg" --data-root "$DATA" --dry-run
  done
  for cfg in "${CONFIGS[@]}"; do
    echo "[P2b] run $cfg $(date)"
    $PY scripts/run_grid.py --config "$cfg" --data-root "$DATA" --skip-existing --continue-on-error
  done
  echo "[P2b] summarize $(date)"
  $PY scripts/summarize.py --group "$GROUP" --out outputs/${GROUP}_runs.csv
  $PY scripts/aggregate_seeds.py --summary outputs/${GROUP}_runs.csv --out outputs/${GROUP}_mean_std.csv
  echo "[P2b] done $(date)"
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

P2b 选择规则：

- 先按 `Val Macro-F1` 与 P2a GMU baseline 对比。
- 若差距 `<0.001`，看 `Val Balanced Acc`。
- 再看 `Sr` F1/recall 与 `Sr<->Co` 混淆。
- 最后看稳定性：best 后最大 drop、best 后最低 `val_macro_f1`、collapse `<0.90` 次数。
- 若没有设置明显优于 P2a GMU，则不进入 P2c，沿用 P2a GMU；若有明显胜出设置，再做 P2c three-seed certification。

P2b 执行备注与结果：

- 服务器 `35610` 已完成 8/8 个 run，tmux session `p2b_ptype_stage1_gmm02_p_v3_gmu_hparam_seed42` 正常退出。
- 结果文件：

```text
outputs/p2b_ptype_stage1_gmm02_p_v3_gmu_hparam_seed42_runs.csv
outputs/p2b_ptype_stage1_gmm02_p_v3_gmu_hparam_seed42_mean_std.csv
outputs/experiments/p2b_ptype_stage1_gmm02_p_v3_gmu_hparam_seed42/
```

- `runs.csv` 为 8 行，8 个 run 均有 `metrics.json`、`predictions.csv`、`confusion_matrix.csv`、`training_log.csv`。
- 未发现 `Traceback`、`RuntimeError`、CUDA OOM、路径错误、磁盘不足或 summarize failed。
- 注意：`mean_std.csv` 只有 3 行，因为多个 P2b 设置共享 `dropout=0.1` 后被聚合到同一行；P2b 八个设置的正式比较以 `runs.csv + metrics/log` 为准。

P2b seed42 主表，括号为相对 P2a GMU seed42 的变化：

| 设置 | Val F1 | Val Acc | Val Bal Acc | Test F1 | Test Acc | Test Bal Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `aux_totstrong` | 0.9878 (+0.0001) | 0.9901 (+0.0003) | 0.9843 (+0.0003) | 0.9821 (-0.0001) | 0.9855 (+0.0001) | 0.9765 (-0.0005) |
| `aux_none` | 0.9878 (+0.0001) | 0.9899 (+0.0000) | **0.9851 (+0.0011)** | 0.9808 (-0.0014) | 0.9846 (-0.0008) | 0.9754 (-0.0016) |
| `dropout005` | 0.9876 (-0.0001) | 0.9897 (-0.0001) | 0.9845 (+0.0005) | 0.9822 (+0.0000) | 0.9854 (+0.0000) | 0.9774 (+0.0004) |
| `bias3` | 0.9875 (-0.0002) | 0.9899 (+0.0000) | 0.9840 (-0.0000) | 0.9824 (+0.0002) | 0.9859 (+0.0005) | 0.9772 (+0.0002) |
| `bias1` | 0.9873 (-0.0005) | 0.9897 (-0.0001) | 0.9815 (-0.0025) | 0.9832 (+0.0010) | 0.9866 (+0.0012) | 0.9751 (-0.0020) |
| `dropout02` | 0.9871 (-0.0006) | 0.9893 (-0.0005) | 0.9841 (+0.0001) | 0.9827 (+0.0005) | 0.9858 (+0.0004) | 0.9780 (+0.0009) |
| `aux_light` | 0.9871 (-0.0006) | 0.9893 (-0.0005) | 0.9839 (-0.0001) | 0.9825 (+0.0004) | 0.9856 (+0.0003) | 0.9779 (+0.0009) |
| `aux_balanced` | 0.9868 (-0.0010) | 0.9893 (-0.0005) | 0.9814 (-0.0026) | 0.9832 (+0.0010) | 0.9866 (+0.0012) | 0.9753 (-0.0018) |

重点类别与稳定性：

| 设置 | Val `Sr` F1/R | Val `Sr->Co` / `Co->Sr` | post-best drop | `Val F1 <0.90` |
| --- | ---: | ---: | ---: | ---: |
| `aux_totstrong` | **0.9630 / 0.9415** | 59 / 14 | 0.0480 | 0 |
| `aux_none` | 0.9617 / **0.9464** | **54** / 22 | **0.0150** | 0 |
| `dropout005` | 0.9611 / 0.9435 | 57 / 20 | 0.0396 | 0 |
| `bias3` | 0.9619 / 0.9405 | 60 / 15 | 0.0179 | 1 |
| `bias1` | 0.9611 / 0.9306 | 70 / **6** | 0.0285 | 0 |
| `dropout02` | 0.9596 / 0.9425 | 58 / 22 | 0.0262 | 0 |
| `aux_light` | 0.9596 / 0.9415 | 59 / 21 | 0.0736 | 0 |
| `aux_balanced` | 0.9595 / 0.9276 | 73 / **6** | 0.0292 | 0 |

阶段判断：

- P2b 没有出现明显超过 P2a GMU three-seed mean 的设置；所有提升都属于 seed42 单 run 上的极小差异。
- `aux_none` 在 `Val Balanced Acc`、`Sr` recall、`Sr->Co` 混淆和训练稳定性上最干净，值得作为候选观察；但其 test 指标低于 P2a GMU seed42，且 test 不用于模型选择。
- `aux_totstrong` 的 `Val F1`、`Val Acc` 和 `Sr` F1 最高，但 best 后下探到 `0.9398` 再恢复，稳定性弱于 `aux_none`。
- `dropout005` 接近 baseline，可作为低优先级备选；`aux_light`、`aux_balanced`、`bias1`、`dropout02`、`bias3` 不显示足够 validation 优势。
- 当前更保守的结论是：P2b 支持继续沿用 P2a GMU 默认设置；如必须做 P2c，则只建议在 `P2a GMU default` 与 `aux_none` / `aux_totstrong` 之间做很小范围 three-seed certification。

#### P2c GMU base vs ToT-strong auxiliary three-seed comparison

P2c 只比较 `dual_stream_gmu_aux` 的默认 auxiliary loss 和 P2b 中 `Val Macro-F1` / `Val Acc` 最靠前的 ToT-strong 变体。该实验用于确认 ToT-strong 是否能在 three-seed 下稳定优于 base；不再扩展 gate bias、dropout 或学习率。

固定设置：

| 项目 | 设置 |
| --- | --- |
| Dataset | `particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3` |
| Server data root | `/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3` |
| Input | `ToT + relative_minmax ToA` |
| Model | `dual_stream_gmu_aux` |
| Loss | `cross_entropy`, `label_encoding=onehot`, `class_weight=balanced` |
| Primary metric | `val_macro_f1` |
| LR / Epoch / Patience | `3e-6` / `50` / `15` |
| Batch / WD / Scheduler | `64` / `1e-4` / `cosine`, `eta_min=1e-7` |
| Seeds | `42, 43, 44` |

P2c 实验矩阵：

| 编号 | Config | `aux_tot` | `aux_toa` | 目的 |
| --- | --- | ---: | ---: | --- |
| P2c-1 | `configs/experiments/p2c_ptype_stage1_gmm02_p_v3_gmu_base_3seed.yaml` | 0.3 | 0.1 | GMU base 复跑，用于同 group 直接对照 |
| P2c-2 | `configs/experiments/p2c_ptype_stage1_gmm02_p_v3_gmu_aux_totstrong_3seed.yaml` | 0.5 | 0.1 | ToT-strong auxiliary 认证 |

服务器训练与汇总命令：

```bash
tmux new -s p2c_ptype_stage1_gmm02_p_v3_gmu_base_vs_totstrong_3seed
cd /root/Timepix
source /etc/network_turbo
PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v3
GROUP=p2c_ptype_stage1_gmm02_p_v3_gmu_base_vs_totstrong_3seed
LOG=outputs/${GROUP}_tmux.log
CONFIGS=(
  configs/experiments/p2c_ptype_stage1_gmm02_p_v3_gmu_base_3seed.yaml
  configs/experiments/p2c_ptype_stage1_gmm02_p_v3_gmu_aux_totstrong_3seed.yaml
)

{
  echo "[P2c] start $(date)"
  git rev-parse --short HEAD
  test -d "$DATA"
  for cfg in "${CONFIGS[@]}"; do
    echo "[P2c] dry-run $cfg"
    $PY scripts/run_grid.py --config "$cfg" --data-root "$DATA" --dry-run
  done
  for cfg in "${CONFIGS[@]}"; do
    echo "[P2c] run $cfg $(date)"
    $PY scripts/run_grid.py --config "$cfg" --data-root "$DATA" --skip-existing --continue-on-error
  done
  echo "[P2c] summarize $(date)"
  $PY scripts/summarize.py --group "$GROUP" --out outputs/${GROUP}_runs.csv
  $PY scripts/aggregate_seeds.py --summary outputs/${GROUP}_runs.csv --out outputs/${GROUP}_mean_std.csv
  echo "[P2c] done $(date)"
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

P2c 判断规则：

- 只按 validation 侧选择，primary 为 `Val Macro-F1`。
- 若 `Val Macro-F1` 差距 `<0.001`，依次比较 `Val Balanced Acc`、`Sr` F1/recall、`Sr->Co` 混淆和训练稳定性。
- test 只作为泛化报告，不用于反选最终设置。
- 若 ToT-strong 未稳定优于 base，则 P 系列 GMU 保持 P2a/P2c base 设置。

P2c 执行备注与结果：

- 服务器 `35610` 已完成 6/6 个 run，tmux session 正常退出，GPU 已释放。
- 结果文件：

```text
outputs/p2c_ptype_stage1_gmm02_p_v3_gmu_base_vs_totstrong_3seed_runs.csv
outputs/p2c_ptype_stage1_gmm02_p_v3_gmu_base_vs_totstrong_3seed_mean_std.csv
outputs/experiments/p2c_ptype_stage1_gmm02_p_v3_gmu_base_vs_totstrong_3seed/
```

- `runs.csv` 为 6 行，6 个 run 均有 `metrics.json`、`predictions.csv`、`confusion_matrix.csv`、`training_log.csv`。
- 未发现 `Traceback`、`RuntimeError`、CUDA OOM、路径错误、磁盘不足或 summarize failed。
- 注意：`mean_std.csv` 只有 1 行并显示 `n_runs=6`，把 base 与 ToT-strong 合并了；P2c 正式结果按 `experiment_name` 手动分组计算。

P2c 三 seed 主表：

| Variant | Val Macro-F1 | Val Acc | Val Balanced Acc | Test Macro-F1 | Test Acc | Test Balanced Acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.9875±0.0003 | 0.9897±0.0004 | 0.9841±0.0009 | 0.9826±0.0009 | 0.9858±0.0007 | 0.9771±0.0007 |
| `ToT-strong` | **0.9881±0.0008** | **0.9902±0.0006** | **0.9850±0.0006** | **0.9831±0.0003** | **0.9862±0.0003** | **0.9778±0.0003** |
| Δ `ToT-strong - base` | +0.0006 | +0.0005 | +0.0009 | +0.0005 | +0.0004 | +0.0007 |

逐 seed 结果：

| Variant | Seed | Best/Stop | Val F1 | Val Acc | Val Bal | Test F1 | Test Acc | Test Bal |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 42 | 9/24 | 0.9875 | 0.9896 | 0.9841 | 0.9817 | 0.9851 | 0.9768 |
| `base` | 43 | 23/38 | 0.9873 | 0.9895 | 0.9833 | 0.9834 | 0.9864 | 0.9779 |
| `base` | 44 | 18/33 | 0.9879 | 0.9901 | 0.9850 | 0.9827 | 0.9859 | 0.9766 |
| `ToT-strong` | 42 | 21/36 | 0.9872 | 0.9896 | 0.9844 | 0.9830 | 0.9864 | 0.9778 |
| `ToT-strong` | 43 | 18/33 | **0.9887** | **0.9907** | 0.9853 | **0.9835** | 0.9864 | **0.9781** |
| `ToT-strong` | 44 | 20/35 | 0.9884 | 0.9904 | **0.9854** | 0.9828 | 0.9859 | 0.9775 |

类别与混淆：

| Variant | Val `Sr` F1/R | Test `Sr` F1/R | Val `Sr->Co` / `Co->Sr` | Test `Sr->Co` / `Co->Sr` |
| --- | ---: | ---: | ---: | ---: |
| `base` | 0.9612±0.0017 / 0.9415±0.0040 | 0.9456±0.0021 / 0.9151±0.0038 | 177 / 53 | 257 / 62 |
| `ToT-strong` | **0.9631±0.0018 / 0.9451±0.0015** | **0.9475±0.0016 / 0.9177±0.0020** | **166 / 53** | **249 / 59** |
| Δ `ToT-strong - base` | +0.0019 / +0.0036 | +0.0019 / +0.0026 | -11 / 0 | -8 / -3 |

训练稳定性：

| Variant | Seed | Best F1(epoch) | Post-best min(epoch) | Post-best drop | Global min(epoch) | `Val F1 <0.90` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 42 | 0.9875 (9) | 0.5894 (13) | 0.3981 | 0.5894 (13) | 1 |
| `base` | 43 | 0.9873 (23) | 0.9677 (29) | 0.0195 | 0.9171 (13) | 0 |
| `base` | 44 | 0.9879 (18) | 0.9581 (25) | 0.0298 | 0.9435 (1) | 0 |
| `ToT-strong` | 42 | 0.9872 (21) | 0.9577 (34) | 0.0295 | 0.8510 (17) | 1 |
| `ToT-strong` | 43 | 0.9887 (18) | 0.6035 (26) | 0.3853 | 0.6035 (26) | 1 |
| `ToT-strong` | 44 | 0.9884 (20) | 0.9681 (35) | 0.0203 | 0.9434 (1) | 0 |

阶段判断：

- 按 primary `Val Macro-F1`，ToT-strong 比 base 高 `+0.0006`，差距小于 `0.001`，属于很小但方向一致的优势。
- 按 tie-break，ToT-strong 在 `Val Balanced Acc`、`Sr` F1/recall、`Sr->Co` 混淆上也均有小幅优势。
- 稳定性仍是风险点：ToT-strong seed43 有一次 post-best 深下探到 `0.6035`；base seed42 也有类似深下探到 `0.5894`，说明 validation collapse 不是 ToT-strong 独有，但 ToT-strong 的 `<0.90` 次数为 2，base 为 1。
- 当前可采用的保守结论：ToT-strong 是 P2c validation 侧略优的 GMU auxiliary 设置，但优势幅度很小，且未解决训练过程中的偶发 validation collapse。若最终模型重视 Sr 类和 validation 均值，可采用 ToT-strong；若更强调训练过程稳定性和最少改动，可继续沿用 base。

#### P3a recurrent-removed dataset, P2a plus GMU ToT-strong three-seed comparison

P3a 使用进一步提纯的数据集 `particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v4_p2c_recurrent_removed`。该数据集从 P2c 诊断中移除了反复出现的 hard recurrent samples，因此不再与 P2a/P2c 的 v3 结果混合汇总。P3a 的目的不是重新调参，而是在新数据集上复刻 P2a 的五模型模态/融合对比，并额外加入 P2c 中略优的 GMU ToT-strong auxiliary 变体。

固定设置：

| Item | Value |
| --- | --- |
| Dataset config | `configs/datasets/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v4_p2c_recurrent_removed.yaml` |
| Server data root | `/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v4_p2c_recurrent_removed` |
| Label type | `categorical_folder`，类别名从文件夹自动读取 |
| Primary metric | `val_macro_f1` |
| Loss | balanced `cross_entropy` + one-hot |
| Training | `lr=3e-6`, `epochs=50`, `patience=15`, `batch_size=64` |
| Split | `outputs/splits/ptype_stage1_gmm02_p_v4rm_ToT-ToA_seed42_0.8_0.1_0.1.json` |
| Group | `p3a_ptype_stage1_gmm02_p_v4rm_p2a_plus_totstrong_3seed` |

实验矩阵：

| 编号 | 配置文件 | 输入/模型 | 目的 |
| --- | --- | --- | --- |
| P3a-a | `configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_tot_3seed.yaml` | `ToT`, `resnet18_no_maxpool` | ToT 单模态基线 |
| P3a-b | `configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_rtoa_3seed.yaml` | `RToA`, `resnet18_no_maxpool` | 相对 ToA 单模态 |
| P3a-c | `configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_input_concat_3seed.yaml` | `[ToT, RToA]`, input concat | 输入层拼接 |
| P3a-d | `configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_dual_concat_3seed.yaml` | dual-stream concat auxiliary | 双分支拼接对照 |
| P3a-e | `configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_gmu_3seed.yaml` | GMU base auxiliary | P2a 默认 GMU |
| P3a-f | `configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_gmu_aux_totstrong_3seed.yaml` | GMU ToT-strong auxiliary | P2c 略优 GMU 变体 |

服务器完整训练与汇总命令：

```bash
tmux new -s p3a_ptype_stage1_gmm02_p_v4rm_3seed
```

```bash
cd /root/Timepix
source /etc/network_turbo || true
git pull --ff-only

PY=/root/miniconda3/bin/python
DATA=/root/autodl-tmp/particle_type_stage1_full_am_co_sr_gmm_k3_label0_2_p_v4_p2c_recurrent_removed
GROUP=p3a_ptype_stage1_gmm02_p_v4rm_p2a_plus_totstrong_3seed
CONFIGS=(
  configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_tot_3seed.yaml
  configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_rtoa_3seed.yaml
  configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_input_concat_3seed.yaml
  configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_dual_concat_3seed.yaml
  configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_gmu_3seed.yaml
  configs/experiments/p3a_ptype_stage1_gmm02_p_v4rm_gmu_aux_totstrong_3seed.yaml
)

{
  echo "[P3a] start $(date)"
  for cfg in "${CONFIGS[@]}"; do
    echo "[P3a] dry-run $cfg"
    "$PY" scripts/run_grid.py --config "$cfg" --data-root "$DATA" --dry-run || exit 1
  done
  for cfg in "${CONFIGS[@]}"; do
    echo "[P3a] run $cfg $(date)"
    "$PY" scripts/run_grid.py --config "$cfg" --data-root "$DATA" --skip-existing --continue-on-error || exit 1
  done
  echo "[P3a] summarize $(date)"
  "$PY" scripts/summarize.py --group "$GROUP" --out "outputs/${GROUP}_runs.csv" || exit 1
  "$PY" scripts/aggregate_seeds.py --summary "outputs/${GROUP}_runs.csv" --out "outputs/${GROUP}_mean_std.csv" || exit 1
  echo "[P3a] done $(date)"
} 2>&1 | tee "outputs/${GROUP}.log"
```

本地拉取命令：

```powershell
rclone copy autodl35610:/root/Timepix/outputs D:\Project\Timepix\outputs --update --progress
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
