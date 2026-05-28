# Timepix 数据分析脚本指南

这份文档记录新建的论文数据分析链路。它独立于训练主链路，不修改 `timepix/training/`、模型训练配置或现有实验输出规则。分析代码集中在 `timepix/analysis/`，入口脚本放在 `scripts/`。

## 1. 目标与边界

分析链路服务于本科论文的数据章节，分成两层：

- 数据集分析：说明原始 `256 x 256` Timepix3 帧如何经过连通区域轨迹提取、ToT 统计量清洗，形成 `Alpha_100` 和全量 `Proton_C` 两个监督学习数据集，并从全量 `Proton_C` 中按角度派生论文训练主线统计用的 `Proton_C_7`。
- 近垂直分辨极限分析：只针对全量 `Proton_C` 的 `ToT` 单模态，分析 `80, 82, 84, 86, 88, 90` 六个近垂直角度在当前数据表示和已测试特征/模型族下是否具备足够可分性。

重要边界：

- 不假设 C/质子数据集存在 `ToA`。
- 统计检验不只报告 p-value，同时报告 KS statistic、Wasserstein distance、Cliff's delta、median difference、IQR overlap ratio 等效应量。
- 传统机器学习基线只在训练集上拟合，test split 只用于最终报告。
- 代表性样本由自动规则选择，不人工挑图。
- 所有输出写入 `outputs/data_analysis/`、`outputs/resolution_limit/` 和 `outputs/analysis_report.md`，这些路径不需要纳入 git。
- 注意：数据分析链路使用全量 `Proton_C`；训练实验主线使用 7 分类子集 `Proton_C_7`。在数据分析脚本中，`Proton_C_7` 不要求单独目录，而是直接从全量 `Proton_C` 中按 `10, 20, 30, 45, 50, 60, 70` 角度过滤得到。二者故意分开命名，避免把全量分析和训练数据集混在一起。

## 2. 代码组织

```text
timepix/analysis/
  io.py                 # 数据集扫描、矩阵读取、split 统计、输出目录
  features.py           # 事件级特征和 ToA 额外特征
  stats.py              # 相邻角度统计距离和效应量
  ml.py                 # 传统机器学习基线、pairwise AUC
  plotting.py           # PNG/PDF 图片输出
  representative.py     # 固定 seed 抽样与代表性样本选择
  tables.py             # CSV/Markdown 表格输出
  reports.py            # Markdown 报告生成

scripts/analyze_datasets.py
scripts/analyze_resolution_limit.py
scripts/make_analysis_report.py
requirements-analysis.txt
```

服务器分析环境建议安装：

```bash
pip install -r requirements-analysis.txt
```

如果只想补齐当前缺失的 UMAP 图依赖，可以只安装：

```bash
pip install umap-learn
```

说明：

- 近垂直分析日志里的 `Skipped UMAP: No module named 'umap'` 表示缺少 `umap-learn`。
- `Skipped t-SNE: pass --tsne to enable the optional slow embedding.` 不是缺依赖，而是脚本默认跳过较慢的 t-SNE；需要时加 `--tsne`。
- 绘图采用 Matplotlib 后端，默认保存 300 dpi PNG 和可编辑/可放大的 PDF，并设置论文友好的字号、色盲友好配色和可嵌入字体。
- 长耗时步骤会显示 `tqdm` 进度条，包括数据扫描、事件特征提取、传统 ML 基线和 pairwise AUC；如果环境缺少 `tqdm`，脚本仍会正常运行但不显示进度条。

本地 Windows 验证环境：

```powershell
conda activate timepix-local
```

本地数据路径：

```text
Alpha_100  -> D:\Project\Timepix\Data\Alpha_100
Proton_C   -> E:\C1Analysis\Proton_C
Proton_C_7 -> 不要求独立目录；从 Proton_C 中过滤 10, 20, 30, 45, 50, 60, 70 得到
```

注意：数据分析脚本的 `--data-root` 参数是默认数据集父目录；如果不同数据集分散在不同硬盘，使用 `--dataset-root DatasetName=具体路径` 显式指定。`Proton_C_7` 是从 `Proton_C` 派生的统计子集，角度固定为 `10, 20, 30, 45, 50, 60, 70`，不要写成 40°。

## 3. 数据集分析

服务器 Linux 命令：

```bash
python scripts/analyze_datasets.py \
  --data-root Data \
  --dataset-root Proton_C=/root/autodl-tmp/Proton_C \
  --output-root outputs/data_analysis \
  --datasets Alpha_100 Proton_C \
  --sample-cap-plot 5000 \
  --seed 42
```

本地 Windows 合并分析命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe scripts\analyze_datasets.py `
  --data-root D:\Project\Timepix\Data `
  --dataset-root Alpha_100=D:\Project\Timepix\Data\Alpha_100 `
  --dataset-root Proton_C=E:\C1Analysis\Proton_C `
  --output-root outputs\data_analysis_v2_local `
  --datasets Alpha_100 Proton_C `
  --sample-cap-plot 5000 `
  --seed 42
```

本地也可以只分析 Proton：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe scripts\analyze_datasets.py `
  --data-root E:\C1Analysis `
  --output-root outputs\data_analysis_proton_local `
  --datasets Proton_C `
  --sample-cap-plot 5000 `
  --seed 42
```

主要输出：

```text
outputs/data_analysis/
  00_dataset_inventory.csv/.md
  01_alpha_pairing_audit.csv/.md
  02_split_distribution.csv/.md
  03_cleaning_thresholds.csv/.md
  03_cleaning_info_missing.md
  04_event_features_alpha_100.csv
  04_event_features_proton_c.csv
  04_event_features_proton_c_7.csv
  05_sample_count_by_angle.csv/.md
  10_alpha_class_summary.csv/.md
  14_proton_full_angle_summary.csv/.md
  15_proton_c7_relation_to_full.csv/.md
  proton_input_shape_audit.csv/.md
  alpha_toa_negative_audit.csv/.md
  proton_angle_consistency_audit.csv/.md
  analysis_tables.xlsx
  dataset_analysis_report.md
  manifest.json
  figures/
    06_alpha_tot_representative_grid.png/.pdf
    06_alpha_toa_representative_grid.png/.pdf
    06_proton_tot_representative_grid.png/.pdf
    07_*_heatmap.png/.pdf
    08_feature_violin_*.png/.pdf
    09_*_active_count_vs_sum.png/.pdf
```

`00_dataset_inventory.csv` 会记录每个角度、模态的样本数、输入形状和值域审计。如果 `Alpha_100` 同时存在 `ToT` 与 `ToA`，脚本会检查二者是否一一配对。`04_event_features_*.csv` 是原始事件级长表，保留为 CSV，不默认塞进最终 xlsx。

补充审计说明：

- `10_alpha_class_summary.csv` 的 `num_train/num_val/num_test` 以样本 key 为单位统计，只选一份 Alpha split，不把 ToT 与 ToT-ToA 两份 split manifest 重复相加。
- `proton_input_shape_audit.csv` 同时记录文件保存尺寸、Dataset loader 推断输出尺寸和训练模型有效输入尺寸。当前本地审计结论为 `Proton_C` 与派生 `Proton_C_7` 均是 `50x50` 文件，训练配置 `data.crop_size=0`，因此 ResNet 训练有效输入为 `1x50x50`。
- `alpha_toa_negative_audit.csv` 定位 Alpha ToA 负值样本、非零 ToA 分布和对应 ToT 文件。
- `proton_angle_consistency_audit.csv` 比较当前 `Proton_C` inventory 与 full Proton split，当前 `75,83,85` 只出现在旧 full split 中，当前本地数据目录没有对应角度，标记为 `split_residual_no_current_data`。

## 4. 近垂直分辨极限分析

服务器 Linux 命令：

```bash
python scripts/analyze_resolution_limit.py \
  --data-root Data \
  --dataset Proton_C \
  --angles 80 82 84 86 88 90 \
  --modality ToT \
  --output-root outputs/resolution_limit \
  --sample-cap-plot 5000 \
  --sample-cap-ml 10000 \
  --seeds 42 43 44 45 46
```

本地 Windows 命令：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe scripts\analyze_resolution_limit.py `
  --dataset-root E:\C1Analysis\Proton_C `
  --dataset Proton_C `
  --angles 80 82 84 86 88 90 `
  --modality ToT `
  --output-root outputs\resolution_limit_v2_local `
  --sample-cap-plot 5000 `
  --sample-cap-ml 10000 `
  --seeds 42 43 44 45 46
```

主要输出：

```text
outputs/resolution_limit/
  proton_c_near_vertical_features.csv
  00_near_vertical_inventory.csv/.md
  01_geometry_projection.csv/.md 或 01_geometry_projection_missing.md
  02_near_vertical_representative_samples.csv/.md
  04_near_vertical_feature_summary.csv/.md
  05_pairwise_effect_size_adjacent.csv/.md
  05_pairwise_effect_size_all_pairs.csv/.md
  05_feature_max_effect_size.csv/.md
  06_single_feature_pairwise_auc.csv/.md
  near_vertical_pairwise_effect_size.csv/.md
  near_vertical_single_feature_auc.csv/.md
  08_classical_ml_by_seed.csv/.md
  08_classical_ml_summary.csv/.md
  near_vertical_classical_ml_summary.csv/.md
  09_pairwise_classical_ml.csv/.md
  09_auc_by_angle_gap.csv/.md
  near_vertical_dl_failure_audit.csv/.md
  near_vertical_overfit_experiment.csv
  near_vertical_overfit_learning_curve.csv
  resolution_limit_tables.xlsx
  resolution_limit_report.md
  analysis_notes.md
  manifest.json
  figures/
    02_near_vertical_representative_tot_grid.png/.pdf
    03_near_vertical_mean_tot_heatmap.png/.pdf
    03_near_vertical_occupancy_heatmap.png/.pdf
    04_near_vertical_feature_violin_*.png/.pdf
    05_*_heatmap_by_feature.png/.pdf
    06_single_feature_auc_heatmap.png/.pdf
    07_pca_near_vertical_features.png/.pdf
    07_umap_near_vertical_features.png/.pdf
    08_classical_ml_confusion_matrix_*.png/.pdf
    near_vertical_overfit_learning_curve.png/.pdf
```

UMAP 是可选图。Windows 本地环境中如果 UMAP/numba 后端不稳定，脚本会跳过 UMAP 并写入 `analysis_notes.md`；Linux 服务器上会自动尝试生成。

传统基线包含：

- Dummy most_frequent
- Dummy stratified
- LogisticRegression
- LinearSVM
- RandomForest
- MLPClassifier
- RBF-SVM，小样本时启用，训练集过大时自动跳过

其中 LogisticRegression 使用 `liblinear` 求解器，以避免部分本地环境中默认 `lbfgs` 后端直接崩溃。

小样本过拟合 sanity check 独立运行，不修改训练主链路：

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe scripts\run_near_vertical_overfit_check.py `
  --dataset-root E:\C1Analysis\Proton_C `
  --output-root outputs\resolution_limit_v2_local `
  --angles 80 82 84 86 88 90 `
  --modality ToT `
  --samples-per-class 5 10 50 100 `
  --epochs 200 `
  --batch-size 64 `
  --seed 42 `
  --device cpu
```

本地 `timepix-local` 当前 PyTorch 为 CPU-only，`torch.cuda.is_available()` 为 `False`。不要在论文分析流程中为追求加速直接修改系统 CUDA 或显卡驱动；如需 GPU，应新建独立 conda 环境安装匹配驱动的 CUDA 版 PyTorch。

## 5. 汇总报告

```bash
python scripts/make_analysis_report.py \
  --data-analysis-root outputs/data_analysis \
  --resolution-root outputs/resolution_limit \
  --out outputs/analysis_report.md \
  --tables-out outputs/analysis_tables/timepix_analysis_tables.xlsx
```

最终报告会把 `dataset_analysis_report.md` 与 `resolution_limit_report.md` 合并，形成一个可交给 5.5 Pro 或用于论文写作参考的总览文档。默认的总 xlsx 只合并汇总表，不包含 `04_event_features_*` 和 `proton_c_near_vertical_features.csv` 这类几十万行长表；如确实需要把原始长表也写入 xlsx，可额外加 `--include-raw-features`，但不推荐作为论文表格工作簿。

## 6. 论文表述边界

近垂直分辨极限报告使用谨慎表述：

```text
在当前探测器设置、事件提取方法、ToT 单模态矩阵表示和已测试模型/特征族条件下，C/质子近垂直角度 80-90 deg、2 deg 间隔的数据没有表现出足够的可分性，难以支持可靠监督分类。
```

不要写成“深度学习绝对无法区分近垂直角度”。这个结论只限定在当前数据、表示方式、清洗流程和已测试模型/特征范围内。
