# Timepix 数据分析脚本指南

这份文档记录新建的论文数据分析链路。它独立于训练主链路，不修改 `timepix/training/`、模型训练配置或现有实验输出规则。分析代码集中在 `timepix/analysis/`，入口脚本放在 `scripts/`。

## 1. 目标与边界

分析链路服务于本科论文的数据章节，分成两层：

- 数据集分析：说明原始 `256 x 256` Timepix3 帧如何经过连通区域轨迹提取、ToT 统计量清洗，形成 `Alpha_100` 和全量 `Proton_C` 两个监督学习数据集。
- 近垂直分辨极限分析：只针对全量 `Proton_C` 的 `ToT` 单模态，分析 `80, 82, 84, 86, 88, 90` 六个近垂直角度在当前数据表示和已测试特征/模型族下是否具备足够可分性。

重要边界：

- 不假设 C/质子数据集存在 `ToA`。
- 统计检验不只报告 p-value，同时报告 KS statistic、Wasserstein distance、Cliff's delta、median difference、IQR overlap ratio 等效应量。
- 传统机器学习基线只在训练集上拟合，test split 只用于最终报告。
- 代表性样本由自动规则选择，不人工挑图。
- 所有输出写入 `outputs/data_analysis/`、`outputs/resolution_limit/` 和 `outputs/analysis_report.md`，这些路径不需要纳入 git。
- 注意：数据分析链路使用全量 `Proton_C`；训练实验主线使用 7 分类子集 `Proton_C_7`。二者故意分开命名，避免把全量分析和训练数据集混在一起。

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
```

## 3. 数据集分析

服务器 Linux 命令：

```bash
python scripts/analyze_datasets.py \
  --data-root Data \
  --output-root outputs/data_analysis \
  --datasets Alpha_100 Proton_C \
  --sample-cap-plot 5000 \
  --seed 42
```

主要输出：

```text
outputs/data_analysis/
  dataset_analysis_report.md
  manifest.json
  tables/
    dataset_index.csv/.md
    dataset_summary.csv/.md
    split_counts.csv/.md
    class_counts_alpha100.csv/.md
    class_counts_proton_c_all.csv/.md
    class_counts_proton_c_10_70.csv/.md
    class_counts_proton_c_80_90.csv/.md
    feature_summary_by_angle_*.csv/.md
  figures/
    preprocessing_pipeline.png/.pdf
    class_counts_*.png/.pdf
    representative_*.png/.pdf
    feature_violin_*.png/.pdf
    feature_scatter_*.png/.pdf
    alpha100_tot_toa_pair_grid.png/.pdf
    alpha100_toa_span_by_angle.png/.pdf
    alpha100_tot_toa_corr_by_angle.png/.pdf
  cache/
    all_event_features.csv
    alpha100_tot_features.csv
    alpha100_toa_features.csv
    proton_c_tot_features.csv
```

`dataset_index.csv` 会记录每个样本的读取状态、shape、角度、模态、路径和归一化后的 `sample_key`。如果 `Alpha_100` 同时存在 `ToT` 与 `ToA`，脚本会检查二者是否一一配对。

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

主要输出：

```text
outputs/resolution_limit/
  resolution_limit_report.md
  analysis_notes.md
  manifest.json
  tables/
    proton_c_near_vertical_features.csv
    near_vertical_dataset_summary.csv/.md
    near_vertical_pairwise_effect_sizes.csv/.md
    near_vertical_adjacent_ks.csv/.md
    near_vertical_adjacent_wasserstein.csv/.md
    near_vertical_feature_distance_summary.csv/.md
    near_vertical_ml_baselines.csv/.md
    near_vertical_ml_baselines_mean_std.csv/.md
    near_vertical_pairwise_auc.csv/.md
    near_vertical_auc_by_angle_gap.csv/.md
    near_vertical_overfit_experiment.csv/.md
  figures/
    near_vertical_representative_tot.png/.pdf
    near_vertical_mean_tot_by_angle.png/.pdf
    near_vertical_adjacent_difference_maps.png/.pdf
    near_vertical_feature_violin_core.png/.pdf
    near_vertical_feature_kde_core.png/.pdf
    near_vertical_ks_heatmap.png/.pdf
    near_vertical_wasserstein_heatmap.png/.pdf
    near_vertical_pca_features.png/.pdf
    near_vertical_umap_features.png/.pdf   # Linux 且 umap-learn 可用时生成
    near_vertical_confusion_matrix_*.png/.pdf
    near_vertical_auc_by_angle_gap.png/.pdf
    near_vertical_balanced_acc_by_angle_gap.png/.pdf
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

## 5. 汇总报告

```bash
python scripts/make_analysis_report.py \
  --data-analysis-root outputs/data_analysis \
  --resolution-root outputs/resolution_limit \
  --out outputs/analysis_report.md
```

最终报告会把 `dataset_analysis_report.md` 与 `resolution_limit_report.md` 合并，形成一个可交给 5.5 Pro 或用于论文写作参考的总览文档。

## 6. 论文表述边界

近垂直分辨极限报告使用谨慎表述：

```text
在当前探测器设置、事件提取方法、ToT 单模态矩阵表示和已测试模型/特征族条件下，C/质子近垂直角度 80-90 deg、2 deg 间隔的数据没有表现出足够的可分性，难以支持可靠监督分类。
```

不要写成“深度学习绝对无法区分近垂直角度”。这个结论只限定在当前数据、表示方式、清洗流程和已测试模型/特征范围内。
