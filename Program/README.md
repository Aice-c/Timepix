AlphaAnalysis 聚类使用说明
==========================

使用 ResNet18 的 CNN 特征，对 `data/Alpha` 下的某一类样本（联合模态：ToT+ToA）进行无监督聚类（默认 k=2），并将样本拷贝到两个输出目录。

快速开始
--------

1) 运行聚类（示例：对 0 类聚类为 2 类）：

```
python -m src.cluster_resnet18 --class-id 0 --k 2 --batch-size 64
```

运行完成后，输出位于：`output/cluster/Alpha_class_0/`，包含：
- `cluster_0/` 与 `cluster_1/`（每个簇内按照模态分别存放在 `ToT/` 与 `ToA/` 子目录）
- `cluster_k2.csv`（每条样本的簇结果与原始路径）
- `summary_k2.txt`（簇大小汇总）
- `pca_k2.png`（特征 PCA 可视化）

2) 指定输出的两个目录（可选）：

```
python -m src.cluster_resnet18 --class-id 0 --k 2 \
	--dest-dir0 /path/to/clusterA \
	--dest-dir1 /path/to/clusterB
```

脚本会在 `clusterA` 与 `clusterB` 下分别建立 `ToT/` 与 `ToA/` 子目录，并将对应样本拷贝过去。

参数说明
--------

- `--class-id`：要聚类的数据类别编号（`data/Alpha/<class-id>/`）
- `--k`：聚类簇数，默认为 2
- `--batch-size`：特征提取时的批大小
- `--crop-size`：中心裁剪尺寸，默认使用 `Config.py` 的 `feature_size`
- `--weights`：可选，若已有 ResNet18 CNN 训练权重，可提供权重路径用于更优的特征
- `--dest-dir0`、`--dest-dir1`：可选，两个簇的输出根目录；未提供时写入默认 `output/cluster/Alpha_class_<id>/cluster_{0,1}`

手工特征拼接（可选）
--------------------

- 在 `Config.py` 中启用手工特征（例如将 `handcrafted_features['ToT']['total_energy'] = True` 等）。
- 运行聚类时可以让聚类特征 = CNN 嵌入 + 手工特征：

```
python -m src.cluster_resnet18 --class-id 0 --k 2 --use-handcrafted true
```

- 默认会对手工特征做 z-score 标准化（仅针对手工特征，避免其量纲影响聚类）；如需关闭：

```
python -m src.cluster_resnet18 --class-id 0 --k 2 --use-handcrafted true --no-standardize-handcrafted
```

- 若不传 `--use-handcrafted`，则脚本自动遵循 `Config.py` 是否启用了手工特征。

实现要点
--------

- 统一使用 ToT+ToA 两通道作为输入；内部自动设置 `Config.modalities = ['ToT','ToA']`
- 以 ResNet18 的 CNN 中间特征作为嵌入向量，使用 KMeans 聚类
- 对每个模态按 `Config.standardization` 计算 z-score 标准化（基于该类样本）
