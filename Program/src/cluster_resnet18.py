import argparse
import os
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from Config import config
from src.dataset import (
    ParticleDataset,
    SampleRecord,
    collect_samples,
    _compute_standardization_stats,
)
from model.Resnet18 import CNN as ResNet18CNN


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _build_normalizer(samples: Sequence[SampleRecord], modalities: Sequence[str], crop_size: int):
    per = config.standardization_settings()
    # 仅传入启用项的配置
    per_enabled = {m: per.get(m, {"enabled": False, "log1p": False, "ignore_zero": False}) for m in modalities}
    stats = _compute_standardization_stats(samples, modalities, crop_size, per_enabled)
    # 复用 dataset.Normalizer 的逻辑：在 ParticleDataset 中传入 normalizer 前，我们只需要返回 stats，
    # 由 ParticleDataset 内部在 __getitem__ 时应用。
    from src.dataset import Normalizer
    if any(per_enabled.get(m, {}).get('enabled', False) for m in modalities):
        return Normalizer(stats=stats, mode=config.standardization_mode(), eps=1e-6)
    return None


def _format_sample_id(record: SampleRecord, modalities: Sequence[str]) -> str:
    # 使用任一模态路径的文件名作为 id（去掉扩展名）
    any_path = record.modalities[modalities[0]]
    base = os.path.basename(any_path)
    return os.path.splitext(base)[0]


def _gather_class_samples(data_root: str, modalities: Sequence[str], class_id: int) -> List[SampleRecord]:
    all_samples = collect_samples(data_root, modalities)
    return [r for r in all_samples if r.label == class_id]


def _build_dataset(samples: Sequence[SampleRecord], modalities: Sequence[str], crop_size: int, normalizer):
    from src.dataset import HandcraftedFeatureExtractor, RotationAugmentor
    feature_flags = {m: config.features_for_modality(m) for m in modalities}
    feature_extractor = HandcraftedFeatureExtractor(feature_flags) if any(
        any(flags.values()) for flags in feature_flags.values()
    ) else None
    rotation_augmentor = RotationAugmentor(enabled=False)
    class_labels = sorted({s.label for s in samples})
    return ParticleDataset(
        samples=samples,
        class_labels=class_labels,
        modalities=modalities,
        feature_extractor=feature_extractor,
        rotation_augmentor=rotation_augmentor,
        is_training=False,
        crop_size=crop_size,
        normalizer=normalizer,
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_cluster_dirs(base_dir: str, modalities: Sequence[str], k: int) -> Dict[int, Dict[str, str]]:
    # 返回 {cluster_id: {modality: path}}
    result: Dict[int, Dict[str, str]] = {}
    for cid in range(k):
        sub = os.path.join(base_dir, f"cluster_{cid}")
        for m in modalities:
            p = os.path.join(sub, m)
            _ensure_dir(p)
            result.setdefault(cid, {})[m] = p
    return result


def _copy_sample(record: SampleRecord, cluster_id: int, cluster_dirs: Dict[int, Dict[str, str]], modalities: Sequence[str]) -> None:
    for m in modalities:
        src = record.modalities[m]
        dst_dir = cluster_dirs[cluster_id][m]
        dst = os.path.join(dst_dir, os.path.basename(src))
        shutil.copy2(src, dst)


def _save_csv(csv_path: str, rows: List[Tuple[str, int, Dict[str, str]]], modalities: Sequence[str]) -> None:
    # rows: (sample_id, cluster, paths_by_modality)
    header = ["sample_id", "cluster"] + [f"path_{m}" for m in modalities]
    lines = [",".join(header)]
    for sid, cid, paths in rows:
        cols = [sid, str(cid)] + [paths[m] for m in modalities]
        # 简单避免逗号影响：不含逗号的路径无需转义
        lines.append(",".join(cols))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_summary(path: str, counts: Dict[int, int]) -> None:
    total = sum(counts.values())
    lines = [f"Total: {total}"] + [f"Cluster {k}: {v}" for k, v in sorted(counts.items())]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save_pca_plot(save_path: str, features: np.ndarray, labels: np.ndarray) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(features)

        plt.figure(figsize=(5, 4))
        for cid in np.unique(labels):
            m = labels == cid
            plt.scatter(reduced[m, 0], reduced[m, 1], s=12, label=f"Cluster {cid}")
        plt.legend()
        plt.title("PCA of features (k=2)")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        # 绘图失败不阻塞主流程
        print(f"[WARN] PCA plot failed: {e}")


def run(class_id: int, k: int, batch_size: int, dest0: Optional[str], dest1: Optional[str], output_root: Optional[str], crop_size: int, weights: Optional[str], use_handcrafted: Optional[bool] = None, standardize_handcrafted: bool = True) -> None:
    # 1) 配置联合模态（ToT+ToA）并构建数据
    modalities = ['ToT', 'ToA']
    # 动态影响 ResNet18 的输入通道
    config.modalities = modalities

    data_root = config.data_dir  # 例如 /root/Timepix/AlphaAnalysis/data/Alpha
    samples = _gather_class_samples(data_root, modalities, class_id)
    if not samples:
        raise RuntimeError(f"未在 {data_root} 中找到类 {class_id} 的样本")

    normalizer = _build_normalizer(samples, modalities, crop_size)
    dataset = _build_dataset(samples, modalities, crop_size, normalizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count() or 1))

    # 2) 构建模型与设备
    device = get_device()
    model = ResNet18CNN().to(device)
    if weights:
        state = torch.load(weights, map_location=device)
        # 兼容多种保存方式
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        # 过滤掉不匹配键
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items() if k.replace('module.', '').startswith('model.')}, strict=False)

    model.eval()

    # 3) 提取嵌入特征
    feats: List[np.ndarray] = []
    feats_hc: List[np.ndarray] = []
    ids: List[str] = []
    paths: List[Dict[str, str]] = []
    # 是否拼接手工特征：默认遵循 Config 中的启用情况，可由入参覆盖
    if use_handcrafted is None:
        use_handcrafted_flag = config.uses_handcrafted_features()
    else:
        use_handcrafted_flag = bool(use_handcrafted)
    with torch.no_grad():
        idx = 0
        for batch in loader:
            # dataset 可能包含手工特征；这里仅用图像模态
            if len(batch) == 3:
                samples_t, _labels, hand = batch
            else:
                samples_t, _labels = batch
            samples_t = samples_t.to(device)
            emb = model(samples_t)  # [B, D]
            feats.append(emb.detach().cpu().numpy())
            if len(batch) == 3 and use_handcrafted_flag:
                # hand: [B, Dh], Tensor(CPU)
                feats_hc.append(hand.numpy())

            # 对应样本元信息（按 DataLoader 顺序）
            for b in range(emb.shape[0]):
                record = samples[idx + b]
                ids.append(_format_sample_id(record, modalities))
                paths.append({m: record.modalities[m] for m in modalities})
            idx += emb.shape[0]

    features_cnn = np.concatenate(feats, axis=0)
    if use_handcrafted_flag and feats_hc:
        hc_all = np.concatenate(feats_hc, axis=0)
        if standardize_handcrafted:
            # 简单 z-score，避免 0 方差
            mean = hc_all.mean(axis=0, keepdims=True)
            std = hc_all.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            hc_all = (hc_all - mean) / std
        features = np.concatenate([features_cnn, hc_all], axis=1)
        print(f"Using handcrafted features: CNN {features_cnn.shape[1]} + HC {hc_all.shape[1]} -> {features.shape[1]}")
    else:
        features = features_cnn

    # 4) 聚类（k=2 默认）
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    pred = kmeans.fit_predict(features)

    # 5) 输出目录与落盘
    if dest0 and dest1:
        # 当用户指定两个目标目录时，仅支持 k==2，避免额外簇无处落地
        if k != 2:
            raise ValueError("当使用 --dest-dir0/--dest-dir1 时，必须将 --k 设为 2。")
        cluster_dirs = {
            0: {m: os.path.join(dest0, m) for m in modalities},
            1: {m: os.path.join(dest1, m) for m in modalities},
        }
        for cid in cluster_dirs:
            for m in modalities:
                _ensure_dir(cluster_dirs[cid][m])
        base_dir = os.path.dirname(dest0)
    else:
        base_dir = output_root or os.path.join(config.output_path, 'cluster', f'Alpha_class_{class_id}')
        _ensure_dir(base_dir)
        cluster_dirs = _make_cluster_dirs(base_dir, modalities, k)

    # 拷贝文件
    counts = {i: 0 for i in range(k)}
    rows: List[Tuple[str, int, Dict[str, str]]] = []
    for i, cid in enumerate(pred):
        record = samples[i]
        _copy_sample(record, int(cid), cluster_dirs, modalities)
        counts[int(cid)] += 1
        rows.append((ids[i], int(cid), paths[i]))

    # 保存 CSV 与汇总
    csv_path = os.path.join(base_dir, f'cluster_k{k}.csv')
    _save_csv(csv_path, rows, modalities)
    _save_summary(os.path.join(base_dir, f'summary_k{k}.txt'), counts)
    _save_pca_plot(os.path.join(base_dir, f'pca_k{k}.png'), features, pred)

    print(f"Done. CSV: {csv_path}")


def parse_args():
    p = argparse.ArgumentParser(description='Cluster Alpha class with ResNet18 embeddings (ToT+ToA)')
    p.add_argument('--class-id', type=int, required=True, help='要聚类的类别编号，例如 0')
    p.add_argument('--k', type=int, default=2, help='聚类簇数，默认为 2')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--crop-size', type=int, default=config.crop_size or 0, help='中心裁剪尺寸，与训练保持一致。0 表示不裁剪')
    p.add_argument('--weights', type=str, default=None, help='可选：ResNet18 CNN 权重路径（若有已训练权重）')
    p.add_argument('--dest-dir0', type=str, default=None, help='可选：簇 0 输出目录（将按模态创建子目录）')
    p.add_argument('--dest-dir1', type=str, default=None, help='可选：簇 1 输出目录（将按模态创建子目录）')
    p.add_argument('--output-root', type=str, default=None, help='未指定 dest-dir 时的默认输出根目录')
    p.add_argument('--use-handcrafted', type=str, default=None, choices=[None, 'true', 'false'], help='是否拼接手工特征；默认遵循 Config 配置，可显式指定 true/false')
    p.add_argument('--no-standardize-handcrafted', action='store_true', help='不对手工特征做 z-score 标准化')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.use_handcrafted is None:
        use_hc = None
    else:
        use_hc = (str(args.use_handcrafted).lower() == 'true')
    run(
        class_id=args.class_id,
        k=args.k,
        batch_size=args.batch_size,
        dest0=args.dest_dir0,
        dest1=args.dest_dir1,
        output_root=args.output_root,
        crop_size=args.crop_size,
        weights=args.weights,
        use_handcrafted=use_hc,
        standardize_handcrafted=(not args.no_standardize_handcrafted),
    )
