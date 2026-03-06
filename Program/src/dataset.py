import os
import random
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(frozen=True)
class SampleRecord:
    label: int
    modalities: Dict[str, str]


@dataclass
class _ModalityStats:
    mean: float
    std: float
    min: float
    max: float
    use_log1p: bool
    ignore_zero_for_stats: bool


class Normalizer:
    def __init__(self, stats: Dict[str, _ModalityStats], mode: str = 'zscore', eps: float = 1e-6) -> None:
        """
        基于训练集统计量、按模态进行标准化/归一化。
        当前支持 'zscore'，保留 mode 以便未来扩展。

        stats: 仅为启用标准化的模态提供统计量
        mode: 'zscore'
        eps: 防止除零
        """
        self.stats = stats
        self.mode = mode
        self.eps = eps

    def apply(self, tensor: torch.Tensor, modality: str) -> torch.Tensor:
        """
        在裁剪/旋转之后对张量进行标准化。
        tensor 形状: [1, H, W]，dtype=float32
        """
        s = self.stats.get(modality)
        if s is None:
            return tensor
        x = tensor
        if s.use_log1p:
            x = torch.log1p(torch.clamp(x, min=0.0))

        if self.mode == 'zscore':
            x = (x - float(s.mean)) / max(float(s.std), self.eps)
        else:
            raise ValueError(f'未知归一化模式: {self.mode}')
        return x


class RotationAugmentor:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def enumerate(self, is_training: bool) -> List[int]:
        if not self.enabled or not is_training:
            return [0]
        return [0, 1, 2, 3]

    def apply(self, tensor: torch.Tensor, rotation_index: int) -> torch.Tensor:
        if not self.enabled or rotation_index == 0:
            return tensor
        return torch.rot90(tensor, k=rotation_index, dims=[1, 2])


class HandcraftedFeatureExtractor:
    _registry: Dict[str, Callable[[np.ndarray], float]] = {
        'total_energy': lambda array: float(np.sum(array)),
    }

    def __init__(self, feature_flags: Mapping[str, Mapping[str, bool]]) -> None:
        self.enabled_features = [
            (modality, name)
            for modality, flags in feature_flags.items()
            for name, enabled in flags.items()
            if enabled
        ]
        self._validate()

    def _validate(self) -> None:
        missing = [name for _, name in self.enabled_features if name not in self._registry]
        if missing:
            raise ValueError(f"未注册的手工特征: {', '.join(sorted(set(missing)))}")

    def is_enabled(self) -> bool:
        return bool(self.enabled_features)

    def extract(self, sample_arrays: Mapping[str, np.ndarray]) -> Optional[torch.Tensor]:
        if not self.is_enabled():
            return None
        values = [
            self._registry[name](self._resolve_array(sample_arrays, modality))
            for modality, name in self.enabled_features
        ]
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _resolve_array(sample_arrays: Mapping[str, np.ndarray], modality: str) -> np.ndarray:
        if modality not in sample_arrays:
            raise KeyError(f"样本中缺少模态 {modality}，无法计算对应的手工特征")
        return sample_arrays[modality]


class ParticleDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[SampleRecord],
        class_labels: Sequence[int],
        modalities: Sequence[str],
        feature_extractor: Optional[HandcraftedFeatureExtractor],
        rotation_augmentor: RotationAugmentor,
        is_training: bool,
        crop_size: int,
        normalizer: Optional[Normalizer] = None,
        feature_scaler: Optional["HandcraftedFeatureScaler"] = None,
    ) -> None:
        self._base_samples = list(samples)
        self._class_labels = list(class_labels)
        self.modalities = list(modalities)
        self.feature_extractor = feature_extractor
        self.rotation_augmentor = rotation_augmentor
        self.is_training = is_training
        self.crop_size = crop_size
        self.normalizer = normalizer
        self.feature_scaler = feature_scaler

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self._expanded_samples: List[Tuple[SampleRecord, int]] = []
        for record in self._base_samples:
            for rotation_index in self.rotation_augmentor.enumerate(self.is_training):
                self._expanded_samples.append((record, rotation_index))

    def __len__(self) -> int:
        return len(self._expanded_samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        record, rotation_index = self._expanded_samples[idx]
        sample_arrays = {
            modality: self._load_array(record.modalities[modality])
            for modality in self.modalities
        }

        handcrafted_features = (
            self.feature_extractor.extract(sample_arrays) if self.feature_extractor else None
        )
        if handcrafted_features is not None and self.feature_scaler is not None:
            handcrafted_features = self.feature_scaler.apply(handcrafted_features)

        channel_tensors: List[torch.Tensor] = []
        for modality in self.modalities:
            t = self._prepare_tensor(sample_arrays[modality], rotation_index)
            if self.normalizer is not None:
                t = self.normalizer.apply(t, modality)
            channel_tensors.append(t)
        sample_tensor = torch.cat(channel_tensors, dim=0).float()

        if handcrafted_features is not None:
            return sample_tensor, record.label, handcrafted_features
        return sample_tensor, record.label

    @property
    def num_classes(self) -> int:
        return len(self._class_labels)

    def _load_array(self, file_path: str) -> np.ndarray:
        return np.loadtxt(file_path).astype(np.float64)

    def _prepare_tensor(self, sample_array: np.ndarray, rotation_index: int) -> torch.Tensor:
        tensor = self.transform(sample_array)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float64)
        tensor = self._center_crop(tensor)
        tensor = self.rotation_augmentor.apply(tensor, rotation_index)
        return tensor

    def _center_crop(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.crop_size <= 0:
            return tensor
        _, height, width = tensor.shape
        if self.crop_size > min(height, width):
            raise ValueError('裁剪大小超出原始尺寸范围')
        top = (height - self.crop_size) // 2
        left = (width - self.crop_size) // 2
        bottom = top + self.crop_size
        right = left + self.crop_size
        return tensor[:, top:bottom, left:right]


def _list_files(directory: str) -> List[str]:
    return [
        name
        for name in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, name))
    ]


def _normalize_key(file_name: str, modality: str) -> str:
    name, ext = os.path.splitext(file_name)
    normalized = name.replace(modality, '', 1)
    return f"{normalized}{ext}"


def collect_samples(data_dir: str, modalities: Sequence[str]) -> Tuple[List[SampleRecord], Dict[int, str]]:
    """返回 (样本列表, 标签映射表)。
    标签映射表: {连续标签int: 原始文件夹名str}，如 {0: '15', 1: '30', 2: '45', 3: '60'}。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f'数据目录不存在: {data_dir}')

    # 收集所有数字命名的子目录，按数值排序
    label_dirs: List[Tuple[str, str]] = []  # (folder_name, full_path)
    for name in os.listdir(data_dir):
        full = os.path.join(data_dir, name)
        if not os.path.isdir(full):
            continue
        try:
            int(name)
        except ValueError as exc:
            raise ValueError(f'标签目录名称必须为数字: {name}') from exc
        label_dirs.append((name, full))
    label_dirs.sort(key=lambda x: int(x[0]))

    # 按排序顺序分配连续标签 0, 1, 2, ...
    label_map: Dict[int, str] = {}  # 连续标签 → 原始文件夹名
    samples: List[SampleRecord] = []

    for seq_label, (folder_name, label_dir) in enumerate(label_dirs):
        label_map[seq_label] = folder_name

        modality_file_sets: List[set] = []
        modality_path_maps: Dict[str, Dict[str, str]] = {}
        for modality in modalities:
            modality_dir = os.path.join(label_dir, modality)
            if not os.path.isdir(modality_dir):
                raise FileNotFoundError(f'模态目录不存在: {modality_dir}')
            files = _list_files(modality_dir)
            if not files:
                raise RuntimeError(f'模态 {modality} 在标签 {folder_name} 中没有数据文件')
            normalized_map: Dict[str, str] = {}
            for file_name in files:
                key = _normalize_key(file_name, modality)
                normalized_map[key] = os.path.join(modality_dir, file_name)
            modality_file_sets.append(set(normalized_map.keys()))
            modality_path_maps[modality] = normalized_map

        common_files = set.intersection(*modality_file_sets) if modality_file_sets else set()
        if not common_files:
            raise RuntimeError(f'标签 {folder_name} 的不同模态文件名不一致，无法配对')

        for file_name in sorted(common_files):
            samples.append(
                SampleRecord(
                    label=seq_label,
                    modalities={
                        modality: modality_path_maps[modality][file_name]
                        for modality in modalities
                    },
                )
            )

    if not samples:
        raise RuntimeError(f'在目录 {data_dir} 中未找到任何样本')

    return samples, label_map


def split_samples(
    samples: Sequence[SampleRecord],
    train_ratio: float,
    seed: int,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    """
    按类别进行分层划分：对每个类别单独随机划分训练/验证，再合并，保证各类在两集合中的比例更稳定。

    规则：
    - 对每个类内索引使用给定 seed 的 RNG 独立打乱；
    - 训练集样本数为 floor(n_class * train_ratio)；
      当 n_class >= 2 时，强制至少 1 个进训练、至少 1 个进验证；
      当 n_class == 1 时，唯一样本分配给训练集；
    - 合并所有类别的划分结果。
    """
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio 必须在 (0, 1) 之间')

    if len(samples) < 2:
        raise ValueError('数据集中样本数量不足以进行划分，至少需要 2 个样本')

    # 收集每个类别的样本索引
    by_label: Dict[int, List[int]] = {}
    for idx, rec in enumerate(samples):
        by_label.setdefault(rec.label, []).append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    valid_indices: List[int] = []

    # 对每个类别独立随机、按比例切分
    for label in sorted(by_label.keys()):
        idxs = by_label[label]
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            k_train = 1  # 唯一样本放入训练集
        else:
            k_train = int(n * train_ratio)
            # 确保两端非空
            k_train = max(1, min(k_train, n - 1))

        train_indices.extend(idxs[:k_train])
        valid_indices.extend(idxs[k_train:])

    train_samples = [samples[i] for i in train_indices]
    valid_samples = [samples[i] for i in valid_indices]

    return train_samples, valid_samples


def _compute_standardization_stats(
    samples: Sequence[SampleRecord],
    modalities: Sequence[str],
    crop_size: int,
    per_modality_cfg: Mapping[str, Mapping[str, bool]],
) -> Dict[str, _ModalityStats]:
    """
    基于训练集样本、在中心裁剪后的 patch 上，按模态计算 z-score 统计量。
    - 对启用的模态：可选 log1p（在统计与应用时一致），可选忽略 0（仅用于统计均值/方差，应用时不改变 0 的值域行为）。
    - 统计使用 float64 以避免数值相消误差，结果保存为 float。
    """
    def center_crop_numpy(arr: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return arr
        h, w = arr.shape
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top:top+size, left:left+size]

    sums: Dict[str, float] = {}
    sumsqs: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    use_log1p: Dict[str, bool] = {}
    ignore_zero_for_stats: Dict[str, bool] = {}

    enabled_modalities = [m for m in modalities if per_modality_cfg.get(m, {}).get('enabled', False)]
    for m in enabled_modalities:
        sums[m] = 0.0
        sumsqs[m] = 0.0
        counts[m] = 0
        mins[m] = math.inf
        maxs[m] = -math.inf
        use_log1p[m] = bool(per_modality_cfg.get(m, {}).get('log1p', False))
        ignore_zero_for_stats[m] = bool(per_modality_cfg.get(m, {}).get('ignore_zero', False))

    for rec in samples:
        for modality in enabled_modalities:
            path = rec.modalities[modality]
            arr = np.loadtxt(path).astype(np.float64)
            arr = center_crop_numpy(arr, crop_size)

            if use_log1p[modality]:
                np.maximum(arr, 0.0, out=arr)
                np.log1p(arr, out=arr)

            if ignore_zero_for_stats[modality]:
                mask = arr != 0.0
                if not np.any(mask):
                    continue
                data = arr[mask]
            else:
                data = arr.ravel()

            sums[modality] += float(np.sum(data))
            sumsqs[modality] += float(np.sum(data * data))
            counts[modality] += int(data.size)
            dmin = float(np.min(data))
            dmax = float(np.max(data))
            mins[modality] = dmin if dmin < mins[modality] else mins[modality]
            maxs[modality] = dmax if dmax > maxs[modality] else maxs[modality]

    stats: Dict[str, _ModalityStats] = {}
    for modality in enabled_modalities:
        n = counts[modality]
        if n == 0:
            stats[modality] = _ModalityStats(
                mean=0.0, std=1.0, min=0.0, max=1.0,
                use_log1p=use_log1p[modality],
                ignore_zero_for_stats=ignore_zero_for_stats[modality],
            )
            continue
        mean = sums[modality] / n
        var = max((sumsqs[modality] / n) - mean * mean, 0.0)
        std = math.sqrt(var)
        stats[modality] = _ModalityStats(
            mean=mean,
            std=max(std, 1e-6),
            min=mins[modality],
            max=maxs[modality],
            use_log1p=use_log1p[modality],
            ignore_zero_for_stats=ignore_zero_for_stats[modality],
        )
    return stats


class HandcraftedFeatureScaler:
    """
    针对手工特征（标量向量）的 z-score 标准化器。
    - 使用训练集上计算的均值/方差对每个特征维度进行标准化；
    - 与 HandcraftedFeatureExtractor.enabled_features 顺序严格对齐。
    """

    def __init__(self, means: torch.Tensor, stds: torch.Tensor, feature_names: List[Tuple[str, str]]) -> None:
        if means.shape != stds.shape:
            raise ValueError("means 与 stds 形状不一致")
        if means.dim() != 1:
            raise ValueError("means/stds 必须为一维向量")
        self.means = means.float()
        self.stds = torch.clamp(stds.float(), min=1e-6)
        self.feature_names = list(feature_names)  # [(modality, feature_name), ...]

    def apply(self, features: torch.Tensor) -> torch.Tensor:
        # features: [D] 或 [N, D]
        if features.dim() == 1:
            return (features - self.means) / self.stds
        elif features.dim() == 2:
            return (features - self.means.unsqueeze(0)) / self.stds.unsqueeze(0)
        else:
            raise ValueError("features 维度错误，期望为 [D] 或 [N, D]")


def _compute_handcrafted_feature_stats(
    samples: Sequence[SampleRecord],
    extractor: HandcraftedFeatureExtractor,
) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[str, str]]]:
    """
    在训练集样本上计算手工特征的均值与标准差（z-score），用于后续标准化。
    注意：这里的手工特征基于完整帧（不裁剪、不旋转）计算，保证物理量一致性。
    返回：means, stds, feature_names（与 extractor.enabled_features 顺序一致）
    """
    if not extractor.is_enabled():
        return torch.tensor([]), torch.tensor([]), []

    # 初始化累积量
    feat_dim = len(extractor.enabled_features)
    sums = np.zeros((feat_dim,), dtype=np.float64)
    sumsqs = np.zeros((feat_dim,), dtype=np.float64)
    count = 0

    # 为避免重复加载，相同样本内按所需模态一次性读取
    needed_modalities = sorted({m for (m, _) in extractor.enabled_features})
    for rec in samples:
        sample_arrays: Dict[str, np.ndarray] = {}
        for modality in needed_modalities:
            path = rec.modalities[modality]
            sample_arrays[modality] = np.loadtxt(path).astype(np.float64)

        vec = extractor.extract(sample_arrays)
        if vec is None:
            continue
        v = vec.numpy().astype(np.float64)
        sums += v
        sumsqs += v * v
        count += 1

    if count == 0:
        means = np.zeros((feat_dim,), dtype=np.float64)
        stds = np.ones((feat_dim,), dtype=np.float64)
    else:
        means = sums / count
        var = np.maximum(sumsqs / count - means * means, 0.0)
        stds = np.sqrt(var)

    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32), list(extractor.enabled_features)


def _has_enabled_features(feature_flags: Mapping[str, Mapping[str, bool]]) -> bool:
    return any(
        enabled
        for flags in feature_flags.values()
        for enabled in flags.values()
    )


def build_datasets(
    data_dir: str,
    modalities: Sequence[str],
    train_ratio: float,
    seed: int,
    rotation_enabled: bool,
    feature_flags: Mapping[str, Mapping[str, bool]],
    crop_size: int,
    standardization_mode: str = 'zscore',
    per_modality_standardization: Optional[Mapping[str, Mapping[str, bool]]] = None,
    handcrafted_standardize: bool = True,
    handcrafted_stats_path: Optional[str] = None,
) -> Tuple[ParticleDataset, ParticleDataset, Dict[int, str]]:
    samples, label_map = collect_samples(data_dir, modalities)
    class_labels = sorted({record.label for record in samples})
    train_samples, valid_samples = split_samples(samples, train_ratio, seed)

    # 基于训练集统计量构建 Normalizer（按模态配置）
    normalizer: Optional[Normalizer] = None
    if per_modality_standardization:
        stats = _compute_standardization_stats(
            samples=train_samples,
            modalities=modalities,
            crop_size=crop_size,
            per_modality_cfg=per_modality_standardization,
        )
        # 若没有任何模态启用，则 stats 可能为空
        if stats:
            normalizer = Normalizer(stats=stats, mode=standardization_mode, eps=1e-6)

    filtered_feature_flags = {
        modality: feature_flags.get(modality, {})
        for modality in modalities
    }

    feature_extractor = (
        HandcraftedFeatureExtractor(filtered_feature_flags)
        if _has_enabled_features(filtered_feature_flags)
        else None
    )
    feature_scaler: Optional[HandcraftedFeatureScaler] = None
    if feature_extractor is not None and handcrafted_standardize:
        means, stds, names = _compute_handcrafted_feature_stats(train_samples, feature_extractor)
        if len(names) > 0:
            feature_scaler = HandcraftedFeatureScaler(means, stds, names)
            # 可选：保存统计量到文件，便于复现实验
            if handcrafted_stats_path:
                try:
                    import json
                    os.makedirs(os.path.dirname(handcrafted_stats_path), exist_ok=True)
                    payload = {
                        'features': [
                            {
                                'modality': m,
                                'name': n,
                                'mean': float(means[i].item()),
                                'std': float(stds[i].item()),
                            }
                            for i, (m, n) in enumerate(names)
                        ]
                    }
                    with open(handcrafted_stats_path, 'w', encoding='utf-8') as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    # 不因保存失败而中断训练
                    print(f"[WARN] 无法保存手工特征统计量到 {handcrafted_stats_path}: {e}")
    rotation_augmentor = RotationAugmentor(rotation_enabled)

    train_dataset = ParticleDataset(
        samples=train_samples,
        class_labels=class_labels,
        modalities=modalities,
        feature_extractor=feature_extractor,
        rotation_augmentor=rotation_augmentor,
        is_training=True,
        crop_size=crop_size,
        normalizer=normalizer,
        feature_scaler=feature_scaler,
    )

    valid_dataset = ParticleDataset(
        samples=valid_samples,
        class_labels=class_labels,
        modalities=modalities,
        feature_extractor=feature_extractor,
        rotation_augmentor=rotation_augmentor,
        is_training=False,
        crop_size=crop_size,
        normalizer=normalizer,
        feature_scaler=feature_scaler,
    )

    return train_dataset, valid_dataset, label_map