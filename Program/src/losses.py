"""
损失函数模块：支持交叉熵和 Earth Mover's Distance (Wasserstein) 损失。

EMD 损失利用角度类别的有序性，通过比较预测分布和目标分布的累积分布函数（CDF）
来计算距离，使得预测偏到相邻类别的惩罚远小于偏到远处类别。

参考文献：
    Okabe et al., Nature Communications (2024)

配置组合说明：
| loss_type       | label_encoding | 含义                                              |
|:----------------|:---------------|:--------------------------------------------------|
| cross_entropy   | onehot         | 原始方案：标准交叉熵 + one-hot 标签               |
| cross_entropy   | gaussian       | 交叉熵 + 软标签（此时取 argmax 退化为 one-hot）   |
| emd             | onehot         | EMD 损失 + one-hot 标签（CDF 为阶跃函数）         |
| emd             | gaussian       | **推荐方案**：EMD 损失 + 高斯软标签               |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class EarthMoverDistanceLoss(nn.Module):
    """
    一维 p 阶 Wasserstein 距离（Earth Mover's Distance）损失函数。

    数学定义：
        对于 K 个有序类别，给定真实分布 y 和预测分布 ŷ（均为归一化概率分布），

            L_EMD = Σ_{k=1}^{K-1} |CDF(y)_k - CDF(ŷ)_k|^p

        其中 CDF(v)_k = Σ_{i=1}^{k} v_i（前缀累加和）。

    Parameters
    ----------
    num_classes : int
        类别数量（有序类别）
    p : int
        Wasserstein 距离的阶数，默认 p=2。
        p=2 训练收敛更快（梯度更平滑），p=1 直接对应角度误差。
    label_encoding : str
        标签编码方式，'onehot' 或 'gaussian'
    angle_values : list of float
        各类别对应的角度值（有序），用于高斯软标签计算
    gaussian_sigma : float
        高斯软标签的宽度参数（度），仅在 label_encoding='gaussian' 时使用
    """

    def __init__(
        self,
        num_classes: int,
        p: int = 2,
        label_encoding: str = 'onehot',
        angle_values: Optional[List[float]] = None,
        gaussian_sigma: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.label_encoding = label_encoding
        self.gaussian_sigma = gaussian_sigma

        # 注册角度值为 buffer（不参与梯度计算，但会随模型移动到 GPU）
        if angle_values is not None:
            self.register_buffer(
                'angle_values',
                torch.tensor(angle_values, dtype=torch.float32),
            )
        else:
            self.register_buffer(
                'angle_values',
                torch.arange(num_classes, dtype=torch.float32),
            )

    def _encode_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        将整数标签转换为目标概率分布。

        Parameters
        ----------
        targets : Tensor
            整数标签 (batch_size,) 或已编码的分布 (batch_size, num_classes)

        Returns
        -------
        Tensor
            目标概率分布 (batch_size, num_classes)
        """
        # 如果已经是分布形式，直接返回
        if targets.dim() == 2 and targets.shape[-1] == self.num_classes:
            return targets

        batch_size = targets.shape[0]

        if self.label_encoding == 'onehot':
            # One-hot 编码: y_k = 1 if k == k_true, else 0
            encoded = torch.zeros(
                batch_size, self.num_classes,
                device=targets.device, dtype=torch.float32,
            )
            encoded.scatter_(1, targets.long().unsqueeze(1), 1.0)
            return encoded

        elif self.label_encoding == 'gaussian':
            # 高斯软标签:
            #   y_k = (1/Z) * exp(-(θ_k - θ_true)^2 / (2σ^2))
            # 其中 Z 为归一化常数，确保 Σ_k y_k = 1
            true_angles = self.angle_values[targets.long()]  # (batch_size,)
            # θ_k - θ_true for all k
            angle_diffs = self.angle_values.unsqueeze(0) - true_angles.unsqueeze(1)  # (batch_size, num_classes)
            # 高斯核的对数值
            log_probs = -(angle_diffs ** 2) / (2.0 * self.gaussian_sigma ** 2)
            # 用 softmax 做归一化（等价于 exp + normalize，且数值稳定——内部自动减最大值）
            encoded = F.softmax(log_probs, dim=-1)
            return encoded

        else:
            raise ValueError(f"不支持的标签编码方式: {self.label_encoding}")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 EMD 损失。

        Parameters
        ----------
        predictions : Tensor
            模型原始输出（logits），shape = (batch_size, num_classes)
        targets : Tensor
            目标标签，整数 (batch_size,) 或分布 (batch_size, num_classes)

        Returns
        -------
        Tensor
            batch 内的平均损失（标量）
        """
        # Step 1: 对 logits 做 softmax 得到预测概率分布
        # F.softmax 内部已做减最大值处理，保证数值稳定性
        pred_probs = F.softmax(predictions, dim=-1)

        # Step 2: 将目标转为概率分布形式
        target_probs = self._encode_targets(targets)

        # Step 3: 计算累积分布函数 CDF(v)_k = Σ_{i=1}^{k} v_i
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)

        # Step 4: 计算 L_EMD = Σ_{k=1}^{K-1} |CDF(y)_k - CDF(ŷ)_k|^p
        # 排除最后一个 CDF 值（恒为 1，两者相等，差为 0）
        cdf_diff = (target_cdf[:, :-1] - pred_cdf[:, :-1]).abs()

        if self.p == 1:
            loss_per_sample = cdf_diff.sum(dim=-1)
        elif self.p == 2:
            loss_per_sample = (cdf_diff ** 2).sum(dim=-1)
        else:
            loss_per_sample = (cdf_diff ** self.p).sum(dim=-1)

        # Step 5: batch 平均
        return loss_per_sample.mean()


class CrossEntropyLossWrapper(nn.Module):
    """
    交叉熵损失的封装，使接口与 EMD Loss 保持一致。

    当 targets 是软标签分布时，取 argmax 转为类别索引后调用标准交叉熵；
    当 targets 是整数标签时，直接调用标准交叉熵。

    Parameters
    ----------
    weight : Tensor, optional
        类别权重
    """

    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失。

        Parameters
        ----------
        predictions : Tensor
            模型原始输出（logits），shape = (batch_size, num_classes)
        targets : Tensor
            目标标签，整数 (batch_size,) 或分布 (batch_size, num_classes)
        """
        # 如果 targets 是分布形式（如高斯软标签），取 argmax 退化为整数标签
        if targets.dim() == 2:
            targets = targets.argmax(dim=-1)
        return self.ce(predictions, targets)


def build_loss_function(cfg, num_classes: int, label_map: dict) -> nn.Module:
    """
    根据配置构建损失函数（工厂函数）。

    Parameters
    ----------
    cfg : config
        配置对象，需包含 loss_type, label_encoding 等属性
    num_classes : int
        类别数量
    label_map : dict
        标签映射表 {连续标签int: 原始文件夹名str}，如 {0: '15', 1: '30'}

    Returns
    -------
    nn.Module
        损失函数实例
    """
    # 从 label_map 提取有序的角度值列表
    angle_values = [float(label_map[i]) for i in range(num_classes)]

    loss_type = getattr(cfg, 'loss_type', 'cross_entropy')
    label_encoding = getattr(cfg, 'label_encoding', 'onehot')

    print(f"[Loss] 损失函数类型: {loss_type}")
    print(f"[Loss] 标签编码方式: {label_encoding}")
    print(f"[Loss] 角度类别列表: {angle_values}")

    if loss_type == 'cross_entropy':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = torch.tensor(cfg.weight, dtype=torch.float32).to(device)
        criterion = CrossEntropyLossWrapper(weight=weights)
        if label_encoding == 'gaussian':
            print("[Loss] 注意: cross_entropy + gaussian 组合下，软标签将被 argmax 退化为整数标签")
        return criterion

    elif loss_type == 'emd':
        emd_p = getattr(cfg, 'emd_p', 2)
        gaussian_sigma = getattr(cfg, 'gaussian_sigma', 2.0)
        print(f"[Loss] EMD 阶数 p={emd_p}")
        if label_encoding == 'gaussian':
            print(f"[Loss] 高斯软标签 σ={gaussian_sigma}°")
        criterion = EarthMoverDistanceLoss(
            num_classes=num_classes,
            p=emd_p,
            label_encoding=label_encoding,
            angle_values=angle_values,
            gaussian_sigma=gaussian_sigma,
        )
        return criterion

    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")


def compute_angle_mae(
    logits: torch.Tensor,
    targets: torch.Tensor,
    angle_values: torch.Tensor,
) -> dict:
    """
    计算平均绝对角度误差（MAE）。

    提供两种预测角度提取方式：
        - argmax:   θ̂ = θ_{argmax(ŷ)}              （取最大概率类别对应的角度）
        - weighted: θ̂ = Σ_k ŷ_k · θ_k              （概率加权平均角度）

    Parameters
    ----------
    logits : Tensor
        模型输出 logits (batch_size, num_classes)
    targets : Tensor
        整数标签 (batch_size,)
    angle_values : Tensor
        各类别对应的角度值 (num_classes,)

    Returns
    -------
    dict
        {'ae_argmax': float, 'ae_weighted': float, 'count': int}
        返回绝对误差总和和样本数，便于跨 batch 累加后计算平均值
    """
    probs = F.softmax(logits.detach(), dim=-1)
    true_angles = angle_values[targets.long()]  # (batch_size,)

    # 方式一：argmax 预测角度
    pred_indices = logits.detach().argmax(dim=-1)
    pred_angles_argmax = angle_values[pred_indices]
    ae_argmax = (pred_angles_argmax - true_angles).abs().sum().item()

    # 方式二：加权均值预测角度
    pred_angles_weighted = (probs * angle_values.unsqueeze(0)).sum(dim=-1)
    ae_weighted = (pred_angles_weighted - true_angles).abs().sum().item()

    return {
        'ae_argmax': ae_argmax,
        'ae_weighted': ae_weighted,
        'count': targets.shape[0],
    }
