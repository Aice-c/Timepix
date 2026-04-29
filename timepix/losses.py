"""Loss functions for Timepix experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AngleTargetMixin:
    def __init__(
        self,
        num_classes: int,
        angle_values: list[float],
        label_encoding: str = "onehot",
        gaussian_sigma: float = 2.0,
    ) -> None:
        self.num_classes = num_classes
        self.label_encoding = label_encoding
        self.gaussian_sigma = gaussian_sigma
        self.register_buffer("angle_values", torch.tensor(angle_values, dtype=torch.float32))

    def _encode_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 2 and targets.shape[-1] == self.num_classes:
            return targets.float()
        if self.label_encoding == "onehot":
            encoded = torch.zeros(targets.shape[0], self.num_classes, device=targets.device)
            encoded.scatter_(1, targets.long().unsqueeze(1), 1.0)
            return encoded
        if self.label_encoding == "gaussian":
            true_angles = self.angle_values[targets.long()]
            diffs = self.angle_values.unsqueeze(0) - true_angles.unsqueeze(1)
            return F.softmax(-(diffs**2) / (2.0 * self.gaussian_sigma**2), dim=-1)
        raise ValueError(f"Unknown label encoding: {self.label_encoding}")


class SoftTargetCrossEntropyLoss(nn.Module, AngleTargetMixin):
    def __init__(
        self,
        num_classes: int,
        angle_values: list[float],
        label_encoding: str = "onehot",
        gaussian_sigma: float = 2.0,
    ) -> None:
        nn.Module.__init__(self)
        AngleTargetMixin.__init__(self, num_classes, angle_values, label_encoding, gaussian_sigma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_encoding == "onehot" and targets.dim() == 1:
            return F.cross_entropy(logits, targets.long())
        target_probs = self._encode_targets(targets)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_probs * log_probs).sum(dim=-1).mean()


class EarthMoverDistanceLoss(nn.Module, AngleTargetMixin):
    def __init__(
        self,
        num_classes: int,
        angle_values: list[float],
        p: int = 2,
        label_encoding: str = "onehot",
        gaussian_sigma: float = 2.0,
        angle_weighted: bool = False,
        normalize_by_angle_range: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        AngleTargetMixin.__init__(self, num_classes, angle_values, label_encoding, gaussian_sigma)
        self.p = p
        self.angle_weighted = angle_weighted
        self.normalize_by_angle_range = normalize_by_angle_range
        gaps = self.angle_values[1:] - self.angle_values[:-1]
        angle_range = torch.clamp(self.angle_values[-1] - self.angle_values[0], min=1.0)
        if normalize_by_angle_range:
            gaps = gaps / angle_range
        self.register_buffer("angle_gaps", gaps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_probs = F.softmax(logits, dim=-1)
        target_probs = self._encode_targets(targets)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)
        diff = (target_cdf[:, :-1] - pred_cdf[:, :-1]).abs()
        if self.angle_weighted:
            diff = diff * self.angle_gaps.unsqueeze(0)
        if self.p == 1:
            return diff.sum(dim=-1).mean()
        return (diff**self.p).sum(dim=-1).mean()


class CEExpectedAngleMAELoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        angle_values: list[float],
        weight: float = 0.1,
        label_encoding: str = "onehot",
        gaussian_sigma: float = 2.0,
        normalize_by_angle_range: bool = True,
    ) -> None:
        super().__init__()
        self.ce = SoftTargetCrossEntropyLoss(num_classes, angle_values, label_encoding, gaussian_sigma)
        self.weight = weight
        self.normalize_by_angle_range = normalize_by_angle_range
        self.register_buffer("angle_values", torch.tensor(angle_values, dtype=torch.float32))
        self.register_buffer("angle_range", torch.clamp(self.angle_values[-1] - self.angle_values[0], min=1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=-1)
        pred_angles = probs @ self.angle_values
        if targets.dim() == 2:
            true_angles = targets.float() @ self.angle_values
        else:
            true_angles = self.angle_values[targets.long()]
        mae = (pred_angles - true_angles).abs()
        if self.normalize_by_angle_range:
            mae = mae / self.angle_range
        return ce_loss + self.weight * mae.mean()


class CEWithEMDLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        angle_values: list[float],
        emd_weight: float = 0.1,
        emd_p: int = 1,
        label_encoding: str = "onehot",
        gaussian_sigma: float = 2.0,
        angle_weighted: bool = True,
        normalize_by_angle_range: bool = True,
    ) -> None:
        super().__init__()
        self.ce = SoftTargetCrossEntropyLoss(num_classes, angle_values, label_encoding, gaussian_sigma)
        self.emd = EarthMoverDistanceLoss(
            num_classes=num_classes,
            angle_values=angle_values,
            p=emd_p,
            label_encoding=label_encoding,
            gaussian_sigma=gaussian_sigma,
            angle_weighted=angle_weighted,
            normalize_by_angle_range=normalize_by_angle_range,
        )
        self.emd_weight = emd_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, targets) + self.emd_weight * self.emd(logits, targets)


def build_loss(cfg: dict, num_classes: int, label_map: dict[int, str]) -> nn.Module:
    task = cfg.get("task", {}).get("type", "classification")
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "cross_entropy")
    if task == "regression":
        if name == "mse":
            return nn.MSELoss()
        return nn.SmoothL1Loss()

    angle_values = [float(label_map[i]) for i in range(num_classes)]
    label_encoding = loss_cfg.get("label_encoding", "onehot")
    gaussian_sigma = float(loss_cfg.get("gaussian_sigma", 2.0))
    if name == "cross_entropy":
        return SoftTargetCrossEntropyLoss(
            num_classes=num_classes,
            angle_values=angle_values,
            label_encoding=label_encoding,
            gaussian_sigma=gaussian_sigma,
        )
    if name == "ce_expected_mae":
        return CEExpectedAngleMAELoss(
            num_classes=num_classes,
            angle_values=angle_values,
            weight=float(loss_cfg.get("expected_mae_weight", 0.1)),
            label_encoding=label_encoding,
            gaussian_sigma=gaussian_sigma,
            normalize_by_angle_range=bool(loss_cfg.get("normalize_by_angle_range", True)),
        )
    if name == "ce_emd":
        return CEWithEMDLoss(
            num_classes=num_classes,
            angle_values=angle_values,
            emd_weight=float(loss_cfg.get("emd_weight", 0.1)),
            emd_p=int(loss_cfg.get("emd_p", 1)),
            label_encoding=label_encoding,
            gaussian_sigma=gaussian_sigma,
            angle_weighted=bool(loss_cfg.get("emd_angle_weighted", True)),
            normalize_by_angle_range=bool(loss_cfg.get("normalize_by_angle_range", True)),
        )
    if name == "emd":
        return EarthMoverDistanceLoss(
            num_classes=num_classes,
            angle_values=angle_values,
            p=int(loss_cfg.get("emd_p", 2)),
            label_encoding=label_encoding,
            gaussian_sigma=gaussian_sigma,
            angle_weighted=bool(loss_cfg.get("emd_angle_weighted", False)),
            normalize_by_angle_range=bool(loss_cfg.get("normalize_by_angle_range", True)),
        )
    raise ValueError(f"Unknown loss: {name}")
