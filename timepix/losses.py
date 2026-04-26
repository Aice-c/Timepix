"""Loss functions for Timepix experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EarthMoverDistanceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        angle_values: list[float],
        p: int = 2,
        label_encoding: str = "onehot",
        gaussian_sigma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.p = p
        self.label_encoding = label_encoding
        self.gaussian_sigma = gaussian_sigma
        self.register_buffer("angle_values", torch.tensor(angle_values, dtype=torch.float32))

    def _encode_targets(self, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 2 and targets.shape[-1] == self.num_classes:
            return targets
        if self.label_encoding == "onehot":
            encoded = torch.zeros(targets.shape[0], self.num_classes, device=targets.device)
            encoded.scatter_(1, targets.long().unsqueeze(1), 1.0)
            return encoded
        if self.label_encoding == "gaussian":
            true_angles = self.angle_values[targets.long()]
            diffs = self.angle_values.unsqueeze(0) - true_angles.unsqueeze(1)
            return F.softmax(-(diffs**2) / (2.0 * self.gaussian_sigma**2), dim=-1)
        raise ValueError(f"Unknown label encoding: {self.label_encoding}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_probs = F.softmax(logits, dim=-1)
        target_probs = self._encode_targets(targets)
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)
        diff = (target_cdf[:, :-1] - pred_cdf[:, :-1]).abs()
        if self.p == 1:
            return diff.sum(dim=-1).mean()
        return (diff**self.p).sum(dim=-1).mean()


def build_loss(cfg: dict, num_classes: int, label_map: dict[int, str]) -> nn.Module:
    task = cfg.get("task", {}).get("type", "classification")
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "cross_entropy")
    if task == "regression":
        if name == "mse":
            return nn.MSELoss()
        return nn.SmoothL1Loss()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    if name == "emd":
        angle_values = [float(label_map[i]) for i in range(num_classes)]
        return EarthMoverDistanceLoss(
            num_classes=num_classes,
            angle_values=angle_values,
            p=int(loss_cfg.get("emd_p", 2)),
            label_encoding=loss_cfg.get("label_encoding", "onehot"),
            gaussian_sigma=float(loss_cfg.get("gaussian_sigma", 2.0)),
        )
    raise ValueError(f"Unknown loss: {name}")

