"""Model registry."""

from __future__ import annotations

import torch.nn as nn

from .resnet import ResNet18Timepix
from .resnet18_original import ResNet18OriginalTimepix
from .resnet_maxpool import ResNet18MaxPoolTimepix
from .shallow import build_shallow_cnn, build_shallow_resnet


def build_model(
    cfg: dict,
    input_channels: int,
    num_classes: int,
    task: str,
    handcrafted_dim: int,
) -> nn.Module:
    model_cfg = cfg.get("model", {})
    name = model_cfg.get("name", "resnet18")
    fusion_mode = model_cfg.get("fusion_mode", "none")
    common = {
        "input_channels": input_channels,
        "num_classes": num_classes,
        "task": task,
        "handcrafted_dim": handcrafted_dim,
        "fusion_mode": fusion_mode,
        "dropout": float(model_cfg.get("dropout", cfg.get("training", {}).get("dropout", 0.1))),
    }
    conv1_kernel_size = int(model_cfg.get("conv1_kernel_size", model_cfg.get("kernel_size", 2)))
    conv1_stride = int(model_cfg.get("conv1_stride", model_cfg.get("stride", 1)))
    conv1_padding = int(model_cfg.get("conv1_padding", model_cfg.get("padding", 0)))

    if name in {"resnet18", "resnet18_no_maxpool"}:
        return ResNet18Timepix(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            pretrained=bool(model_cfg.get("pretrained", False)),
        )
    if name in {"resnet18_maxpool", "resnet18_with_maxpool"}:
        return ResNet18MaxPoolTimepix(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            pretrained=bool(model_cfg.get("pretrained", False)),
        )
    if name == "resnet18_original":
        return ResNet18OriginalTimepix(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            pretrained=bool(model_cfg.get("pretrained", False)),
        )
    if name == "shallow_cnn":
        return build_shallow_cnn(**common, hidden_dim=int(model_cfg.get("hidden_dim", 128)))
    if name == "shallow_resnet":
        return build_shallow_resnet(**common, hidden_dim=int(model_cfg.get("hidden_dim", 128)))

    raise ValueError(f"Unknown model: {name}")

