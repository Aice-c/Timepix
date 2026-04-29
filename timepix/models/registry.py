"""Model registry."""

from __future__ import annotations

import torch.nn as nn

from .dual_stream import (
    DualStreamConcatAuxTimepix,
    DualStreamGMUAuxTimepix,
    ToAConditionedFiLMTimepix,
    WarmStartedExpertGateTimepix,
)
from .handcrafted import HandcraftedMLPTimepix
from .resnet import ResNet18Timepix
from .resnet18_original import ResNet18OriginalTimepix
from .resnet_maxpool import ResNet18MaxPoolTimepix
from .shallow import build_shallow_cnn, build_shallow_resnet
from .torchvision_backbones import (
    build_convnext_tiny,
    build_densenet121,
    build_efficientnet_b0,
    build_vit_tiny,
)


def _default_image_size(cfg: dict) -> int:
    sample_shape = cfg.get("dataset", {}).get("sample_shape", [50])
    if isinstance(sample_shape, int):
        return sample_shape
    if isinstance(sample_shape, (list, tuple)) and sample_shape:
        return int(sample_shape[0])
    return 50


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
    if name == "handcrafted_mlp":
        return HandcraftedMLPTimepix(
            num_classes=num_classes,
            task=task,
            handcrafted_dim=handcrafted_dim,
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", cfg.get("training", {}).get("dropout", 0.1))),
        )
    if name == "shallow_cnn":
        return build_shallow_cnn(**common, hidden_dim=int(model_cfg.get("hidden_dim", 128)))
    if name == "shallow_resnet":
        return build_shallow_resnet(**common, hidden_dim=int(model_cfg.get("hidden_dim", 128)))
    if name == "densenet121":
        return build_densenet121(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            pretrained=bool(model_cfg.get("pretrained", False)),
        )
    if name == "efficientnet_b0":
        return build_efficientnet_b0(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            pretrained=bool(model_cfg.get("pretrained", False)),
        )
    if name == "convnext_tiny":
        return build_convnext_tiny(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            pretrained=bool(model_cfg.get("pretrained", False)),
        )
    if name == "vit_tiny":
        return build_vit_tiny(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            pretrained=bool(model_cfg.get("pretrained", False)),
            image_size=int(model_cfg.get("image_size", _default_image_size(cfg))),
            patch_size=int(model_cfg.get("patch_size", 10)),
        )
    if name == "dual_stream_concat_aux":
        return DualStreamConcatAuxTimepix(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            pretrained=bool(model_cfg.get("pretrained", False)),
        )
    if name == "dual_stream_gmu_aux":
        gate_cfg = model_cfg.get("gate", {})
        return DualStreamGMUAuxTimepix(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            pretrained=bool(model_cfg.get("pretrained", False)),
            gate_init_bias_to_tot=float(gate_cfg.get("init_bias_to_tot", 2.0)),
        )
    if name == "toa_conditioned_film":
        film_cfg = model_cfg.get("film", {})
        return ToAConditionedFiLMTimepix(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            pretrained=bool(model_cfg.get("pretrained", False)),
            film_hidden_dim=int(film_cfg.get("hidden_dim", model_cfg.get("feature_dim", 256))),
            film_zero_init=bool(film_cfg.get("zero_init", True)),
        )
    if name == "warm_started_expert_gate":
        expert_gate_cfg = model_cfg.get("expert_gate", {})
        return WarmStartedExpertGateTimepix(
            **common,
            feature_dim=int(model_cfg.get("feature_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 512)),
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=conv1_padding,
            pretrained=bool(model_cfg.get("pretrained", False)),
            gate_hidden_dim=int(expert_gate_cfg.get("hidden_dim", model_cfg.get("feature_dim", 256))),
            gate_dropout=float(expert_gate_cfg.get("dropout", model_cfg.get("dropout", 0.1))),
            gate_init_bias_to_candidate=float(expert_gate_cfg.get("init_bias_to_candidate", -2.0)),
            gate_include_logits=bool(expert_gate_cfg.get("include_logits", True)),
        )

    raise ValueError(f"Unknown model: {name}")

