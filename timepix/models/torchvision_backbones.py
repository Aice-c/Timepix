"""Torchvision backbone adapters for Timepix experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    convnext_tiny,
    densenet121,
    efficientnet_b0,
)
from torchvision.models.vision_transformer import VisionTransformer

from .base import ModelOutput
from .fusion import FeatureFusion


class TimepixBackboneModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        num_classes: int,
        task: str = "classification",
        handcrafted_dim: int = 0,
        fusion_mode: str = "none",
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.backbone = backbone
        self.fusion = FeatureFusion(backbone_dim, handcrafted_dim, fusion_mode)
        out_dim = 1 if task == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Linear(self.fusion.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> ModelOutput:
        features = self.backbone(image)
        fused = self.fusion(features, handcrafted)
        output = self.head(fused)
        if self.task == "regression":
            return ModelOutput(regression=torch.sigmoid(output).squeeze(-1), features=fused)
        probabilities = F.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return ModelOutput(logits=output, probabilities=probabilities, predictions=predictions, features=fused)


class DenseNet121Backbone(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = densenet121(weights=weights)
        self.model.features.conv0 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.model.classifier = nn.Linear(self.model.classifier.in_features, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EfficientNetB0Backbone(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        stem_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            input_channels,
            stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=stem_conv.bias is not None,
        )
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvNeXtTinyBackbone(nn.Module):
    def __init__(self, input_channels: int, feature_dim: int = 256, pretrained: bool = False) -> None:
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = convnext_tiny(weights=weights)
        stem_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            input_channels,
            stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
            bias=stem_conv.bias is not None,
        )
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ViTTinyBackbone(nn.Module):
    """Small ViT adapted to native 50x50 Timepix matrices."""

    def __init__(
        self,
        input_channels: int,
        feature_dim: int = 256,
        image_size: int = 50,
        patch_size: int = 10,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if pretrained:
            raise ValueError("vit_tiny does not provide pretrained weights in this project adapter")
        if image_size % patch_size != 0:
            raise ValueError(f"vit_tiny requires image_size divisible by patch_size, got {image_size}/{patch_size}")
        self.model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=12,
            num_heads=3,
            hidden_dim=192,
            mlp_dim=768,
            dropout=0.0,
            attention_dropout=0.0,
            num_classes=feature_dim,
        )
        self.model.conv_proj = nn.Conv2d(
            input_channels,
            192,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_densenet121(**kwargs) -> TimepixBackboneModel:
    input_channels = kwargs.pop("input_channels")
    feature_dim = kwargs.pop("feature_dim", 256)
    pretrained = kwargs.pop("pretrained", False)
    backbone = DenseNet121Backbone(input_channels, feature_dim, pretrained)
    return TimepixBackboneModel(backbone, backbone.feature_dim, **kwargs)


def build_efficientnet_b0(**kwargs) -> TimepixBackboneModel:
    input_channels = kwargs.pop("input_channels")
    feature_dim = kwargs.pop("feature_dim", 256)
    pretrained = kwargs.pop("pretrained", False)
    backbone = EfficientNetB0Backbone(input_channels, feature_dim, pretrained)
    return TimepixBackboneModel(backbone, backbone.feature_dim, **kwargs)


def build_convnext_tiny(**kwargs) -> TimepixBackboneModel:
    input_channels = kwargs.pop("input_channels")
    feature_dim = kwargs.pop("feature_dim", 256)
    pretrained = kwargs.pop("pretrained", False)
    backbone = ConvNeXtTinyBackbone(input_channels, feature_dim, pretrained)
    return TimepixBackboneModel(backbone, backbone.feature_dim, **kwargs)


def build_vit_tiny(**kwargs) -> TimepixBackboneModel:
    input_channels = kwargs.pop("input_channels")
    feature_dim = kwargs.pop("feature_dim", 256)
    pretrained = kwargs.pop("pretrained", False)
    image_size = kwargs.pop("image_size", 50)
    patch_size = kwargs.pop("patch_size", 10)
    backbone = ViTTinyBackbone(input_channels, feature_dim, image_size, patch_size, pretrained)
    return TimepixBackboneModel(backbone, backbone.feature_dim, **kwargs)
