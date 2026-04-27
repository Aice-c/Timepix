"""ResNet18 Timepix model variant that keeps the first maxpool layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .base import ModelOutput
from .fusion import FeatureFusion


class ResNet18MaxPoolBackbone(nn.Module):
    def __init__(
        self,
        input_channels: int,
        feature_dim: int = 256,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        self.model.conv1 = nn.Conv2d(
            input_channels,
            64,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.model.fc = nn.Linear(self.model.fc.in_features, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResNet18MaxPoolTimepix(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        task: str = "classification",
        handcrafted_dim: int = 0,
        fusion_mode: str = "none",
        feature_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        kernel_size: int = 2,
        stride: int = 1,
        padding: int = 0,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.task = task
        self.backbone = ResNet18MaxPoolBackbone(input_channels, feature_dim, kernel_size, stride, padding, pretrained)
        self.fusion = FeatureFusion(feature_dim, handcrafted_dim, fusion_mode)
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
            regression = torch.sigmoid(output).squeeze(-1)
            return ModelOutput(regression=regression, features=fused)
        probabilities = F.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return ModelOutput(logits=output, probabilities=probabilities, predictions=predictions, features=fused)
