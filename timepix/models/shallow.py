"""Shallow CNN architectures for sparse Timepix tracks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput
from .fusion import FeatureFusion


class ShallowCNNBackbone(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.flatten(x, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class ShallowResNetBackbone(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(32, 64, stride=1),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.feature_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return torch.flatten(x, 1)


class ShallowTimepixModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        task: str = "classification",
        handcrafted_dim: int = 0,
        fusion_mode: str = "none",
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.task = task
        self.backbone = backbone
        self.fusion = FeatureFusion(backbone.feature_dim, handcrafted_dim, fusion_mode)
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


def build_shallow_cnn(**kwargs) -> ShallowTimepixModel:
    input_channels = kwargs.pop("input_channels")
    return ShallowTimepixModel(ShallowCNNBackbone(input_channels), **kwargs)


def build_shallow_resnet(**kwargs) -> ShallowTimepixModel:
    input_channels = kwargs.pop("input_channels")
    return ShallowTimepixModel(ShallowResNetBackbone(input_channels), **kwargs)

