"""Dual-stream Alpha ToT/ToA fusion models for A4c."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .base import ModelOutput
from .fusion import FeatureFusion
from .resnet import ResNet18Backbone


def _split_tot_toa(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if image.shape[1] < 2:
        raise ValueError("A4c dual-stream models require ToT+ToA input channels")
    return image[:, :1], image[:, 1:2]


def _make_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


def _classification_output(
    logits: torch.Tensor,
    *,
    features: torch.Tensor,
    aux_logits: dict[str, torch.Tensor] | None = None,
    diagnostics: dict[str, torch.Tensor] | None = None,
) -> ModelOutput:
    probabilities = F.softmax(logits, dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    return ModelOutput(
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        features=features,
        aux_logits=aux_logits,
        diagnostics=diagnostics,
    )


class DualStreamConcatAuxTimepix(nn.Module):
    """Two ResNet18 encoders with high-level feature concat and auxiliary heads."""

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
        self.tot_encoder = ResNet18Backbone(1, feature_dim, kernel_size, stride, padding, pretrained)
        self.toa_encoder = ResNet18Backbone(1, feature_dim, kernel_size, stride, padding, pretrained)
        self.fusion = FeatureFusion(feature_dim * 2, handcrafted_dim, fusion_mode)
        out_dim = 1 if task == "regression" else num_classes
        self.head = _make_head(self.fusion.out_dim, hidden_dim, out_dim, dropout)
        self.tot_aux_head = nn.Linear(feature_dim, out_dim)
        self.toa_aux_head = nn.Linear(feature_dim, out_dim)

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> ModelOutput:
        x_tot, x_toa = _split_tot_toa(image)
        f_tot = self.tot_encoder(x_tot)
        f_toa = self.toa_encoder(x_toa)
        features = torch.cat([f_tot, f_toa], dim=1)
        fused = self.fusion(features, handcrafted)
        output = self.head(fused)
        aux_logits = {
            "tot": self.tot_aux_head(f_tot),
            "toa": self.toa_aux_head(f_toa),
        }
        if self.task == "regression":
            return ModelOutput(regression=torch.sigmoid(output).squeeze(-1), features=fused)
        return _classification_output(output, features=fused, aux_logits=aux_logits)


class DualStreamGMUAuxTimepix(nn.Module):
    """Two ResNet18 encoders fused by a GMU-style gate biased toward ToT."""

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
        gate_init_bias_to_tot: float = 2.0,
    ) -> None:
        super().__init__()
        self.task = task
        self.tot_encoder = ResNet18Backbone(1, feature_dim, kernel_size, stride, padding, pretrained)
        self.toa_encoder = ResNet18Backbone(1, feature_dim, kernel_size, stride, padding, pretrained)
        self.tot_proj = nn.Linear(feature_dim, feature_dim)
        self.toa_proj = nn.Linear(feature_dim, feature_dim)
        self.gate = nn.Linear(feature_dim * 2, feature_dim)
        nn.init.constant_(self.gate.bias, float(gate_init_bias_to_tot))
        self.fusion = FeatureFusion(feature_dim, handcrafted_dim, fusion_mode)
        out_dim = 1 if task == "regression" else num_classes
        self.head = _make_head(self.fusion.out_dim, hidden_dim, out_dim, dropout)
        self.tot_aux_head = nn.Linear(feature_dim, out_dim)
        self.toa_aux_head = nn.Linear(feature_dim, out_dim)

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> ModelOutput:
        x_tot, x_toa = _split_tot_toa(image)
        f_tot = self.tot_encoder(x_tot)
        f_toa = self.toa_encoder(x_toa)
        h_tot = torch.tanh(self.tot_proj(f_tot))
        h_toa = torch.tanh(self.toa_proj(f_toa))
        gate_tot = torch.sigmoid(self.gate(torch.cat([f_tot, f_toa], dim=1)))
        features = gate_tot * h_tot + (1.0 - gate_tot) * h_toa
        fused = self.fusion(features, handcrafted)
        output = self.head(fused)
        aux_logits = {
            "tot": self.tot_aux_head(f_tot),
            "toa": self.toa_aux_head(f_toa),
        }
        diagnostics = {
            "gate_tot": gate_tot.mean(dim=1),
            "gate_toa": (1.0 - gate_tot).mean(dim=1),
        }
        if self.task == "regression":
            return ModelOutput(regression=torch.sigmoid(output).squeeze(-1), features=fused, diagnostics=diagnostics)
        return _classification_output(output, features=fused, aux_logits=aux_logits, diagnostics=diagnostics)


class ToAConditionedFiLMTimepix(nn.Module):
    """ToT ResNet18 branch modulated by relative-ToA FiLM parameters after layer3."""

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
        film_hidden_dim: int = 256,
        film_zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.task = task
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.tot_model = resnet18(weights=weights)
        self.tot_model.conv1 = nn.Conv2d(1, 64, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.tot_model.maxpool = nn.Identity()
        self.tot_model.fc = nn.Linear(self.tot_model.fc.in_features, feature_dim)
        self.toa_encoder = ResNet18Backbone(1, feature_dim, kernel_size, stride, padding, pretrained)
        self.film_channels = 256
        self.film = nn.Sequential(
            nn.Linear(feature_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, self.film_channels * 2),
        )
        if film_zero_init:
            last = self.film[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.fusion = FeatureFusion(feature_dim, handcrafted_dim, fusion_mode)
        out_dim = 1 if task == "regression" else num_classes
        self.head = _make_head(self.fusion.out_dim, hidden_dim, out_dim, dropout)

    def _tot_features(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        model = self.tot_model
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = (1.0 + gamma[:, :, None, None]) * x + beta[:, :, None, None]
        x = model.layer4(x)
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        return model.fc(x)

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> ModelOutput:
        x_tot, x_toa = _split_tot_toa(image)
        toa_features = self.toa_encoder(x_toa)
        gamma, beta = self.film(toa_features).chunk(2, dim=1)
        tot_features = self._tot_features(x_tot, gamma, beta)
        fused = self.fusion(tot_features, handcrafted)
        output = self.head(fused)
        diagnostics = {
            "film_gamma_abs": gamma.abs().mean(dim=1),
            "film_beta_abs": beta.abs().mean(dim=1),
        }
        if self.task == "regression":
            return ModelOutput(regression=torch.sigmoid(output).squeeze(-1), features=fused, diagnostics=diagnostics)
        return _classification_output(output, features=fused, diagnostics=diagnostics)

