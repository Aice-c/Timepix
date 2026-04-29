"""Dual-stream Alpha ToT/ToA fusion models for A4c."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .base import ModelOutput
from .fusion import FeatureFusion
from .resnet import ResNet18Backbone, ResNet18Timepix


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


class WarmStartedExpertGateTimepix(nn.Module):
    """Gate warm-started ToT and ToT+relative-ToA experts for A4c-4."""

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
        gate_hidden_dim: int = 256,
        gate_dropout: float = 0.1,
        gate_init_bias_to_candidate: float = -2.0,
        gate_include_logits: bool = True,
    ) -> None:
        super().__init__()
        if task != "classification":
            raise ValueError("warm_started_expert_gate currently supports classification only")
        if input_channels < 2:
            raise ValueError("warm_started_expert_gate requires ToT+ToA input channels")

        self.task = task
        self.gate_include_logits = gate_include_logits
        self._experts_frozen = False
        self.primary = ResNet18Timepix(
            1,
            num_classes,
            task=task,
            handcrafted_dim=handcrafted_dim,
            fusion_mode=fusion_mode,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pretrained=pretrained,
        )
        self.candidate = ResNet18Timepix(
            input_channels,
            num_classes,
            task=task,
            handcrafted_dim=handcrafted_dim,
            fusion_mode=fusion_mode,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pretrained=pretrained,
        )

        gate_in_dim = feature_dim * 2 + (num_classes * 2 if gate_include_logits else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, 1),
        )
        last = self.gate[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, float(gate_init_bias_to_candidate))

    def load_expert_states(
        self,
        primary_state: dict[str, torch.Tensor],
        candidate_state: dict[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> dict[str, object]:
        primary_result = self.primary.load_state_dict(primary_state, strict=strict)
        candidate_result = self.candidate.load_state_dict(candidate_state, strict=strict)
        return {
            "primary_missing_keys": list(primary_result.missing_keys),
            "primary_unexpected_keys": list(primary_result.unexpected_keys),
            "candidate_missing_keys": list(candidate_result.missing_keys),
            "candidate_unexpected_keys": list(candidate_result.unexpected_keys),
        }

    def set_experts_trainable(self, trainable: bool) -> None:
        self._experts_frozen = not trainable
        for module in (self.primary, self.candidate):
            for param in module.parameters():
                param.requires_grad = trainable
        for param in self.gate.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._experts_frozen:
            self.primary.eval()
            self.candidate.eval()
        return self

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> ModelOutput:
        x_tot, _ = _split_tot_toa(image)
        if self._experts_frozen and self.training:
            with torch.no_grad():
                primary_output = self.primary(x_tot, handcrafted)
                candidate_output = self.candidate(image, handcrafted)
        else:
            primary_output = self.primary(x_tot, handcrafted)
            candidate_output = self.candidate(image, handcrafted)

        gate_inputs = [primary_output.features, candidate_output.features]
        if self.gate_include_logits:
            gate_inputs.extend([primary_output.logits, candidate_output.logits])
        gate_candidate = torch.sigmoid(self.gate(torch.cat(gate_inputs, dim=1)))
        logits = (1.0 - gate_candidate) * primary_output.logits + gate_candidate * candidate_output.logits
        features = (1.0 - gate_candidate) * primary_output.features + gate_candidate * candidate_output.features
        aux_logits = {
            "primary": primary_output.logits,
            "candidate": candidate_output.logits,
        }
        diagnostics = {
            "gate_primary": (1.0 - gate_candidate).squeeze(1),
            "gate_candidate": gate_candidate.squeeze(1),
            "gate_tot": (1.0 - gate_candidate).squeeze(1),
            "gate_toa": gate_candidate.squeeze(1),
        }
        return _classification_output(logits, features=features, aux_logits=aux_logits, diagnostics=diagnostics)

