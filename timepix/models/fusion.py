"""Feature fusion modules."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Fuse CNN features with optional handcrafted features."""

    def __init__(self, cnn_dim: int, handcrafted_dim: int, mode: str = "none", reduction: int = 8) -> None:
        super().__init__()
        self.cnn_dim = cnn_dim
        self.handcrafted_dim = handcrafted_dim
        self.mode = mode

        if mode not in {"none", "concat", "gated"}:
            raise ValueError(f"Unknown fusion mode: {mode}")
        if handcrafted_dim == 0 and mode != "none":
            raise ValueError(f"fusion_mode={mode} requires handcrafted features")
        if handcrafted_dim > 0 and mode == "none":
            raise ValueError("handcrafted features are enabled but fusion_mode is 'none'")

        self.out_dim = cnn_dim if mode == "none" else cnn_dim + handcrafted_dim
        if mode == "gated":
            hidden = max(1, self.out_dim // reduction)
            self.gate = nn.Sequential(
                nn.Linear(self.out_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.out_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def forward(self, cnn_features: torch.Tensor, handcrafted: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "none":
            return cnn_features
        if handcrafted is None:
            raise ValueError(f"fusion_mode={self.mode} requires handcrafted features")
        if handcrafted.dim() == 1:
            handcrafted = handcrafted.unsqueeze(0)
        features = torch.cat([cnn_features, handcrafted], dim=1)
        if self.gate is not None:
            features = features * self.gate(features)
        return features

