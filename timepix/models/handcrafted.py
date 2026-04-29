"""Handcrafted-only models."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelOutput


class HandcraftedMLPTimepix(nn.Module):
    def __init__(
        self,
        num_classes: int,
        task: str = "classification",
        handcrafted_dim: int = 0,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if handcrafted_dim <= 0:
            raise ValueError("handcrafted_mlp requires handcrafted features")
        self.task = task
        out_dim = 1 if task == "regression" else num_classes
        self.head = nn.Sequential(
            nn.Linear(handcrafted_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, image: torch.Tensor, handcrafted: torch.Tensor | None = None) -> ModelOutput:
        if handcrafted is None:
            raise ValueError("handcrafted_mlp requires handcrafted features")
        output = self.head(handcrafted)
        if self.task == "regression":
            regression = torch.sigmoid(output).squeeze(-1)
            return ModelOutput(regression=regression, features=handcrafted)
        probabilities = F.softmax(output, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return ModelOutput(logits=output, probabilities=probabilities, predictions=predictions, features=handcrafted)
