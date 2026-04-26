"""Shared model output types."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    logits: torch.Tensor | None = None
    probabilities: torch.Tensor | None = None
    predictions: torch.Tensor | None = None
    regression: torch.Tensor | None = None
    features: torch.Tensor | None = None

