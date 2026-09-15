"""Differentiable 3x3 CDC / mixed APDC kernel reparameterizations.

APDC follows PiDiNet models/ops_theta.py (ad), using the inverse ring
permutation on weights. Only the raw weight is a learned parameter.
"""
import math

from torch import nn
import torch.nn.functional as F


APDC_PI = (1, 2, 5, 0, 4, 8, 3, 6, 7)
APDC_PREV = (3, 0, 1, 6, 4, 2, 7, 8, 5)
LAYER1_PATHS = tuple(f'backbone.model.layer1.{block}.conv{conv}'
                     for block in (0, 1) for conv in (1, 2))


class DifferenceConv2d(nn.Module):
    def __init__(self, original: nn.Conv2d, kind: str, theta: float):
        super().__init__()
        if (original.kernel_size != (3, 3) or original.stride != (1, 1)
                or original.padding != (1, 1) or original.dilation != (1, 1)
                or original.padding_mode != 'zeros'):
            raise ValueError('Difference controls require 3x3, stride1, padding1, dilation1, zero padding')
        if kind not in {'cdc', 'apdc'} or not math.isfinite(theta) or not 0 <= theta <= 1:
            raise ValueError('Expected cdc/apdc and finite theta in [0,1]')
        self.kind, self.theta = kind, float(theta)
        for name in ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups'):
            setattr(self, name, getattr(original, name))
        # Reuse Parameter objects after the whole base network has initialized.
        self.weight = original.weight
        self.bias = original.bias

    def effective_weight(self):
        flat = self.weight.flatten(2)
        if self.kind == 'cdc':
            effective = flat.clone()
            effective[..., 4] = flat[..., 4] - self.theta * flat.sum(dim=-1)
        else:
            effective = flat - self.theta * flat[..., list(APDC_PREV)]
        return effective.reshape_as(self.weight)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return f'kind={self.kind}, theta={self.theta}, channels={self.in_channels}->{self.out_channels}'


def replace_layer1(model, kind: str, theta: float):
    for path in LAYER1_PATHS:
        parent_path, child = path.rsplit('.', 1)
        parent = model.get_submodule(parent_path)
        original = getattr(parent, child)
        if not isinstance(original, nn.Conv2d):
            raise ValueError(f'Expected ordinary convolution at {path}; refusing double conversion')
        setattr(parent, child, DifferenceConv2d(original, kind, theta))
    return model
