"""Targeted tests for the two approved carbon operators, not legacy regression."""
import importlib
import importlib.util

import pytest
import torch
import torch.nn.functional as F

from timepix.config import load_experiment_config
from timepix.models import build_model


def implementation():
    assert importlib.util.find_spec('timepix.models.difference') is not None, 'Difference operators not implemented'
    return importlib.import_module('timepix.models.difference')


def explicit_formula(x, conv, kind, theta):
    # Independent patch-space formula, with explicit channel groups and bias once.
    patches = F.unfold(x, 3, padding=1).reshape(x.shape[0], conv.groups, -1, 9, x.shape[-2] * x.shape[-1])
    weights = conv.weight.reshape(conv.groups, conv.out_channels // conv.groups, -1, 9)
    if kind == 'cdc':
        values = patches - theta * patches[:, :, :, 4:5, :]
    else:
        pi = [1, 2, 5, 0, 4, 8, 3, 6, 7]
        values = (1 - theta) * patches + theta * (patches - patches[:, :, :, pi, :])
    y = torch.einsum('goiq,bgiql->bgol', weights, values).reshape(x.shape[0], conv.out_channels, *x.shape[-2:])
    return y + conv.bias[None, :, None, None]


@pytest.mark.parametrize('kind', ['cdc', 'apdc'])
@pytest.mark.parametrize('theta', [0., .7])
def test_formula_and_gradients(kind, theta):
    module = implementation()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    conv = torch.nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=True).double()
    state = torch.get_rng_state().clone()
    op = module.DifferenceConv2d(conv, kind, theta)
    assert op.weight is conv.weight and op.bias is conv.bias
    assert torch.equal(state, torch.get_rng_state())
    assert list(op.state_dict()) == ['weight', 'bias']
    x = torch.randn(2, 4, 7, 8, dtype=torch.float64, requires_grad=True)
    y = op(x)
    expected = explicit_formula(x, conv, kind, theta)
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-12)
    if theta == 0:
        torch.testing.assert_close(y, conv(x), atol=1e-12, rtol=1e-12)
    params = (x, conv.weight, conv.bias)
    actual_grad = torch.autograd.grad(y.square().mean(), params, retain_graph=True)
    reference_grad = torch.autograd.grad(expected.square().mean(), params)
    for actual, reference in zip(actual_grad, reference_grad):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_apdc_inverse_permutation():
    module = implementation()
    assert [module.APDC_PI[i] for i in module.APDC_PREV] == list(range(9))


def test_two_models_same_raw_init_shapes_and_backward():
    module = implementation()
    torch.set_num_threads(1)
    configs = []
    for kind in ['cdc', 'apdc']:
        cfg = load_experiment_config('configs/experiments/carbon_t7_tot_seed42.yaml')
        cfg['model']['name'] = f'resnet18_no_maxpool_{kind}_layer1'
        cfg['model']['difference_theta'] = .7
        configs.append(cfg)
    state0 = None
    rng0 = None
    for cfg in configs:
        torch.manual_seed(42)
        model = build_model(cfg, 1, 7, 'classification', 0)
        if state0 is None:
            state0 = {k: v.clone() for k, v in model.state_dict().items()}
            rng0 = torch.get_rng_state().clone()
        else:
            assert all(torch.equal(state0[k], v) for k, v in model.state_dict().items())
            assert torch.equal(rng0, torch.get_rng_state())
        locations = [k for k, v in model.named_modules() if isinstance(v, module.DifferenceConv2d)]
        assert locations == list(module.LAYER1_PATHS)
        assert sum(p.numel() for p in model.parameters()) == 11433863
        shapes = {}
        backbone = model.backbone.model
        names = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']
        handles = [getattr(backbone, n).register_forward_hook(
            lambda m, args, out, n=n: shapes.update({n: list(out.shape)})) for n in names]
        x = torch.randn(2, 1, 50, 50, requires_grad=True)
        logits = model(x).logits
        assert list(logits.shape) == [2, 7]
        assert [shapes[n][-1] for n in names] == [49, 49, 25, 13, 7, 1]
        F.cross_entropy(logits, torch.tensor([0, 6])).backward()
        assert torch.isfinite(logits).all() and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        for handle in handles:
            handle.remove()
