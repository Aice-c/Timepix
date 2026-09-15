"""Narrow regression checks for new frozen-statistics and resume behavior."""
import hashlib
import importlib
import importlib.util
import json
import random

import numpy as np
import pytest
import torch

from timepix.data import normalization


def test_frozen_normalizer_reads_metadata_and_preserves_background(tmp_path):
    assert hasattr(normalization, 'load_frozen_normalizer'), 'Frozen normalizer loader missing'
    split = tmp_path / 'split.json'
    split.write_text('{}')
    stats = dict(mean=637.2197424470774, std=1189.814612768605, min=1., max=20000., log1p=False, ignore_zero=True)
    meta = dict(split_manifest_hash=hashlib.sha256(split.read_bytes()).hexdigest(),
                dataset=dict(name='Carbon_T7', class_names=['10', '20']),
                data=dict(crop_size=0, dtype='float32', input_representation='signal'),
                data_info=dict(normalizer_stats={'ToT': stats}, modalities=['ToT'], class_names=['10', '20']))
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(meta))
    config = {'ToT': dict(enabled=True, log1p=False, ignore_zero=True)}
    normalizer, provenance = normalization.load_frozen_normalizer(
        path, ['ToT'], config, split, meta['dataset'], meta['data'])
    assert normalizer.stats['ToT'].mean == stats['mean']
    expected = (torch.tensor([0., 600.]) - stats['mean']) / stats['std']
    torch.testing.assert_close(normalizer.apply(torch.tensor([0., 600.]), 'ToT'), expected)
    assert provenance['refitted'] is False
    split.write_text('{"different": true}')
    with pytest.raises(ValueError, match='split'):
        normalization.load_frozen_normalizer(path, ['ToT'], config, split, meta['dataset'], meta['data'])


def test_rng_roundtrip_is_weights_only_loadable(tmp_path):
    assert importlib.util.find_spec('timepix.training.random_state') is not None, 'Resume RNG state missing'
    module = importlib.import_module('timepix.training.random_state')
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    state = module.capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(5))
    path = tmp_path / 'state.pth'
    torch.save(state, path)
    module.restore_rng_state(torch.load(path, weights_only=True))
    actual = (random.random(), np.random.rand(), torch.rand(5))
    assert actual[0:2] == expected[0:2]
    assert torch.equal(actual[2], expected[2])
