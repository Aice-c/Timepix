"""Small CPU checks; no detector-data training."""
import copy
import importlib
import importlib.util
import json
from types import SimpleNamespace

import numpy as np
import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def module():
    assert importlib.util.find_spec('timepix.training.stability') is not None, 'stability support missing'
    return importlib.import_module('timepix.training.stability')


def test_gradient_observer_preserves_update_and_rng(tmp_path):
    s = module()
    torch.manual_seed(7)
    original = nn.Linear(3, 2)
    observed = copy.deepcopy(original)
    a = torch.optim.Adam(original.parameters(), lr=1e-4)
    b = torch.optim.Adam(observed.parameters(), lr=1e-4)
    x = torch.ones(4, 3)
    original(x).square().mean().backward()
    observed(x).square().mean().backward()
    before = [p.grad.clone() for p in observed.parameters()]
    rng = torch.get_rng_state().clone()
    with s.StepAudit(b, tmp_path / 'steps.jsonl', 1) as audit:
        audit.before_step(observed, None)
        b.step()
        audit.after_step()
        audit.before_step(observed, None)
        audit.after_step()  # Emulate a skipped optimizer call, not a successful step.
    a.step()
    assert torch.equal(rng, torch.get_rng_state())
    for p, q, grad in zip(original.parameters(), observed.parameters(), before):
        assert torch.equal(p, q)
        assert torch.equal(q.grad, grad)
    rows = [json.loads(line) for line in (tmp_path / 'steps.jsonl').read_text().splitlines()]
    assert [r['optimizer_step_executed'] for r in rows] == [True, False]
    assert rows[0]['gradient_l2_unscaled'] > 0
    assert rows[0]['gradient_finite']


class TinyBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(.8)
        self.fc = nn.Linear(4, 2)

    def forward(self, image):
        return self.fc(self.dropout(self.bn(image)).flatten(1))


def test_bn_reestimate_changes_only_bn_buffers_on_copy():
    s = module()
    torch.manual_seed(4)
    model = TinyBN().eval()
    source = copy.deepcopy(model.state_dict())
    clone = copy.deepcopy(model)
    loader = DataLoader(TensorDataset(torch.arange(32.).reshape(8, 1, 2, 2), torch.zeros(8)), batch_size=4)
    rng = torch.get_rng_state().clone()
    result = s.reestimate_batchnorm(clone, loader, torch.device('cpu'))
    assert result['batches'] == 2 and result['examples'] == 8
    assert result['non_bn_state_unchanged']
    assert not clone.training and not clone.dropout.training and not clone.bn.training
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(clone.bn.running_mean, model.bn.running_mean)
    assert clone.bn.momentum == model.bn.momentum
    for key, value in source.items():
        assert torch.equal(model.state_dict()[key], value)


def test_stratified_subset_is_repeatable_and_only_from_given_records():
    s = module()
    records = [SimpleNamespace(label=i // 8, key=str(i)) for i in range(24)]
    a = s.stratified_indices(records, 3, 20260916)
    assert a == s.stratified_indices(records, 3, 20260916)
    assert len(a) == len(set(a)) == 9
    assert np.bincount([records[i].label for i in a]).tolist() == [3, 3, 3]


def test_manifest_export_uses_angle_qualified_mapping():
    from pathlib import Path
    from scripts.diagnose_carbon_stability import manifest_rows
    records = [SimpleNamespace(key=f'{a}/same_001.txt', angle=a, modalities={'ToT': Path(f'{a}/ToT/same_001.txt')})
               for a in ['10', '20']]
    mapping = {r.key: dict(frame_group_key=f'{r.angle}/same.txt', raw_frame_member=f'C1/{r.angle}/same.txt')
               for r in records}
    rows = manifest_rows(SimpleNamespace(records=records), [0, 1], 'train', 'probe', mapping)
    assert rows[0]['frame_group_key'] != rows[1]['frame_group_key']
    assert rows[1]['raw_frame_member'] == 'C1/20/same.txt'


def test_new_config_acc_first_and_only_one_training_cell():
    from timepix.config import load_experiment_config, PROJECT_ROOT
    path = PROJECT_ROOT / 'configs/experiments/carbon_t7_stability_lr1e4_seed42.yaml'
    assert path.exists(), 'single approved stability config missing'
    cfg = load_experiment_config(path)
    from timepix.config_validation import validate_experiment_config
    validate_experiment_config(cfg)
    assert cfg['task']['primary_metric'] == 'val_accuracy'
    assert cfg['task']['tie_break_metrics'] == ['val_mae_argmax', 'val_macro_f1']
    assert cfg['training']['learning_rate'] == 1e-4
    assert cfg['training']['seed'] == 42
    assert cfg['training']['epochs'] == 25
    assert cfg['training']['early_stopping_patience'] == 8
    assert cfg['training']['stability_diagnostics']['enabled']
    assert cfg['data']['normalizer_metadata'].endswith('20260909_014839_carbon_t7_tot_seed42/metadata.json')
    assert cfg['evaluation']['run_test'] is False
    from timepix.training.selection import selection_key
    high_acc = dict(accuracy=.92, mae_argmax=.8, macro_f1=.90)
    low_mae = dict(accuracy=.91, mae_argmax=.6, macro_f1=.91)
    assert selection_key(high_acc, 'classification', cfg['task']) > selection_key(low_mae, 'classification', cfg['task'])


def test_historical_sources_are_rehashed_not_assumed(tmp_path):
    from scripts.summarize_carbon_stability import verify_historical_sources
    from scripts.diagnose_carbon_stability import sha256
    names = ['best_model.pth', 'last_checkpoint.pth', 'metadata.json', 'validation_predictions.csv']
    for name in names:
        (tmp_path / name).write_bytes(b'original')
    checks = {f'/root/old/{name}': sha256(tmp_path / name) for name in names}
    assert verify_historical_sources(checks, tmp_path)
    (tmp_path / 'metadata.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Historical source changed'):
        verify_historical_sources(checks, tmp_path)


@pytest.mark.parametrize('value', [True, {'enabled': 'yes'}, {'enabled': True, 'typo': 1}])
def test_diagnostic_switch_rejects_invalid_types(value):
    from timepix.config import load_experiment_config
    from timepix.config_validation import validate_experiment_config
    from scripts.diagnose_carbon_stability import CONFIG
    cfg = load_experiment_config(CONFIG)
    cfg['training']['stability_diagnostics'] = value
    with pytest.raises(ValueError):
        validate_experiment_config(cfg)


def test_diagnostic_directory_claim_is_exclusive(tmp_path):
    from scripts.diagnose_carbon_stability import claim_output
    target = tmp_path / 'new/diagnostic'
    claim_output(target)
    with pytest.raises(FileExistsError):
        claim_output(target)


def test_run_identity_checks_metadata_and_saved_config():
    from timepix.config import load_experiment_config
    from scripts.diagnose_carbon_stability import CONFIG
    from scripts.summarize_carbon_stability import verify_run_identity
    cfg = load_experiment_config(CONFIG)
    meta = {k: copy.deepcopy(cfg[k]) for k in ['dataset', 'data', 'model', 'loss', 'training']}
    meta.update(primary_metric='val_accuracy', experiment_name=cfg['experiment_name'],
                experiment_group=cfg['experiment_group'])
    assert verify_run_identity(cfg, meta, cfg)
    for section, key, value in [('training', 'seed', 99), ('training', 'learning_rate', .003),
                                ('model', 'name', 'resnet18_no_maxpool_apdc_layer1')]:
        changed = copy.deepcopy(meta)
        changed[section][key] = value
        with pytest.raises(ValueError, match='identity'):
            verify_run_identity(cfg, changed, cfg)
    changed_cfg = copy.deepcopy(cfg)
    changed_cfg['task']['primary_metric'] = 'val_mae_argmax'
    with pytest.raises(ValueError, match='identity'):
        verify_run_identity(changed_cfg, meta, cfg)


def test_step_inventory_rejects_missing_duplicate_or_unexpected_batches():
    from scripts.summarize_carbon_stability import verify_step_inventory
    assert verify_step_inventory({1: [1, 2], 2: [1, 2]}, [1, 2], 2, 4)
    for observed in [{1: [1], 2: [1, 2]}, {1: [1, 1, 2], 2: [1, 2]},
                     {1: [1, 2], 2: [1, 2], 3: [1]}]:
        with pytest.raises(ValueError, match='batch'):
            verify_step_inventory(observed, [1, 2], 2, 4)
