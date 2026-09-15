"""Queue/summaries tested using metadata only; no model training."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from timepix.config import load_experiment_config
from timepix.config_validation import validate_experiment_config


def get_module(name):
    assert importlib.util.find_spec(name) is not None, f'{name} missing'
    return importlib.import_module(name)


def test_six_configs_and_completion_rules():
    module = get_module('scripts.run_carbon_difference_queue')
    assert [(x['method'], x['seed']) for x in module.plan()] == [
        ('A', 42), ('B', 42), ('A', 43), ('B', 43), ('A', 44), ('B', 44)]
    for item in module.plan():
        cfg = load_experiment_config(item['config'])
        validate_experiment_config(cfg)
        assert cfg['training']['seed'] == item['seed'] and cfg['split']['seed'] == 42
        assert cfg['model']['difference_theta'] == .7
        assert not cfg['evaluation']['run_test']
        assert cfg['data']['normalizer_metadata'].endswith('metadata.json')
    assert module.normal_completion(dict(stopped_epoch=25, max_epochs=25, early_stopped=False)) == 'completed'
    assert module.normal_completion(dict(stopped_epoch=11, max_epochs=25, early_stopped=True)) == 'early_stopped'
    assert module.normal_completion(dict(stopped_epoch=6, max_epochs=25, early_stopped=False)) is None


def test_aggregate_and_pairing_not_pooled_or_fake_baseline_seeds():
    module = get_module('scripts.summarize_carbon_difference')
    rows = [dict(method=m, seed=s, mae_argmax=v) for m, s, v in [
        ('R', 42, 1.1), ('A', 42, 1.0), ('A', 43, 2.0), ('A', 44, 3.0),
        ('B', 42, .9), ('B', 43, 1.8), ('B', 44, 2.7)]]
    aggregate, pairs = module.aggregate_rows(rows, ['mae_argmax'])
    r = next(x for x in aggregate if x['method'] == 'R')
    assert r['n'] == 1 and r['std'] is None
    a = next(x for x in aggregate if x['method'] == 'A')
    assert a['mean'] == 2 and a['std'] == 1
    pair = next(x for x in pairs if x['comparison_kind'] == 'paired_B_minus_A_summary')
    assert pair['n_pairs'] == 3 and pair['improved_seeds'] == 3
    assert pair['difference'] == pytest.approx(-.2)
    assert pair['std'] == pytest.approx(.1)
    assert len([x for x in pairs if x['comparison_kind'] == 'paired_B_minus_A_seed']) == 3


def test_external_signal_is_interrupted_and_child_inherits_queue_lock(monkeypatch):
    module = get_module('scripts.run_carbon_difference_queue')
    assert hasattr(module, 'exit_status'), 'Signal exit classification missing'
    assert module.exit_status(-15) == 'interrupted'
    assert module.exit_status(1) == 'failed'
    captured = {}
    def run(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    module.run_child(['python', 'train.py'], None, 97)
    assert captured['pass_fds'] == (97,) and 'timeout' not in captured


def test_truncated_metadata_is_not_completed(tmp_path):
    module = get_module('scripts.run_carbon_difference_queue')
    for name in ['metadata.json', 'metrics.json', 'validation_predictions.csv',
                 'validation_metrics.json', 'training_log.csv', 'best_model.pth', 'last_checkpoint.pth']:
        (tmp_path / name).write_text('{')
    (tmp_path / 'metrics.json').write_text(json.dumps(dict(
        stopped_epoch=25, max_epochs=25, early_stopped=False, test_evaluated=False)))
    assert module.read_completed(tmp_path) is None


def test_bad_run_identification_is_recorded_without_raising(monkeypatch):
    module = get_module('scripts.run_carbon_difference_queue')
    assert hasattr(module, 'inspect_item'), 'Per-run identification failure handler missing'
    def fail(cfg):
        raise ValueError('partial config missing')
    monkeypatch.setattr(module, 'matching_runs', fail)
    item = {'id': 'A42', 'status': 'pending'}
    path, completed = module.inspect_item({}, item)
    assert path is None and completed is None and item['status'] == 'failed'
    assert 'partial config' in item['stop_reason']


@pytest.mark.parametrize('content', ['model: [', ''])
def test_real_truncated_config_does_not_stop_other_cells(tmp_path, monkeypatch, content):
    module = get_module('scripts.run_carbon_difference_queue')
    monkeypatch.setattr(module, 'RUN_ROOT', tmp_path)
    run = tmp_path / 'timestamp_example'
    run.mkdir()
    (run / 'config.yaml').write_text(content)
    item = dict(id='A42', status='pending')
    module.inspect_item(dict(experiment_name='example'), item)
    assert item['status'] == 'failed'
