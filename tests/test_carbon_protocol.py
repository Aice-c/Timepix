"""Carbon protocol regressions; CPU only, no real optimizer steps."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from timepix.data.dataset import SampleRecord, TimepixDataset
from timepix.models import build_model
from timepix.training import runner


torch.set_num_threads(1)


def test_frame_keys_include_angle_and_source_and_strip_only_component():
    from timepix.data.frame_groups import event_frame_stem, frame_key
    assert event_frame_stem('C_r00000_0085_007.txt') == 'C_r00000_0085'
    assert frame_key('80', 'C1/80/C_r00000_0085.txt') != frame_key('82', 'C1/82/C_r00000_0085.txt')
    assert frame_key('80', 'C1/80/a.txt') != frame_key('80', 'C2/80/a.txt')
    with pytest.raises(ValueError):
        event_frame_stem('unrecognized.txt')


def test_group_split_is_deterministic_complete_and_never_splits_frame():
    from timepix.data.frame_groups import grouped_manifest, validate_group_manifest, frame_key
    rows = [dict(sample_key=f'{a}/frame{f}_{e}.txt', angle=str(a),
                 frame_group_key=frame_key(str(a), f'C1/{a}/frame{f}.txt'),
                 raw_frame_member=f'C1/{a}/frame{f}.txt')
            for a in [80, 82] for f in range(15) for e in range(1 + f % 4)]
    p = grouped_manifest(rows, seed=42)
    assert p == grouped_manifest(list(reversed(rows)), seed=42)
    validate_group_manifest(p, [r['sample_key'] for r in rows])
    broken = json.loads(json.dumps(p))
    broken['val'].append(broken['train'][0])
    with pytest.raises(ValueError):
        validate_group_manifest(broken, [r['sample_key'] for r in rows])
    from timepix.data.frame_groups import manifest_digest
    broken = json.loads(json.dumps(p))
    train_key = next(k for k in broken['train'] if sum(
        broken['samples'][q]['frame_group_key'] == broken['samples'][k]['frame_group_key'] for q in broken['train']) > 1)
    broken['train'].remove(train_key)
    broken['val'].append(train_key)
    broken['manifest_sha256'] = manifest_digest(broken)
    with pytest.raises(ValueError, match='overlaps'):
        validate_group_manifest(broken, [r['sample_key'] for r in rows])
    broken = json.loads(json.dumps(p))
    broken['test'].pop()
    broken['manifest_sha256'] = manifest_digest(broken)
    with pytest.raises(ValueError, match='cover'):
        validate_group_manifest(broken, [r['sample_key'] for r in rows])


def test_mask_bypasses_normalizer_and_preserves_geometric_views(tmp_path):
    path = tmp_path / 'a.txt'
    a = np.zeros((50, 50)); a[24, 23] = 17; a[25, 23] = 2
    np.savetxt(path, a)
    class ForbiddenNormalizer:
        def apply(self, *args):
            raise AssertionError('Mask must not use intensity normalizer')
    ds = TimepixDataset([SampleRecord(0, '80', '80/a.txt', {'ToT': path})],
                        {0: '80'}, ['ToT'], training=True, rotation_enabled=True,
                        normalizer=ForbiddenNormalizer(), input_representation='hit_mask')
    assert len(ds) == 4
    for i in range(4):
        image, _ = ds[i]
        assert image.shape == (1, 50, 50)
        assert set(image.unique().tolist()) == {0., 1.}
        assert torch.equal(image[0], torch.rot90(torch.tensor(a > 0), i, (0, 1)))


def test_hires_shapes_and_parameter_count():
    counts = []
    for hires, expected in [(False, [49, 49, 25, 13, 7, 1]), (True, [49, 49, 25, 25, 25, 1])]:
        cfg = {'model': {'name': 'resnet18_no_maxpool', 'preserve_late_resolution': hires}}
        model = build_model(cfg, 1, 6, 'classification', 0).eval()
        net = model.backbone.model
        shapes = []
        handles = [getattr(net, k).register_forward_hook(lambda m, x, y: shapes.append(y.shape[-1]))
                   for k in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']]
        with torch.inference_mode():
            assert model(torch.zeros(1, 1, 50, 50)).logits.shape == (1, 6)
        for h in handles:
            h.remove()
        assert shapes == expected
        counts.append(sum(p.numel() for p in model.parameters()))
    assert counts[0] == counts[1]


def test_selection_ties_and_missing_metric():
    from timepix.training.selection import selection_key
    cfg = {'primary_metric': 'val_mae_argmax', 'tie_break_metrics': ['val_macro_f1']}
    a = selection_key({'mae_argmax': 1., 'macro_f1': .7}, 'classification', cfg)
    b = selection_key({'mae_argmax': 1., 'macro_f1': .6}, 'classification', cfg)
    assert a > b
    assert not a > a
    with pytest.raises(ValueError):
        selection_key({'accuracy': .9}, 'classification', cfg)


def test_validation_only_runner_never_reads_test(tmp_path, monkeypatch):
    from torch.utils.data import DataLoader, TensorDataset
    from timepix.models.base import ModelOutput
    payload = dict(loss=.1, labels=np.array([0, 1]), logits=np.array([[3., 0.], [0., 3.]]))
    val = DataLoader(TensorDataset(torch.zeros(2, 1, 50, 50), torch.tensor([0, 1])), batch_size=2)
    sentinel = object()
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.zeros(1))
    def fake_evaluate(model, loader, *args, **kwargs):
        assert loader is not sentinel, 'test evaluation forbidden'
        return payload
    info = dict(label_map={0: '80', 1: '82'}, num_classes=2, label_type='angle_folder',
                class_names=['80', '82'], angle_values=[80., 82.], modalities=['ToT'], handcrafted_dim=0,
                split_path=str(tmp_path / 'absent_test_fixture.json'))
    monkeypatch.setattr(runner, 'build_dataloaders', lambda *a, **k: ({'train': val, 'val': val, 'test': sentinel}, info))
    monkeypatch.setattr(runner, 'build_model', lambda *a, **k: Tiny())
    monkeypatch.setattr(runner, 'train_one_epoch', lambda *a, **k: payload)
    monkeypatch.setattr(runner, 'evaluate', fake_evaluate)
    cfg = dict(experiment_name='synthetic_control_flow_only', task={'type': 'classification'},
               model={'name': 'resnet18_no_maxpool'}, training={'epochs': 1, 'save_last_checkpoint': False},
               evaluation={'run_test': False}, loss={'name': 'cross_entropy'})
    result = runner.run_experiment(cfg, output_root=tmp_path)
    assert result['metrics']['test'] == {}
    assert result['metrics']['test_evaluated'] is False
    assert not (Path(result['experiment_dir']) / 'predictions.csv').exists()


def test_validation_export_contains_probabilities_and_frame_ids(tmp_path):
    from types import SimpleNamespace
    from timepix.training.validation_export import export_validation
    records = [SampleRecord(i, str(a), f'{a}/x.txt', {'ToT': Path('unused')}) for i, a in enumerate([80, 82])]
    manifest = dict(samples={r.key: {'frame_group_key': f'group{r.angle}'} for r in records})
    manifest_path = tmp_path / 'split.json'
    manifest_path.write_text(json.dumps(manifest))
    loaders = {'val': SimpleNamespace(dataset=SimpleNamespace(records=records)),
               'train': SimpleNamespace(dataset=SimpleNamespace(records=records))}
    info = dict(angle_values=[80., 82.], split_path=str(manifest_path), label_type='angle_folder')
    payload = dict(labels=np.array([0, 1]), logits=np.array([[1., 0.], [0., 1.]]))
    cfg = dict(evaluation={'task_id': 'V6', 'experiment_id': 'V6-Base'}, training={'seed': 42})
    export_validation(tmp_path, payload, loaders, info, cfg)
    import csv
    rows = list(csv.DictReader((tmp_path / 'validation_predictions.csv').open()))
    assert rows[0]['sample_key'] == '80/x.txt'
    assert rows[0]['frame_group_key'] == 'group80'
    assert abs(float(rows[0]['prob_80']) + float(rows[0]['prob_82']) - 1) < 1e-7
    payload['labels'] = np.array([1, 0])
    with pytest.raises(ValueError):
        export_validation(tmp_path, payload, loaders, info, cfg)


def test_source_mapping_reuses_verified_rows_and_rejects_ambiguity(tmp_path):
    from scripts.prepare_carbon_controls import source_rows, archive_index
    import zipfile
    archive = tmp_path / 'C.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for member in ['C1/80/C_r00000_0001.txt', 'C1/82/C_r00000_0001.txt']:
            z.writestr(member, 'not extracted')
    records = [SampleRecord(0, '80', '80/C_r00000_0001_001.txt',
                            {'ToT': Path('/events/80/ToT/C_r00000_0001_001.txt')}),
               SampleRecord(0, '80', '80/C_r00000_0001_002.txt',
                            {'ToT': Path('/events/80/ToT/C_r00000_0001_002.txt')}),
               SampleRecord(1, '82', '82/C_r00000_0001_001.txt',
                            {'ToT': Path('/events/82/ToT/C_r00000_0001_001.txt')})]
    index = archive_index(archive)
    rows = source_rows(records, index, {}, {})
    assert rows[0]['frame_group_key'] == rows[1]['frame_group_key']
    assert rows[0]['frame_group_key'] != rows[2]['frame_group_key']
    index[('80', 'C_r00000_0001')].append('C2/80/C_r00000_0001.txt')
    with pytest.raises(ValueError, match='Ambiguous'):
        source_rows(records, index, {}, {})
    prior = {records[0].key: dict(angle='80', raw_frame_member='C1/80/wrong.txt')}
    with pytest.raises(ValueError):
        source_rows(records[:1], archive_index(archive), prior, {})


def test_configs_are_validation_only_and_pairs_share_protocol():
    from scripts.check_carbon_controls import CONFIGS, pair_protocol
    from timepix.config import load_experiment_config
    from timepix.config_validation import validate_experiment_config
    configs = [load_experiment_config(f'configs/experiments/{name}.yaml') for name in CONFIGS]
    for cfg in configs:
        validate_experiment_config(cfg)
        assert cfg['evaluation']['run_test'] is False
        assert cfg['split']['require_frame_groups'] is True
    assert pair_protocol(configs[0]) == pair_protocol(configs[1])
    assert pair_protocol(configs[2]) == pair_protocol(configs[3])


def test_immutable_artifacts_idempotent_not_overwritten(tmp_path):
    from scripts.prepare_carbon_controls import immutable_text, csv_text
    value = csv_text([dict(angle=80, event='80/file.txt')])
    path = tmp_path / 'file.csv'
    immutable_text(path, value)
    immutable_text(path, value)
    with pytest.raises(FileExistsError):
        immutable_text(path, 'different')


def test_required_manifest_never_falls_back_and_mask_builder_skips_stats(tmp_path, monkeypatch):
    from timepix.data.builders import build_dataloaders
    from timepix.data.frame_groups import grouped_manifest, frame_key, manifest_digest
    from timepix.config import load_experiment_config
    from timepix.data import builders
    rows = []
    for angle in ['80', '82']:
        directory = tmp_path / angle / 'ToT'
        directory.mkdir(parents=True)
        for i in range(4):
            name = f'C_r00000_{i:04d}_001.txt'
            a = np.zeros((50, 50)); a[24, 25] = 23.
            np.savetxt(directory / name, a)
            member = f'C1/{angle}/C_r00000_{i:04d}.txt'
            rows.append(dict(sample_key=f'{angle}/{name}', angle=angle, raw_frame_member=member,
                             frame_group_key=frame_key(angle, member)))
    cfg = load_experiment_config('configs/experiments/carbon_t7_mask_seed42.yaml')
    cfg['dataset']['class_names'] = ['80', '82']
    cfg['training'].update(num_workers=0, pin_memory=False)
    cfg['split']['path'] = str(tmp_path / 'split.json')
    with pytest.raises(ValueError, match='missing'):
        build_dataloaders(cfg, str(tmp_path))
    payload = grouped_manifest(rows)
    (tmp_path / 'split.json').write_text(json.dumps(payload))
    real = builders.compute_normalizer
    def spy(records, modalities, normalization_config, **kwargs):
        assert normalization_config == {}
        return real(records, modalities, normalization_config, **kwargs)
    monkeypatch.setattr(builders, 'compute_normalizer', spy)
    loaders, info = build_dataloaders(cfg, str(tmp_path))
    assert info['input_channels'] == 1
    assert info['normalizer_stats'] == {}
    image, label = next(iter(loaders['val']))
    assert set(image.unique().tolist()) == {0., 1.}


def test_summary_rejects_mismatched_training_protocol(tmp_path):
    from scripts.summarize_carbon_controls import summarize, METRICS
    from timepix.config import load_experiment_config
    import yaml
    paths = []
    for name in ['carbon_v6_base_seed42', 'carbon_v6_hires_seed42']:
        path = tmp_path / name; path.mkdir(); paths.append(path)
        cfg = load_experiment_config(f'configs/experiments/{name}.yaml')
        if 'hires' in name:
            cfg['training']['learning_rate'] = .1
        (path / 'config.yaml').write_text(yaml.safe_dump(cfg))
        metrics = dict(test_evaluated=False, validation={k: 0. for k in METRICS},
                       best_epoch=1, stopped_epoch=1, fit_seconds=1.)
        (path / 'metadata.json').write_text(json.dumps(dict(metrics=metrics, split_manifest_hash='same')))
        (path / 'validation_predictions.csv').write_text('header\n')
    with pytest.raises(ValueError, match='protocol'):
        summarize(paths)


def test_alias_cannot_split_one_raw_frame_into_different_groups():
    from timepix.data.frame_groups import grouped_manifest, frame_key
    rows = [dict(sample_key=f'80/x_{i}.txt', angle='80', raw_frame_member='C1/80/raw.txt',
                 canonical_frame_member=f'C1/80/fake{i}.txt', frame_group_key=frame_key('80', f'C1/80/fake{i}.txt'))
            for i in range(3)]
    with pytest.raises(ValueError, match='alias'):
        grouped_manifest(rows)


@pytest.mark.parametrize('name,section,key,value', [
    ('carbon_t7_mask_seed42', 'model', 'preserve_late_resolution', True),
    ('carbon_v6_hires_seed42', 'data', 'input_representation', 'hit_mask'),
    ('carbon_v6_hires_seed42', 'model', 'preserve_late_resolution', False),
])
def test_pair_protocol_rejects_wrong_cell(name, section, key, value):
    from scripts.check_carbon_controls import pair_protocol
    from timepix.config import load_experiment_config
    cfg = load_experiment_config(f'configs/experiments/{name}.yaml')
    cfg[section][key] = value
    with pytest.raises(ValueError, match='cell'):
        pair_protocol(cfg)


def test_empty_supplemental_mapping_writes_header():
    from scripts.prepare_carbon_controls import supplemental_csv
    assert supplemental_csv([]).startswith('sample_key,angle,')


def test_carbon_training_refuses_cpu_before_data_or_output_access(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(runner, 'build_dataloaders', lambda *a, **k: pytest.fail('Must fail before dataset scan'))
    with pytest.raises(RuntimeError, match='CUDA'):
        runner.run_experiment({'training': {'require_cuda': True}}, output_root=tmp_path / 'not_created')
    assert not (tmp_path / 'not_created').exists()
