# Architecture Notes

## Data Contract

Training code expects a dataset root with numeric angle folders. Inside each
angle folder, each enabled modality has a subfolder:

```text
root/
  <angle>/
    ToT/<sample files>
    ToA/<sample files>
```

Dataset-specific modality constraint:

- Alpha data has both `ToT` and `ToA`.
- C/proton data has only `ToT`; do not configure `modalities = ["ToT", "ToA"]`
  for that dataset unless ToA data is actually added later.

The loader pairs modalities by normalizing filenames: it removes the first
occurrence of the modality token from the stem and keeps the extension. Example:

```text
1_r0003_ToT_0002.txt -> 1_r0003__0002.txt
1_r0003_ToA_0002.txt -> 1_r0003__0002.txt
```

Only samples present in every enabled modality are used.

## Main Training Flow

`scripts/train.py` owns the CLI, and `timepix.training.runner.run_experiment`
owns the experiment loop:

1. Load an experiment YAML and merge its dataset YAML.
2. Apply CLI overrides from `--data-root`, `--output-root`, `--name`,
   `--resume`, and `--set`.
3. Validate the resolved config with `timepix.config_validation`.
4. Build train/validation/test dataloaders with `timepix.data.build_dataloaders`.
5. Build a model with `timepix.models.build_model`.
6. Build a loss function with `timepix.losses.build_loss`.
7. Configure optional CUDA AMP from `training.mixed_precision`.
8. Train with `timepix.training.trainer.train_one_epoch`.
9. Validate with `timepix.training.trainer.evaluate`.
10. Track the best validation result and save `best_model.pth` whenever it
   improves.
11. Save `training_log.csv`, `config.yaml`, `last_checkpoint.pth`,
   `predictions.csv`, `metrics.json`, and `metadata.json`.

`last_checkpoint.pth` is updated after each completed epoch and contains model,
optimizer, scheduler, optional AMP GradScaler state, best-metric, best-model,
config, and resume metadata. This supports
`python scripts/train.py --resume <last_checkpoint.pth>`.

## Dataset Details

`timepix.data.dataset.collect_samples`:

- Lists numeric angle folders.
- Sorts them numerically.
- Maps them to consecutive labels: `{0: "15", 1: "30", ...}`.
- Collects paired modality paths into `SampleRecord` objects.

`timepix.data.splits.stratified_split`:

- Perform per-class stratified splits.
- Use `random.Random(seed)`, where the config-level seed is `split.seed` when
  present and falls back to `training.seed` for older configs.
- Three-way split requires ratios summing to 1.0.

`TimepixDataset`:

- Loads text matrices through `np.loadtxt`.
- Converts them to tensors with `torch.as_tensor(...).unsqueeze(0)`.
- Optionally center-crops.
- Optionally rotates training samples by `0`, `90`, `180`, and `270` degrees.
- Optionally normalizes per modality.
- Optionally appends handcrafted features.
- Returns either `(sample_tensor, label)` or
  `(sample_tensor, label, handcrafted_features)`.

## Normalization

Per-modality z-score normalization is configured in experiment YAML under
`normalization`. Statistics are computed only on the training split. Options:

- `enabled`: whether to normalize this modality.
- `log1p`: apply `log1p(max(x, 0))` before statistics and application.
- `ignore_zero`: ignore zeros when computing statistics.

Handcrafted feature standardization is configured under
`handcrafted_features.standardize`.

## Handcrafted Features

Currently registered handcrafted features:

- `total_energy`: sum of the modality array.

Feature flags are stored by modality in YAML:

```python
handcrafted_features = {
    "ToT": {"total_energy": True},
    "ToA": {"total_energy": False},
}
```

Enabled handcrafted features are concatenated with CNN features inside models
that support this path.

## Model Interface

The expected model contract is:

```python
class TimepixModel(nn.Module):
    def forward(self, samples, handcrafted_features=None) -> ModelOutput: ...
```

Models return `ModelOutput`, which may contain classification `logits` and/or a
regression tensor. The current factory supports `resnet18`,
`resnet18_no_maxpool`, `resnet18_maxpool`, `resnet18_original`,
`shallow_resnet`, `shallow_cnn`, `densenet121`, `efficientnet_b0`,
`convnext_tiny`, and `vit_tiny`.

`resnet18` is an alias for `resnet18_no_maxpool`. Both ResNet18 variants accept
`model.conv1_kernel_size`, `model.conv1_stride`, `model.conv1_padding`,
`model.dropout`, `model.feature_dim`, and `model.hidden_dim`. The older
`model.kernel_size` field remains supported as an alias for
`model.conv1_kernel_size`.

`resnet18_original` is a separate baseline model file with the original
torchvision stem: conv1 `7x7/stride=2/padding=3` plus first maxpool. It still
uses the project `FeatureFusion` and task head, so handcrafted features,
classification/regression labels, and all existing loss choices remain
compatible.

DenseNet121, EfficientNet-B0, ConvNeXt-Tiny, and ViT-Tiny are adapted in
`timepix.models.torchvision_backbones`. They replace the input stem for 1/2
Timepix channels, project the backbone output to `model.feature_dim`, and then
reuse the same `FeatureFusion` and task head as the ResNet models. `vit_tiny`
is a local 50x50 adapter with default `model.patch_size=10`; pretrained weights
are not provided for this adapter.

## Losses and Metrics

`timepix.losses` supports:

- `cross_entropy`: wrapper around `nn.CrossEntropyLoss`.
- `emd`: ordinal Earth Mover's Distance / Wasserstein-style loss over ordered
  angle classes.
- `mse`: regression.
- `smooth_l1`: regression.

Angle-aware metrics:

- `compute_angle_mae`: classification MAE using argmax angle and probability
  weighted angle.
- `compute_regression_mae`: regression MAE/RMSE after multiplying by
  `max_angle`.
- `p90_error`: 90th percentile of absolute angle error in degrees. For
  classification it is based on argmax angle by default; weighted-angle P90 is
  also recorded as `p90_error_weighted`.

## Experiment Scripts

`scripts/train.py` runs a single experiment YAML.

`scripts/run_grid.py` expands a grid config and runs multiple experiments.
It supports `--skip-existing`, `--continue-on-error`, and CSV grid manifests
under `outputs/grid_manifests/`.

`scripts/summarize.py` rebuilds summary CSV files from experiment outputs,
including model hyperparameters such as conv1 kernel/stride/padding and
dropout, early-stopping state, training hyperparameters, mixed-precision state,
split seed/hash, fit/test/total timing, and git metadata.

`scripts/aggregate_seeds.py` aggregates a summary CSV into `mean`/`std` rows for
repeated-seed certification.

`scripts/search_hparams.py` runs Optuna hyperparameter search from
`configs/search/*.yaml`. It consumes the top-level `search` section, samples
dotted config paths such as `training.learning_rate`, applies them to the same
resolved experiment config used by `scripts/train.py`, and then calls
`timepix.training.runner.run_experiment` for each trial. Search outputs are
written under `outputs/hparam_search/`; Optuna storage defaults to SQLite under
`outputs/optuna/`. The objective should use validation metrics only.

`configs/experiments/a1_resnet18_original_baseline.yaml` defines the original
ResNet18 baseline for A1. `configs/experiments/a1_structure_adaptation.yaml`
defines the A1 alpha ToT ResNet18 structure-adaptation grid: no-maxpool vs
maxpool, conv1 kernel sizes 2/3/5, conv1 strides 1/2, and dropout 0/0.1/0.3.
`configs/experiments/compare_mixed_precision.yaml` compares FP32 and CUDA AMP
under the current A1 best structure: `resnet18_no_maxpool`,
`conv1_kernel_size=2`, `conv1_stride=1`, and `dropout=0.3`.
`configs/search/a2_alpha_resnet18_tot_training.yaml` defines the A2
training-hyperparameter search. The current A2-best settings are captured in
`configs/experiments/alpha_tot_a2_best_base.yaml` for later ablations and model
comparisons. `configs/experiments/a3_backbone_comparison.yaml` compares
ShallowCNN, ShallowResNet, ResNet18, DenseNet121, EfficientNet-B0,
ConvNeXt-Tiny, and ViT-Tiny under the same Alpha ToT CE one-hot setting with a
single fixed training seed. `configs/experiments/a4_modality_comparison.yaml`
compares ToT, ToA, and ToT+ToA while reusing a paired split manifest generated
from the dual-modality sample intersection. `configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml`
re-runs the current A2 best configuration with `training.seed` values 42/43/44
while keeping `split.seed=42`.

Legacy scripts under `Program/` are preserved as references during the refactor.
