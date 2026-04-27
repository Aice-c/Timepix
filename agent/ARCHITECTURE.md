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
3. Build train/validation/test dataloaders with `timepix.data.build_dataloaders`.
4. Build a model with `timepix.models.build_model`.
5. Build a loss function with `timepix.losses.build_loss`.
6. Train with `timepix.training.trainer.train_one_epoch`.
7. Validate with `timepix.training.trainer.evaluate`.
8. Track the best validation result and save `best_model.pth` whenever it
   improves.
9. Save `training_log.csv`, `config.yaml`, `last_checkpoint.pth`,
   `predictions.csv`, `metrics.json`, and `metadata.json`.

`last_checkpoint.pth` is updated after each completed epoch and contains model,
optimizer, scheduler, best-metric, best-model, config, and resume metadata.
This supports `python scripts/train.py --resume <last_checkpoint.pth>`.

## Dataset Details

`timepix.data.dataset.collect_samples`:

- Lists numeric angle folders.
- Sorts them numerically.
- Maps them to consecutive labels: `{0: "15", 1: "30", ...}`.
- Collects paired modality paths into `SampleRecord` objects.

`timepix.data.splits.stratified_split`:

- Perform per-class stratified splits.
- Use `random.Random(seed)`.
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
regression tensor. The current factory supports `resnet18`, `shallow_resnet`,
and `shallow_cnn`.

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

`scripts/summarize.py` rebuilds summary CSV files from experiment outputs.

Legacy scripts under `Program/` are preserved as references during the refactor.
