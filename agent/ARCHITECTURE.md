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

`Program/main.py` owns the experiment loop:

1. Apply temporary config overrides passed by scripts such as `sweep.py` or
   `run_ablation.py`.
2. Build datasets with `src.dataset.build_datasets`.
3. Build a model with `model.utils.build_model`.
4. Build a loss function with `src.losses.build_loss_function`.
5. Train with `src.trainer.trainer`.
6. Validate with `src.evaluater.evaluater`.
7. Track the best validation result.
8. Optionally evaluate the test split.
9. Save `best_model.pth`, `training_log.csv`, `config.yaml`, curves, and
   optional `test_predictions.npz`.

The code uses a mutable class `Config.config` rather than immutable config
objects. Scripts mutate this class directly through override dictionaries.

## Dataset Details

`src.dataset.collect_samples`:

- Lists numeric angle folders.
- Sorts them numerically.
- Maps them to consecutive labels: `{0: "15", 1: "30", ...}`.
- Collects paired modality paths into `SampleRecord` objects.

`split_samples` and `split_samples_three_way`:

- Perform per-class stratified splits.
- Use `random.Random(seed)`.
- Three-way split requires ratios summing to 1.0.

`ParticleDataset`:

- Loads text matrices through `np.loadtxt`.
- Converts them to tensors with `torchvision.transforms.ToTensor`.
- Optionally center-crops.
- Optionally rotates training samples by `0`, `90`, `180`, and `270` degrees.
- Optionally normalizes per modality.
- Optionally appends handcrafted features.
- Returns either `(sample_tensor, label)` or
  `(sample_tensor, label, handcrafted_features)`.

## Normalization

Per-modality z-score normalization is configured in `Config.standardization`.
Statistics are computed only on the training split. Options:

- `enabled`: whether to normalize this modality.
- `log1p`: apply `log1p(max(x, 0))` before statistics and application.
- `ignore_zero`: ignore zeros when computing statistics.

Handcrafted feature standardization is separate and configured by
`Config.handcrafted_standardization`.

## Handcrafted Features

Currently registered handcrafted features:

- `total_energy`: sum of the modality array.

Feature flags are stored by modality:

```python
handcrafted_features = {
    "ToT": {"total_energy": True},
    "ToA": {"total_energy": False},
}
```

Enabled handcrafted features are concatenated with CNN features inside models
that support this path.

## Model Interface

The expected model module contract is:

```python
class Model(nn.Module):
    def __init__(self, num_classes, task=None): ...
    def forward(self, samples, handcrafted_features=None):
        return logits_or_regression, probabilities_or_none, pred_or_none
```

In practice, only `Resnet18` and `ShallowResNet` currently match this contract
well enough for classification and regression. `ShallowCNN` supports
handcrafted classification but does not accept `task`. Several older model
files are not compatible with the current factory or input-channel contract.
See `agent/REVIEW_NOTES.md`.

## Losses and Metrics

`src.losses` supports:

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

`run_ablation.py` defines six thesis-style comparisons:

- A: ResNet18 + CE classification.
- B: ResNet18 + EMD classification.
- C: ShallowResNet + CE classification.
- D: ShallowResNet + EMD classification.
- E: ResNet18 + SmoothL1 regression.
- F: ShallowResNet + SmoothL1 regression.

`rebuild_summary.py` rebuilds summary CSV/JSON files from experiment outputs.

`generate_figures.py` and `generate_report.py` consume ablation outputs and
produce thesis figures and text.

`sweep.py` is intended for Optuna hyperparameter search but currently has a
return-value bug; see the review notes.
