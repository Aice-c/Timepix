# Review Notes

This review is based on static inspection plus lightweight validation. It did
not run model training because the available Python environment lacks `torch`.

## Validation Performed

- Passed: `python -m compileall -q Program ProcessProgram generate_presentation.py generate_ppt.py generate_literature_ppt.py generate_midterm_report.py near_vertical_analysis.py near_vertical_analysis_v2.py`
- Failed due environment: `python Program\test_losses.py`
  - Python: 3.14.0
  - Error: `ModuleNotFoundError: No module named 'torch'`

## High-Priority Bugs

### 1. Model factory breaks several advertised models

`Program/model/utils.py` instantiates every model as:

```python
model_class(num_classes=num_classes, task=...)
```

Only `Resnet18.Model` and `ShallowResNet.Model` accept `task`. Models such as
`ShallowCNN.Model` and `Densenet201.Model` accept only `num_classes`, so selecting
them through `Config.model_name` raises `TypeError: unexpected keyword argument
'task'`.

Suggested fix: standardize every model constructor to `__init__(self,
num_classes, task=None)` or make the factory introspect the signature. Prefer
standardizing the interface.

### 2. Some configured model names are not loadable or not compatible

`Efficientnetb0.py` defines `class Efficientnetb0`, not `class Model`, so the
factory cannot load it. `CNN.py` is empty. `Resnet34`, `Resnet50`, and
`Shufflenet` are legacy torchvision wrappers with default 3-channel input, while
the dataset usually produces 1 or 2 channels.

Suggested fix: either remove these names from comments/search spaces or update
each file to the current `Model` contract and input-channel adaptation.

### 3. `sweep.py` returns a result dict instead of a float

`run_experiment` returns a dictionary. `sweep.objective` returns that dictionary
directly, but Optuna objective functions must return a scalar value.

Suggested fix:

```python
result = run_experiment(overrides=overrides, save_plots=False)
return float(result["best_vacc"])
```

### 4. `config.inchannel` can go stale when modalities are overridden

`Config.inchannel` is computed once at import time from `modalities`. When
scripts call `run_experiment(overrides={"modalities": ["ToT", "ToA"]})`,
`modalities` changes but `inchannel` does not. Some models use
`config.input_channels()` dynamically, while others use `config.inchannel`.

Suggested fix: remove the stored `inchannel` attribute and use
`config.input_channels()` everywhere, or update dependent fields after overrides.

### 5. Empty validation/test loaders can cause division by zero

The split code can create empty validation/test subsets for tiny classes. The
trainer/evaluator then compute `sum(loss_list) / len(loss_list)`, which raises
on empty loaders.

Suggested fix: validate minimum samples per class for the selected split, or
raise a clear error before constructing loaders.

## Medium-Priority Risks

### 6. Default data path points outside the repository

`Config.data_dir` defaults to `../Alpha0`, while the local clean dataset is under
`Data/Alpha_Clean`. A fresh agent running `Program/main.py` from this checkout
will likely hit `FileNotFoundError`.

Suggested fix: expose `--data_dir` on `main.py` or set the default to a checked
repository path when appropriate.

### 7. Angle folder parsing accepts only integer folder names

`collect_samples` rejects labels such as `15deg`, `15.0`, or `0_1_merged`. This
is fine for the current clean alpha data but brittle for merged or decimal angle
datasets.

Suggested fix: parse numeric prefixes more flexibly or support an explicit label
manifest.

### 8. Saved split indices are not validated against the current dataset

`load_split_indices` trusts old integer indices. If the dataset root, file
ordering, or modalities change, a stale split file can silently select the wrong
samples.

Suggested fix: save a manifest of normalized sample IDs/paths and validate it
before reusing a split.

### 9. `rebuild_summary.py` expects `param_count` in `config.yaml`, but it is not saved

`rebuild_summary.extract_param_count` reads `cfg["param_count"]["total"]`, but
`main._save_config_yaml` does not include `param_count`. Rebuilt summaries will
therefore report zero parameters.

Suggested fix: include `param_count` in `config.yaml` or read it from another
saved metadata file.

### 10. `ExperimentLogger` dynamic columns corrupt CSV shape

When new columns appear, `ExperimentLogger.log` appends rows using an expanded
field list but does not rewrite the header of an existing CSV. This creates rows
with more columns than the header.

Suggested fix: rewrite the CSV with the expanded header, or write JSONL instead.

### 11. `Resnet50.forward` runs the backbone twice

`Resnet50.forward` computes logits once, then calls `self.model(x)` again inside
`F.softmax`. This doubles compute and can update BatchNorm statistics twice
during training.

Suggested fix: compute probabilities from the existing `logits`.

### 12. ResNet18 may download pretrained weights unexpectedly

`Resnet18.CNN` uses `resnet18(pretrained=True)`. In a server or sandbox without
cached torchvision weights, this can trigger a network download or fail.

Suggested fix: use the modern `weights=` API and make pretrained weights a
config option, defaulting to no download for reproducibility.

## Refactor Recommendations

1. Replace the mutable global `Config.config` class with a dataclass or plain
   config object passed through the pipeline.
2. Add a small CLI to `main.py` for `--data_dir`, `--modalities`, `--model`,
   `--epochs`, and `--output_dir`.
3. Standardize the model contract across all model files; mark unsupported
   features explicitly.
4. Move obsolete or legacy model files into `Program/model/legacy/` or remove
   their names from advertised choices.
5. Treat `run_ablation.py` as a temporary comparison script, not the final
   thesis experiment framework. A future experiment runner should support
   controlled comparisons across models, losses, labels, modalities, and
   handcrafted-feature settings.
6. Rebuild or replace `sweep.py` together with the new experiment runner. The
   current Optuna script is both technically broken and too sparse in metadata
   logging for serious comparison work.
7. Add pytest-style tests for dataset pairing, split sizes, model factory
   loading, loss functions, and a one-batch train/eval smoke test.
8. Persist experiment metadata in a single `metadata.json` containing config,
   label map, param count, git commit, split manifest hash, and metrics.
9. Separate data preprocessing notebooks from reusable Python modules. Keep
   notebooks for exploration, but extract stable logic into scripts with CLI
   arguments and no hard-coded paths.
10. Normalize output locations: current code writes both `output/` and
   `Program/output/`.
11. Consider using weighted sampling or class-balanced loss only after recording
   class counts per split; current `weight` defaults assume exactly four classes.
12. Rename `evaluater.py` to `evaluator.py` after updating imports, or at least
    document the typo to avoid duplicate modules.
