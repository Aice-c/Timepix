# A4b Selector Fusion Plan

This document records the implementation plan after the A4b-2.5 prediction-complementarity analysis.

## Current Evidence

A4b-2.5 showed that `ToT baseline` remains the best single model, but `ToT + relative_minmax ToA, no mask` is complementary on part of the test set:

- ToT baseline test accuracy: 70.48%
- relative_minmax/no-mask test accuracy: 67.10%
- ToT OR relative_minmax/no-mask oracle test accuracy: 81.51%
- 30 deg oracle accuracy gain over ToT: +25.52 percentage points

Interpretation:

- Do not replace ToT as the primary model.
- Do not continue asking whether ToA should be added unconditionally.
- The useful question is whether a selector/gate can learn when to trust the ToT expert and when to trust the relative-ToA expert.

Important risk:

- The candidate model is not ToA-only; it is an early-fusion `ToT + relative ToA` model.
- The oracle gain may come from ToA information, a different ToT decision boundary, random seed effects, or a mixture of these.
- A ToT-vs-ToT seed oracle control is required before claiming ToA-specific complementarity.

## Existing Project Facts

- `predictions.csv` is saved only for the test split and contains labels, predicted labels, predicted angles, weighted angles, and errors. It does not contain full logits.
- `scripts/analyze_prediction_complementarity.py` reads existing `predictions.csv` files and computes test-set overlap/oracle diagnostics.
- `scripts/evaluate_logit_fusion.py` already knows how to reload checkpoints, rebuild dataloaders, run validation/test inference, and compute metrics from logits.
- Current A4/A4b configs inherit `augmentation.rotation_90: true`, and `build_dataloaders()` creates the train loader with `training=True` and `shuffle=True`.
- Selector training must not use that augmented/shuffled train loader. It needs deterministic evaluation loaders for train/val/test: `training=False`, no rotation expansion, and `shuffle=False`.
- Sample alignment should be checked by split name, labels, and sample keys, not only by row index.

## Stage 1: Oracle Controls Before New Training

Goal: determine whether the observed oracle gain is larger than ordinary seed-to-seed model diversity.

Implemented script:

```text
scripts/evaluate_oracle_complementarity.py
```

Required capabilities:

- Compare arbitrary trained runs, not only ToT-vs-ToA.
- Support `train`, `val`, and `test` splits by reloading checkpoints.
- Build deterministic eval loaders for every split.
- Save summary CSV, by-class CSV, and JSON.
- Validate that two runs share label map, split keys, and labels.

Required analyses:

```text
ToT_seed42 vs ToT_seed43
ToT_seed42 vs ToT_seed44
ToT_seed43 vs ToT_seed44
ToT_seed42 vs relative_minmax_no_mask_seed42
same comparisons on validation and test
```

Current project mapping:

- Use `outputs/experiments/a2_best_3seed` for the ToT seed-control runs. This is the current canonical `Alpha_100 + ToT + resnet18_no_maxpool + A2 best` three-seed group.
- Use `outputs/experiments/a4b_toa_transform_seed42` for the first relative ToT+ToA candidate replay. At present this is seed42 only, so A4b-3b is a seed42 validation/test diagnostic until the A4b transform grid is rerun for more seeds.
- Do not use `a4_modality_comparison_seed42` as the formal ToT seed-control source; it only has one ToT seed and cannot answer the seed-diversity question.
- Because `a2_best_3seed` was trained before the dataset rename, its run config still points to `/root/autodl-tmp/Alpha` and the default split name `Alpha_ToT_seed42_0.8_0.1_0.1.json`. When replaying it on the current server, pass `--data-root /root/autodl-tmp/Alpha_100` and create `Alpha_ToT...` as a compatibility copy of `Alpha_100_ToT...`; do not edit the historical run metadata.
- Select `relative_minmax/no mask` first because it was the strongest A4b-2.5 complementarity candidate, not because it had the highest standalone accuracy. It reached 81.51% ToT-oracle test accuracy, +11.03 percentage points oracle gain, and improved 30 deg oracle accuracy from 29.66% to 55.17%.

Decision rule:

- If ToT-vs-ToT oracle gain is close to ToT-vs-relative gain, treat much of the complementarity as seed/model-diversity effect.
- If ToT-vs-relative is clearly higher, especially for 30 deg, then relative ToA gives stronger evidence of useful auxiliary information.

## Stage 2: Frozen-Logit Selector Fusion

Goal: test whether the oracle choice can be learned from model outputs without retraining ResNet experts.

Add a script, recommended name:

```text
scripts/evaluate_selector_fusion.py
```

Inputs:

- Primary run: ToT baseline checkpoint.
- Auxiliary run: relative_minmax/no-mask checkpoint.
- Splits: train/val/test from the same fixed split manifest.

Model-output features:

```text
primary logits
auxiliary logits
primary probabilities
auxiliary probabilities
primary top1 confidence
auxiliary top1 confidence
primary top1-top2 margin
auxiliary top1-top2 margin
primary entropy
auxiliary entropy
models disagree flag
predicted angle difference
probability difference features
```

Selector labels:

- Primary target: `aux_error < primary_error`, because angle MAE matters.
- Conservative target for ablation: `aux_correct and primary_wrong`.

Training protocol:

- Train selector on train split only.
- Select threshold and fusion type on validation split only.
- Report test metrics once.
- Never use test to choose selector type, threshold, or feature set.

Fusion modes:

```text
hard_switch:
  choose auxiliary prediction if selector_prob > threshold, otherwise primary

soft_gate:
  logits = (1 - g) * primary_logits + g * auxiliary_logits

residual_gate:
  logits = primary_logits + g * (auxiliary_logits - primary_logits)
```

Recommended first selector:

- LogisticRegression from scikit-learn.
- Use `requirements-analysis.txt` dependency set.
- Optional second selector: small MLP, only if logistic regression underfits.

Report:

- ToT baseline metrics.
- Auxiliary expert metrics.
- Oracle upper bound.
- Selector fusion metrics.
- Selection rate overall and by class.
- 30 deg precision/recall/F1 and confusion matrix.
- Validation-selected threshold.

## Stage 3: ToA-Only Relative Controls

Goal: isolate whether relative ToA itself is stronger than raw ToA.

This is mostly configuration work; current `data.toa_transform` already supports:

```text
relative_minmax
relative_rank
```

Recommended compact configs:

```text
ToA_relative_minmax_no_mask
ToA_relative_rank_no_mask
```

Do not run the full mask grid again unless Stage 1/2 indicates it is necessary.

Interpretation:

- If ToA-only relative improves over raw ToA, relative time representation is useful by itself.
- If ToA-only remains weak but ToT+relative is complementary, ToA likely needs ToT spatial context.

## Stage 4: ToT Image + ToA Scalar Features

Goal: test a physically interpretable ToA representation.

Current blocker:

- `timepix/data/features.py` only supports `total_energy`.
- The current dataset uses `dataset.modalities` for both image channels and handcrafted-feature loading.
- A pure `ToT image + ToA scalar features` experiment needs separate image modalities and feature modalities.

Required data changes:

- Extend feature registry with ToA scalar features:

```text
toa_valid_count
toa_span
toa_std
toa_p90_minus_p10
toa_iqr
toa_relative_mean
toa_relative_std
toa_major_axis_slope_abs
toa_major_axis_corr_abs
```

- Let records include auxiliary feature modalities even when image modalities are only `[ToT]`.
- Keep model-side handcrafted feature fusion mostly unchanged; existing ResNet models already accept handcrafted vectors.

Recommended config direction:

```yaml
dataset:
  modalities: [ToT]

handcrafted_features:
  enabled: true
  source_modalities: [ToA]
  features:
    ToA:
      - toa_span
      - toa_std
      - toa_p90_minus_p10
      - toa_major_axis_slope_abs
      - toa_major_axis_corr_abs
```

## Stage 5: End-to-End Gated Expert Fusion

Only start this after Stage 1/2.

Preferred route:

- Expert A: ToT-only ResNet branch.
- Expert B: ToT+relative-ToA ResNet branch.
- Gate is biased toward Expert A at initialization.

Why this route:

- It directly matches the strongest A4b-2.5 oracle pair.
- It is more grounded than generic two-stream concat.

Implementation implications:

- Add a new model such as `resnet18_gated_expert_fusion`.
- The model can receive stacked `[ToT, ToA_relative]` channels and internally slice:
  - `x_tot = x[:, :1]`
  - `x_relative = x[:, :2]`
- Return a normal `ModelOutput` first; auxiliary losses and gate diagnostics can be added later.

Defer:

- MMTM
- ordinary feature concat without gating
- large mask grids
- FiLM, unless selector fusion succeeds and time remains

## Strict Evaluation Rules

- Test set is for final reporting only.
- Oracle metrics are diagnostic upper bounds, not trainable performance.
- Any threshold, selector type, alpha, feature set, or gate variant must be chosen on validation.
- Report both accuracy and angle-aware metrics: MAE, P90, macro-F1, per-class F1, and confusion matrix.
- For 30 deg, always report class-specific metrics because current evidence shows the main complementarity there.
