# A4b ToA Fusion Implementation Plan

## Goal

A4 showed that `ToT` single modality is stronger than `ToT+ToA` early channel
concat under the current raw/log1p ToA representation. A4b asks a narrower
question: can ToA help if it is represented or fused more carefully, while
keeping the A2-best training setup and the restored `Alpha_100` paired split?

Fixed baseline for A4b:

- Dataset: `Alpha_100`
- Split: `outputs/splits/Alpha_100_ToT-ToA_seed42_0.8_0.1_0.1.json`
- Backbone: `resnet18_no_maxpool`
- Stem: `conv1_kernel_size=2`, `conv1_stride=1`, `conv1_padding=0`
- Loss/label: `cross_entropy`, `onehot`
- Handcrafted features: disabled unless a later ToA-scalar feature phase says otherwise
- Training: A2 best hyperparameters, AMP enabled

## Phase 1: ToA Input Representation

Implemented first because it is low-risk and does not change model structure.

New config fields:

```yaml
data:
  toa_transform: relative_minmax
  add_hit_mask: false
```

Supported `toa_transform` values:

- `none`
- `raw_log1p`
- `relative_minmax`
- `relative_centered`
- `relative_rank`

When `add_hit_mask: true`, the image input for ToT+ToA becomes:

```text
[ToT, transformed_ToA, hit_mask]
```

The same ToA transform helper is used by `TimepixDataset` and
`compute_normalizer`, so normalization statistics match the actual model input.

Configs:

```text
configs/experiments/a4b_toa_transform.yaml
configs/experiments/a4b_toa_transform_seed42.yaml
```

Current result:

- Relative ToA representations improved over the original raw/log1p ToT+ToA
  early fusion.
- None of the phase-1 variants exceeded the ToT-only baseline.
- `relative_centered, no mask` produced the best Test Acc among phase-1 variants
  at 68.79%, still below the ToT baseline at 70.48%.
- `relative_minmax, no mask` improved the 30 deg F1 to 0.447, suggesting local
  complementary signal, but not enough to improve the overall model.

Phase 2 adds a validation-only late logit fusion evaluator using existing ToT
and ToA checkpoints:

```text
scripts/evaluate_logit_fusion.py
```

It computes:

```text
logits = (1 - alpha_toa) * logits_tot + alpha_toa * logits_toa
```

and selects `alpha_toa` on validation data only. The selected alpha is then
reported on test. This phase does not train a new model and is intended to
quickly check whether ToA has decision-level complementary information.

Current result:

- Validation selected `alpha_toa=0.00`, meaning the selected late-fusion model is
  the ToT-only baseline.
- Some nonzero alpha values slightly improved test accuracy, but they were not
  selected by validation and therefore should not be used as evidence of a
  reliable ToA gain.
- Phase 1 and Phase 2 together suggest that ToA may contain weak or class-local
  information, but current early fusion and late logit fusion are not sufficient
  to show a robust overall improvement.

## Phase 2.5: Prediction Complementarity Diagnosis

Implemented as:

```text
scripts/analyze_prediction_complementarity.py
```

This phase uses existing `predictions.csv` files only. It measures whether ToA
or a ToT+ToA candidate is correct on samples where ToT is wrong, whether it has
lower angle error on ToT failures, and the oracle upper bound if a selector
could choose the better prediction per sample.

Current seed-42 result:

- ToA alone gives an oracle accuracy of 77.83% with ToT, a +7.36 percentage
  point upper-bound gain over ToT.
- `relative_minmax, no mask` gives the best oracle result among A4b-1
  candidates: 81.51% oracle accuracy and 3.698 deg oracle MAE.
- For the 30 deg class, ToA alone does not rescue ToT failures, but
  `relative_minmax, no mask` raises the oracle accuracy from 29.66% to 55.17%.
- This supports continuing with selective/gated/residual fusion, because the
  problem appears to be choosing when to trust the auxiliary prediction, not a
  complete absence of complementary information.

## Phase 3: Oracle Controls

Implemented as:

```text
scripts/evaluate_oracle_complementarity.py
```

Current result:

- ToT-vs-ToT seed-control oracle gain is small: about +2.33% on validation and
  +2.55% on test.
- ToT vs `relative_minmax/no mask` is much stronger: +10.19% oracle gain on
  validation and +11.03% on test.
- For 30 deg, ToT vs `relative_minmax/no mask` gives +27.08% oracle gain on
  validation and +25.52% on test.

Conclusion: the relative candidate's complementarity is larger than ordinary
seed diversity, so the next question is whether a selector/gate can learn when
to use it.

## Phase 4: Selector Fusion

Implemented as:

```text
scripts/evaluate_selector_fusion.py
```

This phase freezes the trained ToT and `relative_minmax/no mask` checkpoints.
It has three numbered variants:

- A4b-4a: rule-based selector, selected on validation.
- A4b-4b: train-logit selector, trained on train split and selected on
  validation; exploratory because train logits may be overconfident.
- A4b-4c: validation-CV selector, using validation cross-fitting for threshold
  selection and final validation fit for test reporting.

`primary_only` is included as a validation-selectable fallback in every variant.
The earlier unnumbered A4b-4 selector output is discarded and should be rerun
under A4b-4a/4b/4c.

Current result:

- A4b-4a rule selector: 70.97% test accuracy, +0.50% over ToT.
- A4b-4b train-logit selector: 71.17% test accuracy, +0.70% over ToT.
- A4b-4c validation-CV selector: 70.38% test accuracy, -0.10% versus ToT.
- Oracle remains 81.51% test accuracy.

Interpretation: simple rule/train-logit selectors can obtain small real gains,
but the stricter validation-CV selector does not beat ToT. This means the
oracle complementarity is not yet reliably learnable from frozen logits alone.

## Later Phases

The active order after the A4b-4 results is intentionally conservative:

1. A4b-4d switch diagnostics. No training. Recompute the selected A4b-4a rule
   and report switch precision, switch recall, harmful/neutral switch rates,
   per-class switch behavior, and selector-score distributions.
2. A4b-4e three-seed selector confirmation, if the A4b-4a result is promoted
   from diagnostic evidence to a formal positive method. Only the key
   `relative_minmax/no mask` candidate needs seeds 43 and 44; ToT three-seed
   baselines already exist.
3. A4b-5 entropy soft gate. Use the A4b-4a entropy advantage signal to build a
   validation-selected sample-wise soft interpolation; do not train a large gate
   first.
4. A4b-6 constrained residual interpolation. Keep ToT primary and move logits
   partly toward the candidate with a small validation-selected beta grid.
5. A4b-7 ToA-only relative controls. Run only compact `relative_minmax` and
   `relative_rank` no-mask ToA-only diagnostics unless new evidence requires a
   larger grid.
6. A4b-8 ToT image plus ToA scalar physical features. Candidate features include
   `toa_span`, `toa_std`, `toa_valid_count`, `toa_p90_minus_p10`, `toa_iqr`,
   `toa_relative_mean`, `toa_relative_std`, `toa_major_axis_slope_abs`, and
   `toa_major_axis_corr_abs`. This likely requires separating loaded modalities
   from image modalities so ToA can provide scalar features without also being
   passed as an image channel.
7. A4b-9 optional end-to-end gated expert fusion. Use a new model-level key such
   as `multimodal_fusion` rather than overloading the existing `fusion_mode`,
   which currently means CNN feature plus handcrafted-feature fusion.

Deferred for now:

- GMU and FiLM until low-cost selector/gate variants are better understood.
- MMTM, ordinary feature concat, and larger mask/transform grids.

A4b-4d implementation:

```text
scripts/analyze_selector_switches.py
```

Default rule:

```text
entropy_adv_0p03
```

The script reloads the same frozen experts as A4b-4, applies the fixed rule
without selecting any new threshold on test, and writes JSON plus summary,
per-class, per-sample, and score-distribution CSV files.

A4b-4e implementation:

```text
configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml
scripts/aggregate_selector_fusion.py
```

This phase does not retrain seed42. It combines:

- seed42 candidate from `a4b_toa_transform_seed42`
- seed43/44 candidates from `a4b_4e_relative_minmax_no_mask_seed43_44`
- ToT seeds 42/43/44 from `a2_best_3seed`

`aggregate_selector_fusion.py` keeps `primary_only`, `candidate_only`,
`oracle`, and the validation-selected rule row from each
`evaluate_selector_fusion.py` summary, then writes mean/std metrics.
