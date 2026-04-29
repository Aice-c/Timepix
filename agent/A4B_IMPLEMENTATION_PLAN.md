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

## Phase 4: Frozen-Logit Selector

Implemented as:

```text
scripts/evaluate_selector_fusion.py
```

This phase freezes the trained ToT and `relative_minmax/no mask` checkpoints.
It trains a lightweight selector on train-split logits/probabilities/confidence
features, selects the threshold on validation, and reports test metrics once.
`primary_only` is included as a validation-selectable fallback.

## Later Phases

Phase 5 can add ToA scalar physical features such as `toa_span`, `toa_std`,
`toa_valid_count`, and `toa_p90_minus_p10`. This likely requires separating
loaded modalities from image modalities so ToA can provide scalar features
without also being passed as an image channel.

Phase 6 can add trainable multimodal models such as dual-stream feature concat
or GMU. This should use a new model-level key such as `multimodal_fusion` rather
than overloading the existing `fusion_mode`, which currently means CNN feature
plus handcrafted-feature fusion.
