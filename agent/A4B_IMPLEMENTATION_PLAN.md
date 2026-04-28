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

## Later Phases

Phase 2 should add a validation-only late logit fusion evaluator using existing
ToT and ToA checkpoints. It should select the fusion weight on validation data
only, then report the selected setting on test.

Phase 3 can add ToA scalar physical features such as `toa_span`, `toa_std`,
`toa_valid_count`, and `toa_p90_minus_p10`. This likely requires separating
loaded modalities from image modalities so ToA can provide scalar features
without also being passed as an image channel.

Phase 4 can add trainable multimodal models such as dual-stream feature concat
or GMU. This should use a new model-level key such as `multimodal_fusion` rather
than overloading the existing `fusion_mode`, which currently means CNN feature
plus handcrafted-feature fusion.
