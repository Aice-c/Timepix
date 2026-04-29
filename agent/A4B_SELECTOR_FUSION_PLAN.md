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

## Command And Summary Policy

For every A4b comparison or diagnostic phase, the documented execution block must include both the run command and the result collection command.

- Training/grid phases must include the server `run_grid.py` command and the matching `summarize.py --group ... --out ...` command.
- Multi-seed phases must also include the mean/std aggregation command, using `aggregate_seeds.py` for standard training runs or `aggregate_selector_fusion.py` for selector/gate summary CSVs.
- Checkpoint diagnostic phases that do not create a normal `outputs/experiments/<group>/` run directory must explicitly name their `--output-summary`, `--output-by-class`, `--output-json`, and any sample/distribution CSV outputs.
- Test metrics remain final-report-only; validation is the only source for choosing rules, thresholds, gates, betas, regularization, or final variants.

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

Observed A4b-3 result:

- ToT-vs-ToT seed-control oracle gain is small: about +2.33% on validation and +2.55% on test.
- ToT-vs-ToT 30 deg oracle gain is also small: about +2.55% on validation and +1.15% on test.
- ToT vs `relative_minmax/no mask` is much stronger: +10.19% oracle gain on validation and +11.03% on test.
- For 30 deg, ToT vs `relative_minmax/no mask` gives +27.08% oracle gain on validation and +25.52% on test.
- Therefore, the relative candidate's complementarity is larger than ordinary seed diversity and remains visible on validation. This supports moving to selector/gated fusion rather than stopping at oracle diagnostics.

## Stage 2: Rule And Frozen-Logit Selector Fusion

Goal: test whether the oracle choice can be learned from model outputs without retraining ResNet experts.

Add a script, recommended name:

```text
scripts/evaluate_selector_fusion.py
```

Implemented as:

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

- A4b-4a: rule-based selector. No model training; select the rule on validation only.
- A4b-4b: train-logit selector. Train selector on train split; select threshold on validation. This is exploratory because expert train logits can be overconfident.
- A4b-4c: validation-CV selector. Use validation cross-fitting to get out-of-fold selector scores, select threshold on validation, then fit the final selector on full validation.
- Report test metrics once.
- Never use test to choose selector type, threshold, or feature set.

Implemented fusion mode:

```text
hard_switch:
  choose auxiliary prediction if selector_prob >= threshold, otherwise primary
```

`primary_only` is included as a validation-selectable fallback. If the selector does not improve validation performance, the script can select the ToT baseline rather than forcing a harmful switch.

Current selector:

- Torch logistic selector by default, so the server training environment does not need scikit-learn.
- Optional small MLP via `--selector-hidden-dim`, only if the logistic selector underfits and that decision is logged as a new variant.

Current commands:

A4b-4a:

```bash
python scripts/evaluate_selector_fusion.py \
  --selector-mode rule \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_4a_rule_selector_seed42.json \
  --output-summary outputs/a4b_4a_rule_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4a_rule_selector_seed42_by_class.csv
```

A4b-4b:

```bash
python scripts/evaluate_selector_fusion.py \
  --selector-mode trained \
  --selector-fit train \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --selector-target lower-error \
  --selector-epochs 500 \
  --selector-lr 0.01 \
  --selector-weight-decay 0.0001 \
  --output-json outputs/a4b_4b_train_logit_selector_seed42.json \
  --output-summary outputs/a4b_4b_train_logit_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4b_train_logit_selector_seed42_by_class.csv
```

A4b-4c:

```bash
python scripts/evaluate_selector_fusion.py \
  --selector-mode trained \
  --selector-fit val-cv \
  --cv-folds 5 \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --selector-target lower-error \
  --selector-epochs 500 \
  --selector-lr 0.01 \
  --selector-weight-decay 0.0001 \
  --output-json outputs/a4b_4c_val_cv_selector_seed42.json \
  --output-summary outputs/a4b_4c_val_cv_selector_seed42_summary.csv \
  --output-by-class outputs/a4b_4c_val_cv_selector_seed42_by_class.csv
```

Report:

- ToT baseline metrics.
- Auxiliary expert metrics.
- Oracle upper bound.
- Selector fusion metrics.
- Selection rate overall and by class.
- 30 deg precision/recall/F1 and confusion matrix.
- Validation-selected threshold.

Observed A4b-4 results:

| Experiment | Val-selected strategy | Test Acc | vs ToT | Test MAE | Test Macro-F1 | Test selection rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ToT baseline | `primary_only` | 70.48% | 0 | 5.964 deg | 0.646 | 0% |
| A4b-4a rule | `entropy_adv_0p03` | 70.97% | +0.50% | 5.905 deg | 0.658 | 14.51% |
| A4b-4b train selector | `threshold=0.95` | 71.17% | +0.70% | 5.890 deg | 0.654 | 6.96% |
| A4b-4c val-CV selector | `threshold=0.95` | 70.38% | -0.10% | 6.009 deg | 0.644 | 1.39% |
| Oracle | ideal switch | 81.51% | +11.03% | 3.698 deg | 0.784 | 12.43% |

Interpretation:

- Rule-based and train-logit selectors can extract a small real gain from the complementary candidate.
- The stricter validation-CV selector does not beat the ToT baseline, so the most rigorous selector result is negative/neutral.
- There remains a large gap to oracle, so later gated/residual or physical-feature selectors should be framed as attempts to make the complementarity more reliably learnable, not as already proven performance improvements.

## Revised A4b Status And Next Steps

This is the active A4b numbering after the A4b-4 results.

| ID | Name | Status | Role |
| --- | --- | --- | --- |
| A4 | Modality baseline | Done | ToT, ToA, and raw/log1p ToT+ToA comparison. |
| A4b-1 | Relative ToA early fusion | Done | Tests relative ToA image transforms; improves raw early fusion but does not beat ToT. |
| A4b-2 | Fixed late logit fusion | Done | Validation selects alpha=0, so global fixed ToA weight is not reliable. |
| A4b-2.5 | Prediction complementarity | Done | Test-only oracle diagnostic from existing predictions. |
| A4b-3a | ToT-vs-ToT seed oracle control | Done | Shows ordinary seed diversity is much smaller than ToT-vs-relative complementarity. |
| A4b-3b | Val/test oracle check | Done | Confirms ToT vs `relative_minmax/no mask` complementarity on both validation and test. |
| A4b-4a | Rule selector | Done | Formal positive selector baseline; validation chooses `entropy_adv_0p03`. |
| A4b-4b | Train-logit selector | Done | Exploratory result because train logits may be overconfident. |
| A4b-4c | Validation-CV selector | Done | More rigorous learned selector; does not beat ToT. |
| A4b-4d | Switch diagnostics | Next | Explain which selected samples are beneficial, harmful, neutral, or missed. No training. |
| A4b-4e | Three-seed selector confirmation | Implemented config | Rerun only the key candidate for seeds 43/44 and report mean/std with the existing seed42 candidate. |
| A4b-5 | Sample-wise gated late fusion | Implemented script | Compare entropy soft gate, learned scalar gates, class-aware gate, and conservative gate over frozen logits. |
| A4b-6 | Constrained residual interpolation | Implemented script | Keep ToT primary and move logits only partly toward the candidate. |
| A4b-7 | ToA-only relative controls | Later | Isolate whether relative ToA itself carries independent angle information. |
| A4b-8 | ToT image plus ToA scalar features | Later | Physically interpretable ToA feature route for the thesis narrative. |
| A4b-9 | GMU/gated expert model | Optional | End-to-end feature gate only after simpler selector/gate diagnostics. |

Current decision:

- Do not immediately implement GMU, FiLM, MMTM, or ordinary feature concat.
- First explain A4b-4a/4b/4c with A4b-4d switch diagnostics.
- Then try low-cost validation-selected soft-gate/residual variants before any new end-to-end multimodal network.
- If the A4b-4a +0.50% result is presented as a formal positive method, add a three-seed confirmation by training only the key `relative_minmax/no mask` candidate for seeds 43 and 44; the ToT three-seed baseline already exists in `a2_best_3seed`.

## A4b-4d: Switch Diagnostics

Goal: explain why the current selectors remain far below the oracle upper bound.

Implemented script:

```text
scripts/analyze_selector_switches.py
```

Primary target:

```text
A4b-4a selected rule = entropy_adv_0p03
primary expert       = ToT seed42 from a2_best_3seed
candidate expert     = ToT + relative_minmax ToA, no mask, seed42
```

Rule definition from the implemented selector script:

```text
entropy_adv_0p03 selects candidate only when:
  primary prediction and candidate prediction disagree
  candidate_entropy <= primary_entropy - 0.03
```

Required diagnostics:

- switch precision: among switched samples, how often the candidate has lower angle error.
- switch recall: among oracle-beneficial samples, how often the selector switches.
- harmful switch rate: switched samples where candidate angle error is larger.
- neutral switch rate: switched samples where the angle error is unchanged.
- per-class switch statistics, especially 30 deg.
- score distributions for beneficial, harmful, neutral, missed-beneficial, and no-benefit groups.
- per-sample CSV for later plotting or manual inspection.

Interpretation target:

- A4b-4a switches at a similar rate to the oracle, but gains far less. A4b-4d should determine whether the bottleneck is low switch precision, missed 30 deg beneficial samples, or overlapping entropy-score distributions.

Server command:

```bash
cd /root/Timepix

python scripts/analyze_selector_switches.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --rule entropy_adv_0p03 \
  --output-json outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42.json \
  --output-summary outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_summary.csv \
  --output-by-class outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_by_class.csv \
  --output-samples outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_samples.csv \
  --output-distribution outputs/a4b_4d_switch_diagnostics_entropy_adv_0p03_seed42_distribution.csv
```

## A4b-4e: Three-Seed Selector Confirmation

Goal: determine whether the A4b-4a rule-selector gain is stable across seeds.

Training requirement:

- ToT experts: already available in `a2_best_3seed` for seeds 42, 43, and 44.
- Candidate seed42: already available in `a4b_toa_transform_seed42`.
- Candidate seed43/44: train only `ToT + relative_minmax ToA, no mask`.

Candidate config:

```text
configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml
```

Training command:

```bash
cd /root/Timepix

python scripts/run_grid.py \
  --config configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml \
  --continue-on-error
```

Candidate summary:

```bash
python scripts/summarize.py \
  --group a4b_4e_relative_minmax_no_mask_seed43_44 \
  --out outputs/a4b_4e_relative_minmax_no_mask_seed43_44_runs.csv
```

Oracle confirmation across three seeds:

```bash
python scripts/evaluate_oracle_complementarity.py \
  --mode tot-vs-candidate \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
  --seeds 42 43 44 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_4e_oracle_3seed.json \
  --output-summary outputs/a4b_4e_oracle_3seed_summary.csv \
  --output-by-class outputs/a4b_4e_oracle_3seed_by_class.csv
```

Rule selector confirmation, one command per seed:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_selector_fusion.py \
    --selector-mode rule \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_4e_rule_selector_seed${seed}.json" \
    --output-summary "outputs/a4b_4e_rule_selector_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_4e_rule_selector_seed${seed}_by_class.csv"
done
```

Aggregate selector summaries:

```bash
python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_4e_rule_selector_seed42_summary.csv \
    outputs/a4b_4e_rule_selector_seed43_summary.csv \
    outputs/a4b_4e_rule_selector_seed44_summary.csv \
  --out outputs/a4b_4e_rule_selector_mean_std.csv
```

Interpretation rule:

- If validation-selected rule selector improves mean test accuracy/MAE/macro-F1 over `primary_only` with acceptable std, A4b-4a can be reported as a small but stable selector baseline.
- If the mean gain disappears or has high variance, report A4b-4a as a seed42 diagnostic and keep the oracle complementarity as the stronger scientific evidence.

## A4b-5: Sample-Wise Gated Late Fusion

Goal: compare sample-wise gates that dynamically weight ToT and the `relative_minmax/no mask` candidate. This is now treated as a formal selective-fusion comparison rather than a pre-check.

Implemented script:

```text
scripts/evaluate_gated_late_fusion.py
```

Fixed experts:

- Primary: ToT baseline from `a2_best_3seed`.
- Candidate: `ToT + relative_minmax ToA, no mask`.
- ResNet experts are frozen; only the gate is trained or calibrated.

Implemented variants:

| ID | Variant | Fit/selection |
| --- | --- | --- |
| A4b-5a | entropy soft gate, probability fusion | validation grid over threshold and slope |
| A4b-5b | learned scalar gate, probability fusion | train-fit and val-CV |
| A4b-5c | learned scalar gate, logit fusion | train-fit and val-CV |
| A4b-5d | class-aware gate, probability fusion | train-fit and val-CV |
| A4b-5e | conservative scalar gate, probability fusion | train-fit and val-CV, ToT-biased init and mean-gate penalty |

Entropy soft gate:

```text
entropy_adv = primary_entropy - candidate_entropy
g = sigmoid(k * (entropy_adv - threshold))
p_final = (1 - g) * p_tot + g * p_candidate
```

Learned-gate features include:

```text
logits_tot, logits_candidate, logit differences
probabilities_tot, probabilities_candidate, probability differences
top1 confidence, top1-top2 margin, entropy
disagreement flag, predicted angle difference
ToT-predicts-30 flag, candidate-predicts-30 flag
```

Selection rule:

- Select the A4b-5 variant using validation accuracy.
- Tie-break by lower validation MAE, higher validation macro-F1, and lower mean gate.
- Test metrics are reported after selection and are not used to choose gate type, threshold, slope, fit mode, or regularization.

Seed42 command:

```bash
cd /root/Timepix

python scripts/evaluate_gated_late_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_5_gated_late_fusion_seed42.json \
  --output-summary outputs/a4b_5_gated_late_fusion_seed42_summary.csv \
  --output-by-class outputs/a4b_5_gated_late_fusion_seed42_by_class.csv
```

Three-seed command after A4b-4e candidates finish:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_gated_late_fusion.py \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_5_gated_late_fusion_seed${seed}.json" \
    --output-summary "outputs/a4b_5_gated_late_fusion_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_5_gated_late_fusion_seed${seed}_by_class.csv"
done
```

The existing `scripts/aggregate_selector_fusion.py` can aggregate the A4b-5 summaries because it keeps `primary_only`, `candidate_only`, `oracle`, and each seed's `selected_by_val` row:

```bash
python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_5_gated_late_fusion_seed42_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed43_summary.csv \
    outputs/a4b_5_gated_late_fusion_seed44_summary.csv \
  --out outputs/a4b_5_gated_late_fusion_mean_std.csv
```

## A4b-6: Constrained Residual Interpolation

Goal: test whether the candidate should partially correct ToT rather than fully replace it.

Implemented script:

```text
scripts/evaluate_residual_gated_fusion.py
```

Core formula:

```text
logits_final = logits_tot + g * beta * (logits_candidate - logits_tot)
```

Implemented variants:

| ID | Variant | Fit/selection |
| --- | --- | --- |
| A4b-6a | scalar beta residual | validation grid over beta |
| A4b-6b | per-class beta residual | validation grid over class-wise beta vector |
| A4b-6c | learned sample gate + scalar beta | train-fit and val-CV |
| A4b-6d | learned sample gate + per-class beta | train-fit and val-CV |
| A4b-6e | conservative residual | entropy-constrained residual grid plus conservative learned scalar residual |

The candidate is always used as a correction to ToT, not as an equal expert. The
summary records `residual_weight_mean`, high-residual rate, true-30 residual
weight, beneficial high-residual count, and harmful high-residual count.

Seed42 command:

```bash
cd /root/Timepix

python scripts/evaluate_residual_gated_fusion.py \
  --tot-group a2_best_3seed \
  --candidate-group a4b_toa_transform_seed42 \
  --seed 42 \
  --data-root /root/autodl-tmp/Alpha_100 \
  --num-workers 4 \
  --candidate-toa-transform relative_minmax \
  --candidate-add-hit-mask false \
  --output-json outputs/a4b_6_residual_gated_fusion_seed42.json \
  --output-summary outputs/a4b_6_residual_gated_fusion_seed42_summary.csv \
  --output-by-class outputs/a4b_6_residual_gated_fusion_seed42_by_class.csv
```

Three-seed command after A4b-4e candidates finish:

```bash
for seed in 42 43 44; do
  python scripts/evaluate_residual_gated_fusion.py \
    --tot-group a2_best_3seed \
    --candidate-group a4b_toa_transform_seed42 \
    --candidate-group a4b_4e_relative_minmax_no_mask_seed43_44 \
    --seed "$seed" \
    --data-root /root/autodl-tmp/Alpha_100 \
    --num-workers 4 \
    --candidate-toa-transform relative_minmax \
    --candidate-add-hit-mask false \
    --output-json "outputs/a4b_6_residual_gated_fusion_seed${seed}.json" \
    --output-summary "outputs/a4b_6_residual_gated_fusion_seed${seed}_summary.csv" \
    --output-by-class "outputs/a4b_6_residual_gated_fusion_seed${seed}_by_class.csv"
done
```

Aggregate residual summaries:

```bash
python scripts/aggregate_selector_fusion.py \
  --inputs \
    outputs/a4b_6_residual_gated_fusion_seed42_summary.csv \
    outputs/a4b_6_residual_gated_fusion_seed43_summary.csv \
    outputs/a4b_6_residual_gated_fusion_seed44_summary.csv \
  --out outputs/a4b_6_residual_gated_fusion_mean_std.csv
```

## A4b-7: ToA-Only Relative Controls

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

## A4b-8: ToT Image + ToA Scalar Features

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

## A4b-9: End-to-End Gated Expert Fusion

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
