# Carbon T7 Difference Controls

Approved scope: 2026-09-15 attachment; implementation starts 2026-09-16.

This supersedes the earlier conditional four-cell execution only for this new
stage. Historical outputs remain immutable. Main controller owns all edits;
one experimenter deploys/runs/monitors, then an analyst reviews local outputs.

## Frozen Protocol

- Group: `carbon_difference_controls_20260915`.
- Reuse R42 (`20260909_014839_carbon_t7_tot_seed42`) predictions/metadata only.
- Six independent processes: A42 CDC, B42 APDC, A43, B43, A44, B44.
- Same T7 frame split, classes 10/20/30/45/50/60/70, counts 82162/10204/10572.
- Load original training normalizer from R42 metadata, never refit.
- Full 50x50 ToT, four fixed training rotation views, original validation view.
- Only layer1's four 3x3 convolutions change, theta=0.7; same raw initialization.
- CE onehot, Adam 3e-4, wd1e-4, batch128, dropout0.1, cosine eta_min1e-7,
  25 epochs, patience8, original AMP float16 and determinism settings.
- Select validation argmax degree MAE, then Macro-F1, then earlier epoch.
- No test inference, new filtering, V6, baseline seeds, hyperparameter search,
  full legacy audit, performance timer stop or CPU training.
- Resume compatible interrupted runs with optimizer/scheduler/early-stop state;
  completed exact matches are reused. Ordinary failed runs do not gate later runs.
- Report A/B paired training-seed differences. R42 is n=1 with blank std.

## Implementation Checklist

- [x] Targeted operator and model tests, preserving raw weights/RNG/state keys.
- [x] Frozen metadata normalizer and reproducible checkpoint resume support.
- [x] Two templates, six concrete configs, sequential resumable queue.
- [x] Prediction-based summaries, per-class and paired comparisons, light ZIP.
- [ ] Local targeted verification and review; isolated Git deployment.
- [ ] Cloned-server preflight, six runs, complete local return and analyst review.
- [ ] Final report and experiment-log closeout.

## Interpretation Limits

CDC and mixed APDC at theta0.7 are invertible kernel reparameterizations, not
additional measurements or expanded convolution function classes. They operate
on early features, not a uniquely isolated physical gradient. Improvements do
not establish a gradient mechanism, significance, or V6 separability. Three
seeds quantify training variability on one fixed split, not independent datasets.

Sources: [CDC](https://arxiv.org/html/2003.04092v1),
[PiDiNet official mixed operators](https://github.com/hellozhuo/pidinet/blob/master/models/ops_theta.py).
