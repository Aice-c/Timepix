# Particle Dataset Processing

This folder contains scripts for the new particle-identification dataset.

## Goal

Convert raw detector frames into paired single-particle candidate matrices, then inspect simple cluster features before deciding data-cleaning or clustering rules:

- input: raw `256 x 256` paired `ToT` / `ToA` text frames
- output: bbox-centered `100 x 100` single-particle candidate matrices
- labels: particle type folders, such as `Am`, `Co60`, `Sr`
- Stage 1: only technical extraction; physics cleaning based on cluster features is a later step
- Stage 2a: raw ToT/morphology feature statistics; no transformation, standardization, or clustering yet
- Stage 2b/2c: particle-wise clustering diagnostics and representative crops; useful for exploration, but not final labels
- Stage 2d: visual-first PCA/density inspection; KMeans/GMM colors are reference partitions only

## Paths

Raw data root:

```text
E:\TimepixData\particle\raw
```

Expected raw layout:

```text
E:\TimepixData\particle\raw\
  Am\
    0\
      1_000_ToT.txt
      1_000_ToA.txt
  Co60\
    <acquisition_subdir, e.g. Co60-0deg or Co60-45deg>\
      1_000_ToT.txt
      1_000_ToA.txt
  Sr\
    <acquisition_subdir, e.g. Sr-0deg or Sr-45deg>\
      1_000_ToT.txt
      1_000_ToA.txt
```

Stage-1 output root:

```text
E:\TimepixData\particle\stage1_single_particle_candidates_100x100
```

Stage-2a feature-statistics output root:

```text
E:\TimepixData\particle\stage2_cluster_features_v1
```

Output layout:

```text
stage1_single_particle_candidates_100x100\
  dataset\
    Am\
      ToT\
      ToA\
    Co60\
      ToT\
      ToA\
    Sr\
      ToT\
      ToA\
  manifests\
    extraction_manifest.csv
    rejected_components.csv
    pairing_audit.csv
    summary.json
```

The `dataset` folder is intentionally close to the current training-data layout:

```text
<dataset_root>/<class_name>/<modality>/*.txt
```

## Extraction Logic

`extract_single_particle_candidates.py` does the following:

1. Pair raw files by removing `_ToT` / `_ToA` from file stems.
2. Load paired ToT/ToA matrices.
3. Find 8-connected components using `ToT > 0` by default.
4. For each component, use its bounding box only to decide where it should sit on the output canvas.
5. Split ToA into connected components too; each ToT candidate must overlap exactly one ToA component, and the ToT/ToA component pixel coordinates must match exactly.
6. Copy only the connected-component pixels from ToT and ToA into the centered `100 x 100` canvas; all surrounding pixels, including non-component pixels inside the bbox, are zero-filled.
7. Save paired output files and record statistics in the manifests.

By default, only technical rejections are applied:

- missing ToT/ToA pair
- matrix load or shape failure
- component touches detector edge
- component cannot fit into the target canvas
- ToT/ToA candidate connected regions do not match exactly

Do not tune active-pixel or ToT thresholds in this stage. Those thresholds should be decided after inspecting `extraction_manifest.csv`.

The summary report includes discovered ToT/ToA pairing status counts, total accepted/rejected counts, per-particle counts, and a `reject_reasons` breakdown. Because placement is bbox-centered, there is no centroid-shift overflow rejection path.

## Commands

Quick dry run on a small subset:

```powershell
python ProcessProgram\Particle\extract_single_particle_candidates.py `
  --raw-root E:\TimepixData\particle\raw `
  --output-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --limit-pairs-per-class 5 `
  --dry-run
```

Full extraction:

```powershell
python ProcessProgram\Particle\extract_single_particle_candidates.py `
  --raw-root E:\TimepixData\particle\raw `
  --output-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100
```

Plot stage-1 cleaning diagnostics:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\plot_stage1_cleaning_diagnostics.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100
```

Plot detailed Co60 cleaning diagnostics:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\plot_co60_cleaning_detail.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100
```

Plot angle-stratified cleaning diagnostics:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\plot_stage1_angle_diagnostics.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100
```

The diagnostic plots use two explicit naming conventions:

- `*_feature_histograms_count`: one-dimensional feature distributions. The x-axis is the feature value and the y-axis is candidate count.
- `*_active_total_hexbin_count`: two-dimensional active-pixel / total-ToT count maps. The x/y axes are feature values and color encodes candidate count per bin.

Optional debug variants:

```powershell
# Use union of ToT/ToA active pixels for connected components.
python ProcessProgram\Particle\extract_single_particle_candidates.py --component-mask union

# Keep components touching detector edges for diagnosis.
python ProcessProgram\Particle\extract_single_particle_candidates.py --allow-edge
```

## Stage-2a Cluster Feature Statistics

`stage2_extract_cluster_features.py` computes a compact ToT/morphology feature table from accepted Stage-1 candidates. This stage intentionally does not use ToA features and does not apply log transforms, robust scaling, HDBSCAN, GMM, or any clustering. Its purpose is to inspect raw feature distributions first.

Current Stage-2a features:

- `Npix`: active pixel count of the connected candidate.
- `S_total_ToT`: sum of ToT values over active pixels.
- `Pmax`: maximum active-pixel ToT divided by `S_total_ToT`.
- `Rg`: unweighted radius of gyration of active-pixel coordinates.
- `E_pca`: PCA elongation ratio of the active-pixel geometry, regularized for tiny clusters.
- `Fbox`: active-pixel count divided by the candidate bounding-box area.

Run:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2_extract_cluster_features.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --output-root E:\TimepixData\particle\stage2_cluster_features_v1
```

Outputs:

```text
stage2_cluster_features_v1\
  features_raw.csv
  feature_summary.csv
  feature_correlation_pearson.csv
  feature_correlation_spearman.csv
  feature_notes.md
  figures\
    stage2_raw_feature_histograms_by_particle_count.png/pdf/svg
    stage2_raw_feature_scatter_pairs.png/pdf/svg
    stage2_<particle>_raw_feature_histograms_by_angle_count.png/pdf/svg
```

Stage-2b should be decided only after reviewing these raw distributions. Expected decisions include which variables need `log1p`, which variables should be kept or dropped due to correlation, and which scaler should be used before HDBSCAN/GMM.

## Stage-2b Particle-Wise Transform And Clustering Diagnostics

`stage2b_particlewise_clustering.py` transforms and clusters each particle source independently. It never fits one shared clustering model across `Am`, `Co60`, and `Sr`, because the current goal is to inspect whether each source internally contains separable response structures.

Current transformation policy:

- `Npix`, `S_total_ToT`, `Rg`: `log1p`
- `E_pca`: `log1p(E_pca - 1)`
- `Pmax`, `Fbox`: identity
- scaling: robust per-particle scaling, `(x - median) / IQR`

Current clustering diagnostics:

- `HDBSCAN`: density/noise diagnostic inside each particle.
- `GMM`: model selection for 1, 2, and 3 components inside each particle; saves soft labels and confidence.

Run:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2b_particlewise_clustering.py `
  --stage2a-root E:\TimepixData\particle\stage2_cluster_features_v1 `
  --output-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1
```

Outputs:

```text
stage2b_particlewise_clustering_v1\
  features_transformed_clustered.csv
  transform_parameters.csv
  gmm_model_selection.csv
  particle_cluster_summary.csv
  cluster_feature_summary.csv
  stage2b_notes.json
  figures\
    stage2b_gmm_bic_by_particle.png/pdf
    <particle>_gmm_k2_label_*.png/pdf
    <particle>_gmm_k3_label_*.png/pdf
    <particle>_hdbscan_label_*.png/pdf
```

Interpretation caution:

- GMM labels are morphology groups, not physical truth labels.
- HDBSCAN may fragment highly discrete small-cluster data into many tiny clusters; use it as an anomaly/rare-shape diagnostic before treating it as a cleaning rule.
- Physical interpretation should still be anchored by source composition and representative event images.

## Stage-2c Cluster Representative Images

`stage2c_cluster_representatives.py` samples representative candidates from the particle-wise clustering results and renders centered `10 x 10` ToT crops. This stage is for visual morphology confirmation before assigning physics meaning to any cluster.

Default behavior:

- Reads `features_transformed_clustered.csv` from Stage-2b.
- Uses `gmm_k3_label` and `gmm_k3_confidence`.
- Samples each particle/cluster independently.
- Prefers high-confidence samples with `confidence >= 0.90`.
- Saves a sample manifest so every plotted crop can be traced back to the original Stage-1 candidate.

Run:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2c_cluster_representatives.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --stage2b-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1 `
  --output-root E:\TimepixData\particle\stage2c_cluster_representatives_v1 `
  --label-column gmm_k3_label `
  --confidence-column gmm_k3_confidence `
  --samples-per-cluster 10 `
  --min-confidence 0.90
```

Outputs:

```text
stage2c_cluster_representatives_v1\
  cluster_sample_manifest.csv
  cluster_sample_summary.csv
  stage2c_notes.json
  figures\
    Am_gmm_k3_label_tot_samples_10x10.png/pdf/svg
    Co60_gmm_k3_label_tot_samples_10x10.png/pdf/svg
    Sr_gmm_k3_label_tot_samples_10x10.png/pdf/svg
```

Interpretation caution:

- These images confirm morphology, not physical truth.
- Use the plots to decide which clusters are plausible high-confidence candidates for alpha-like, electron-like, or photon-like response groups.

## Stage-2d Visual Cluster Inspection

`stage2d_visual_cluster_inspection.py` is a visual-first diagnostic stage. It was added after reviewing the HDBSCAN/GMM outputs because the current cleaning goal is to inspect whether each source has visible internal response structures before turning any automatic cluster assignment into a cleaning rule.

This stage still fits plots per particle source independently:

- `Am`
- `Co60`
- `Sr`

It does not mix sources in one PCA or clustering space.

Run:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2d_visual_cluster_inspection.py `
  --stage2b-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1 `
  --output-root E:\TimepixData\particle\stage2d_visual_cluster_inspection_v1 `
  --sample-size 30000
```

Optional draggable 3D HTML views:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage2d_visual_cluster_inspection.py `
  --stage2b-root E:\TimepixData\particle\stage2b_particlewise_clustering_v1 `
  --output-root E:\TimepixData\particle\stage2d_visual_cluster_inspection_v1 `
  --sample-size 30000 `
  --interactive-html `
  --interactive-sample-size 30000
```

The interactive mode requires Plotly in `timepix-local`:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe -m pip install plotly
```

Outputs:

```text
stage2d_visual_cluster_inspection_v1\
  pca_scores_with_kmeans_reference.csv
  pca_loadings.csv
  pca_summary.csv
  stage2d_notes.json
  figures\
    <particle>_density_log1p_Npix_vs_Pmax.png/pdf/svg
    <particle>_density_log1p_Rg_vs_log1p_Epca_minus1.png/pdf/svg
    <particle>_pca_density_PC1_vs_PC2.png/pdf/svg
    <particle>_pca_kmeans_k2_reference.png/pdf/svg
    <particle>_pca_kmeans_k3_reference.png/pdf/svg
    <particle>_pca_gmm_k2_reference.png/pdf/svg
    <particle>_pca_gmm_k3_reference.png/pdf/svg
    <particle>_pca3_gmm_k3_reference.png/pdf/svg
    <particle>_angle_density_log1p_Npix_vs_Pmax.png/pdf/svg
    <particle>_angle_density_log1p_Rg_vs_log1p_Epca_minus1.png/pdf/svg
    <particle>_angle_pca_density_PC1_vs_PC2.png/pdf/svg
    <particle>_angle_pca_kmeans_k3_reference.png/pdf/svg
    <particle>_angle_pca_gmm_k3_reference.png/pdf/svg
  interactive\
    <particle>_pca3_gmm_k3_interactive.html
```

How to read these plots:

- `density_*` plots are density/count maps. The x/y axes are feature values, and the colorbar means candidate count per bin. Brighter or higher color means more candidates in that region.
- `pca_density_PC1_vs_PC2` shows whether the transformed six-feature space has dense islands or only a continuous cloud.
- `pca_kmeans_*_reference` uses KMeans colors only as a visual reference. These colors are not physical labels and must not be used directly as training labels.
- `pca_gmm_*_reference` uses GMM probability-cloud colors on the same PCA plane. These colors are also diagnostic only.
- `pca3_gmm_k3_reference` fits GMM k=3 in `PC1/PC2/PC3` space and plots a static 3D view.
- `interactive/*_pca3_gmm_k3_interactive.html` contains self-contained Plotly 3D views. Open these files in a browser to drag, rotate, zoom, and inspect sample-level hover text.
- `angle_*` plots split each source by angle so we can check whether apparent structures remain inside the same angle or mostly come from angle-dependent morphology.

Current Stage-2d visual interpretation:

- `Am` shows the clearest source-internal structure. It contains a large, dense alpha-like main response region plus small/low-response side populations.
- `Sr` shows a compact-to-extended continuum. The extended side is consistent with electron-like/beta-like response candidates, but the boundary is gradual rather than sharply separated.
- `Co60` also shows a compact-to-extended continuum. The compact region cannot be called pure gamma just from this visualization, because gamma events are recorded through secondary electron responses and compact clusters also appear in Sr.

## Stage-3a Source-Label Conservative Cleaning Audit

`stage3a_source_cleaning_audit.py` implements the current revised cleaning direction:

- use radiation source as the dataset label: `Am`, `Co60`, `Sr`
- do not split `Co60` / `Sr` into automatic beta/gamma clusters
- reject only clearly abnormal candidates before final dataset export
- keep Stage-3a as an audit/proposal stage; it does not write the final cleaned training dataset

Rules:

- `Am`: use a simple active-pixel threshold estimated from the `Am` `Npix` distribution.
- `Co60` / `Sr`: keep the source label and only flag conservative anomalies:
  - low-signal noise-like candidates
  - extreme-large components that may be pileup/multiple events
  - extreme sparse shapes
  - multi-feature outliers

Run:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage3a_source_cleaning_audit.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --stage2a-root E:\TimepixData\particle\stage2_cluster_features_v1 `
  --output-root E:\TimepixData\particle\stage3a_source_cleaning_audit_v2
```

Outputs:

```text
stage3a_source_cleaning_audit_v2\
  source_cleaning_audit.csv
  rejected_candidate_audit.csv
  rejected_sample_manifest.csv
  cleaning_rule_summary.csv
  cleaning_counts_by_source_angle.csv
  stage3a_notes.json
  figures\
    stage3a_keep_reject_feature_histograms.png/pdf/svg
    stage3a_keep_reject_pca_overlay.png/pdf/svg
    Am_stage3a_rejected_samples_10x10.png/pdf/svg
    Co60_stage3a_rejected_samples_10x10.png/pdf/svg
    Sr_stage3a_rejected_samples_10x10.png/pdf/svg
```

Current Stage-3a v2 audit result:

- total candidates: `119667`
- kept candidates: `112750`
- rejected candidates: `6917` (`5.78%`)
- `Am` threshold: `Npix < 21` rejected as low-pixel non-main response
- per-source rejection rates:
  - `Am`: `1979 / 5736 = 34.50%`
  - `Co60`: `4365 / 98079 = 4.45%`
  - `Sr`: `573 / 15852 = 3.61%`

Important v2 change:

- `Co60` / `Sr` reject only `low_signal_noise_like`.
- `extreme_large_component`, `extreme_sparse_shape`, and `multi_feature_outlier` are retained as `review_flags`, because visual inspection showed that large/sparse long tracks can be valid source responses, especially for Sr beta/electron-like events.

## Next Step

After Stage-3a, review the keep/reject feature overlays and rejected sample crops before exporting a final cleaned dataset.

## Stage-3b Source-Label Cleaned Dataset Export

`stage3b_export_cleaned_dataset.py` exports the current source-label cleaned multimodal dataset from the Stage-3a v2 audit. It copies only rows with `recommended_keep == true` and preserves the audit fields, including `review_flags`, in the cleaned manifest.

Run:

```powershell
D:\Program\Anaconda\envs\timepix-local\python.exe `
  ProcessProgram\Particle\stage3b_export_cleaned_dataset.py `
  --stage1-root E:\TimepixData\particle\stage1_single_particle_candidates_100x100 `
  --audit-path E:\TimepixData\particle\stage3a_source_cleaning_audit_v2\source_cleaning_audit.csv `
  --output-root E:\TimepixData\particle\particle_source_label_cleaned_tot_toa_v1
```

Output:

```text
particle_source_label_cleaned_tot_toa_v1\
  dataset\
    Am\ToT
    Am\ToA
    Co60\ToT
    Co60\ToA
    Sr\ToT
    Sr\ToA
  manifests\
    cleaned_manifest.csv
    cleaned_rejected_manifest.csv
    cleaned_counts_by_particle.csv
    cleaned_review_flags.csv
  summary.json
```

Current export counts:

| Source | Exported ToT | Exported ToA | Rejected retained in audit |
| --- | ---: | ---: | ---: |
| Am | 3757 | 3757 | 1979 |
| Co60 | 93714 | 93714 | 4365 |
| Sr | 15279 | 15279 | 573 |
| Total | 112750 | 112750 | 6917 |

Label policy:

- The training labels are radiation-source labels: `Am`, `Co60`, `Sr`.
- They should not be described as event-level pure alpha/beta/gamma truth labels.
- `review_flags` identify retained long/sparse/outlier-shaped candidates for later analysis, not rejection.
