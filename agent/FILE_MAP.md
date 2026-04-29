# File Map

## Repository Root

| Path | Role | Notes |
| --- | --- | --- |
| `.github/` | GitHub metadata | Not inspected deeply. |
| `.vscode/` | Editor settings | Not part of runtime. |
| `.gitignore` | Ignore rules | Excludes local data, cache, and generated outputs. |
| `Data/Alpha_100/` | Local Alpha_100 dataset | Current formal alpha dataset line; angle-first layout with `ToA` and `ToT`. |
| `Data/Alpha_50/` | Local Alpha_50 dataset | Retained as a comparison/history dataset, not the current formal line. |
| `Data/Alpha_Original/` | Local original alpha data | Likely raw or less processed angle folders. |
| `Document/` | Project documents | Not part of training code. |
| `Thesis/` | Thesis materials | Contains `images/`. |
| `representation/` | Presentation/report artifacts | Not inspected deeply. |
| `output/` | Root-level analysis outputs | Used by near-vertical feature analysis and PPT generation. |
| `agent/` | Project handoff docs | Includes architecture notes, experiment guides, and the experiment log. |
| `timepix/` | New experiment-driven training package | First-stage refactor target; old `Program/` is preserved. |
| `configs/` | YAML dataset and experiment configs | Main user-facing way to define experiments. |
| `scripts/` | CLI entry points | `train.py`, `run_grid.py`, `summarize.py`, `aggregate_seeds.py`. |
| `requirements.txt` | Refactored runtime dependencies | Minimal new-system dependencies including Optuna. |
| `generate_presentation.py` | Builds analysis PPT | Uses handcrafted feature CSV/plots for near-vertical analysis. |
| `generate_ppt.py` | Builds a PPT deck | General presentation generation helper. |
| `generate_literature_ppt.py` | Builds literature review PPT | Untracked at inspection time. |
| `generate_midterm_report.py` | Builds midterm report DOCX | Untracked at inspection time. |
| `near_vertical_analysis.py` | Feature analysis for near-vertical angles | Extracts features, plots distributions, RF, PCA/t-SNE-like views, report. |
| `near_vertical_analysis_v2.py` | Expanded near-vertical analysis | More advanced features and reporting. |

## Local Data Paths

These paths are local Windows paths for notebook validation, checkpoint diagnostics, and paper data analysis. Server commands in experiment docs are still written for Linux.

| Dataset | Local Path | Usage |
| --- | --- | --- |
| `Alpha_100` | `D:\Project\Timepix\Data\Alpha_100` | Formal Alpha dataset; use this exact path for training/checkpoint `--data-root`, or parent `D:\Project\Timepix\Data` for analysis scripts. |
| `Proton_C` | `E:\C1Analysis\Proton_C` | Full C/proton dataset for thesis data analysis and near-vertical separability analysis. |
| `Proton_C_7` | `E:\C1Analysis\Proton_C_7` | Seven-class C/proton training dataset for B1 and later Proton/C training experiments. |

Local conda environment:

```powershell
conda activate timepix-local
```

## `Program/`

`Program/` is now legacy/reference code during the refactor. It is still useful
for comparison and for functions not yet migrated, but new experiments should
prefer `configs/` + `scripts/` + `timepix/`.

## `agent/`

| Path | Role | Notes |
| --- | --- | --- |
| `agent/EXPERIMENT_LOG.md` | Experiment log | Human-maintained record of experiment numbering, stage purposes, A/B/D series status, configs, results, commands, and design decisions. |
| `agent/RESEARCH_HANDOFF_5_5_PRO.md` | Research handoff | Best first document for literature-review/thesis-outline agents; summarizes topic, current status, A1-A4, and paper narrative. |
| `agent/A4B_IMPLEMENTATION_PLAN.md` | A4b implementation plan | Staged plan for ToA representation, late logit fusion, ToA scalar features, and later multimodal fusion models. |
| `agent/A4B_SELECTOR_FUSION_PLAN.md` | A4b selector plan | Follow-up plan after A4b-2.5, including ToT seed-control diagnostics with `a2_best_3seed` and selective/gated fusion ideas. |
| `agent/DATA_ANALYSIS_HANDOFF_5_5_PRO.md` | Data analysis handoff | Thesis data-analysis context: raw frames, trajectory extraction, cleaning, final datasets, and near-vertical C/proton distinguishability. |
| `agent/DATA_ANALYSIS_GUIDE.md` | Data analysis script guide | Documents the new `timepix/analysis/` subsystem, output layout, server commands, and thesis wording boundaries. |
| `agent/NEW_SYSTEM_GUIDE.md` | Usage guide | Main how-to for the refactored experiment system. |
| `agent/CODE_CONTEXT.md` | Code context | Practical overview for future code changes. |
| `agent/ARCHITECTURE.md` | Architecture reference | English notes on data/model/training internals. |
| `agent/EXPERIMENT_GROUPS.md` | Experiment grouping guide | Output grouping, metadata, summary commands. |
| `agent/SERVER_TRAINING.md` | Server training guide | Linux server persistence, tmux, resume, AMP notes. |

## `timepix/`

| Path | Role | Notes |
| --- | --- | --- |
| `timepix/config.py` | YAML loading and override helpers | Supports environment placeholders such as `${TIMEPIX_DATA_ROOT:-Data}`. |
| `timepix/config_validation.py` | Config validation | Checks common schema errors before training or grid runs. |
| `timepix/data/` | New dataset subsystem | Modality validation, pairing, splits, normalization, handcrafted features. |
| `timepix/analysis/` | Thesis analysis subsystem | Dataset scanning, event features, statistical distances, ML baselines, plotting, and Markdown reports. |
| `timepix/models/` | New model subsystem | Unified interface for ResNet18 variants, shallow models, DenseNet/EfficientNet/ConvNeXt, ViT-Tiny, A4c dual-stream ToT/ToA fusion models, and A4c warm-started expert gate. |
| `timepix/losses.py` | New loss module | CrossEntropy and EMD in first stage. |
| `timepix/training/` | New training subsystem | Runner, epoch loops, metrics, logging. |
| `timepix/utils/` | Utility helpers | Seed and output path helpers. |

## `configs/`

| Path | Role | Notes |
| --- | --- | --- |
| `configs/datasets/alpha_100.yaml` | Alpha_100 dataset description | Current formal Alpha dataset config; 100x100, supports `ToT` and `ToA`. |
| `configs/datasets/alpha_50.yaml` | Alpha_50 dataset description | Comparison/history Alpha dataset config; 50x50, supports `ToT` and `ToA`. |
| `configs/datasets/alpha.yaml` | Legacy Alpha alias | Kept for compatibility and points to Alpha_100. |
| `configs/datasets/proton_c_7.yaml` | Proton_C_7 dataset description | Current formal Proton/C 7-class dataset; supports only `ToT`. |
| `configs/datasets/proton_c.yaml` | Legacy Proton/C dataset alias | Compatibility entry; points to `Proton_C_7` and should not be used for new training configs. |
| `configs/experiments/alpha_resnet18_tot.yaml` | Alpha ToT baseline | Single-modality baseline. |
| `configs/experiments/alpha_resnet18_tot_toa.yaml` | Alpha ToT+ToA baseline | Multimodal alpha experiment. |
| `configs/experiments/alpha_resnet18_tot_handcrafted_concat.yaml` | Handcrafted concat experiment | Uses ToT `total_energy`. |
| `configs/experiments/alpha_resnet18_tot_handcrafted_gated.yaml` | Handcrafted gated experiment | Uses gated feature fusion. |
| `configs/experiments/proton_resnet18_tot.yaml` | C/proton ToT baseline | ToT-only by dataset constraint. |
| `configs/experiments/b1_proton_c7_resnet18_tot_lr_batch.yaml` | B1-1 Proton_C_7 search | First Proton/C 7-class training search over learning rate and batch size with fixed A1 ResNet18 stem; B1-1 selected `learning_rate=3e-4`, `batch_size=128`. |
| `configs/experiments/b1_proton_c7_resnet18_tot_weight_decay.yaml` | B1-2 Proton_C_7 search | Second Proton/C 7-class training search over `weight_decay=[0, 1e-5, 1e-4]` with B1-1 best `learning_rate=3e-4`, `batch_size=128`. |
| `configs/experiments/b1_proton_resnet18_tot_lr_batch.yaml` | Legacy B1-1 wrapper | Compatibility wrapper that inherits the `Proton_C_7` B1-1 config. |
| `configs/experiments/alpha_tot_a2_best_base.yaml` | A2 best base config | Fixed Alpha ToT CE one-hot setup with A2 best training hyperparameters. |
| `configs/experiments/a1_resnet18_original_baseline.yaml` | A1 original ResNet18 baseline | Alpha ToT, CE, no handcrafted features; original 7x7/stride-2/maxpool stem. |
| `configs/experiments/a1_structure_adaptation.yaml` | A1 ResNet18 structure grid | Alpha ToT, CE, no handcrafted features; compares maxpool, conv1 kernel/stride, and dropout. |
| `configs/experiments/a3_backbone_comparison.yaml` | A3 backbone comparison | Three-seed comparison of ShallowCNN, ShallowResNet, ResNet18, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny, and ViT-Tiny from the A2 best base. |
| `configs/experiments/a3_backbone_comparison_seed42.yaml` | A3 quick comparison | Single-seed-42 shortcut inheriting full A3; 7 backbone runs. |
| `configs/experiments/a4_modality_comparison.yaml` | A4 modality comparison | Three-seed comparison of ToT, ToA, and ToT+ToA with ResNet18 no-maxpool and A2 best training hyperparameters. |
| `configs/experiments/a4_modality_comparison_seed42.yaml` | A4 quick comparison | Single-seed-42 shortcut inheriting full A4; 3 modality runs. |
| `configs/experiments/a4b_toa_transform.yaml` | A4b ToA transform comparison | Three-seed grid over relative ToA transforms and optional hit-mask channel. |
| `configs/experiments/a4b_toa_transform_seed42.yaml` | A4b quick comparison | Single-seed-42 shortcut inheriting full A4b transform grid; 6 runs. |
| `configs/experiments/a4b_4e_relative_minmax_no_mask_seed43_44.yaml` | A4b-4e key candidate config | Trains only the `relative_minmax/no mask` ToT+ToA candidate for seeds 43 and 44. |
| `configs/experiments/a4c_end_to_end_bimodal_fusion.yaml` | A4c full bimodal fusion | Three-seed grid over `dual_stream_concat_aux`, `dual_stream_gmu_aux`, and `toa_conditioned_film` using `ToT + relative_minmax ToA, no mask`. |
| `configs/experiments/a4c_end_to_end_bimodal_fusion_seed42.yaml` | A4c quick comparison | Single-seed-42 shortcut for the first A4c implementation batch. |
| `configs/experiments/a4c_warm_started_expert_gate.yaml` | A4c-4 warm-started expert gate | Three-seed comparison of frozen vs fine-tuned warm-start expert gate using automatically discovered ToT primary and relative-minmax candidate checkpoints. |
| `configs/experiments/a4c_warm_started_expert_gate_seed42.yaml` | A4c-4 quick comparison | Single-seed-42 shortcut for the warm-started expert gate. |
| `configs/experiments/compare_losses.yaml` | Grid config | Compares CE and EMD variants. |
| `configs/experiments/compare_models.yaml` | Grid config | Compares ShallowCNN, ShallowResNet, ResNet18, DenseNet121, EfficientNet-B0, ConvNeXt-Tiny, and ViT-Tiny. |
| `configs/experiments/compare_mixed_precision.yaml` | Grid config | Compares FP32 and CUDA AMP under the current A1 best ResNet18 structure. |
| `configs/experiments/a2_best_alpha_resnet18_tot_3seed.yaml` | Grid config | Re-runs the current A2 best Alpha ToT ResNet18 training setup with three training seeds and fixed `split.seed`. |
| `configs/search/alpha_resnet18_tot_training.yaml` | Optuna search config | Searches representative Alpha ToT ResNet18 training hyperparameters. |
| `configs/search/a2_alpha_resnet18_tot_training.yaml` | A2 Optuna search config | Searches training hyperparameters after fixing the A1 ResNet18 structure. |

## `scripts/`

| Path | Role | Notes |
| --- | --- | --- |
| `scripts/train.py` | Run one experiment | Supports `--data-root`, `--output-root`, `--set`, and `--resume`. |
| `scripts/run_grid.py` | Run grid experiments | Uses a YAML `grid` mapping; supports dry-run, skip-existing, continue-on-error, and manifest CSVs. |
| `scripts/extend_runs.py` | Extend existing runs | Batch-resumes existing runs from `last_checkpoint.pth` to a larger epoch budget, optionally copying them into a new experiment group first. |
| `scripts/search_hparams.py` | Run Optuna search | Uses `configs/search/*.yaml`; writes trials, best params, and best config. |
| `scripts/aggregate_selector_fusion.py` | Aggregate selector fusion | Aggregates A4b selector summary CSVs across seeds, keeping primary/candidate/oracle and validation-selected selector rows. |
| `scripts/evaluate_logit_fusion.py` | Evaluate late logit fusion | Uses trained ToT/ToA single-modality checkpoints, selects alpha on validation, reports test metrics. |
| `scripts/analyze_prediction_complementarity.py` | Analyze prediction complementarity | Reads existing `predictions.csv` files and computes overlap/oracle diagnostics for A4b. |
| `scripts/evaluate_oracle_complementarity.py` | Evaluate oracle complementarity | Reloads checkpoints and recomputes deterministic train/val/test logits for A4b-3a/b ToT-vs-ToT and ToT-vs-candidate oracle controls. |
| `scripts/evaluate_selector_fusion.py` | Evaluate selector fusion | A4b-4 selector entrypoint: rule-based selector, train-logit selector, and validation-CV selector over frozen ToT/candidate logits. |
| `scripts/analyze_selector_switches.py` | Analyze selector switches | A4b-4d diagnostic script that applies a fixed A4b-4 rule and reports switch precision/recall, harmful switches, per-class behavior, per-sample outcomes, and score distributions. |
| `scripts/evaluate_gated_late_fusion.py` | Evaluate gated late fusion | A4b-5 entrypoint for entropy soft gate, learned scalar gate, class-aware gate, and conservative gate over frozen ToT/candidate logits. |
| `scripts/evaluate_residual_gated_fusion.py` | Evaluate residual gated fusion | A4b-6 entrypoint for scalar/per-class beta residuals, learned residual gates, and conservative residual correction over frozen ToT/candidate logits. |
| `scripts/analyze_datasets.py` | Dataset analysis | Generates dataset index, event features, summary tables, representative samples, and dataset-analysis report; defaults to full `Proton_C`, not training-only `Proton_C_7`. |
| `scripts/analyze_resolution_limit.py` | Resolution-limit analysis | Analyzes full `Proton_C` near-vertical ToT separability with effect sizes, ML baselines, pairwise AUC, and figures. |
| `scripts/make_analysis_report.py` | Combined analysis report | Merges dataset and resolution-limit reports into `outputs/analysis_report.md`. |
| `scripts/summarize.py` | Summarize outputs | Supports `--all`, `--group`, and explicit `--root`; writes CSV summaries with `experiment_group`, model hyperparameters, mixed-precision state, and timing fields. |
| `scripts/aggregate_seeds.py` | Aggregate seed repeats | Reads a summary CSV and writes mean/std metrics grouped by stable config fields, including A4c gate/FiLM diagnostic means when present. |

## Legacy `Program/` Files

| Path | Role | Notes |
| --- | --- | --- |
| `Program/README.md` | Cluster script README | File is UTF-8; read with UTF-8 output to avoid mojibake. |
| `Program/requirements.txt` | Python dependencies | Includes torch/torchvision, sklearn, optuna, plotting, notebooks. |
| `Program/Config.py` | Global config class | Mutable class attributes used by all training modules. |
| `Program/main.py` | Main training workflow | Builds data/model/loss, trains, validates, tests, saves outputs. |
| `Program/check_data.py` | Data inspection CLI | Prints angle folders, counts, shape examples, imbalance hints. |
| `Program/sweep.py` | Optuna search | Currently returns a dict instead of a float objective value. |
| `Program/run_ablation.py` | Thesis ablation runner | Defines A-F experiments and shared splits. |
| `Program/rebuild_summary.py` | Rebuild ablation summaries | Reads logs/npz/config and writes summary CSV/JSON. |
| `Program/generate_figures.py` | Plot ablation figures | Uses experiment folders under output root. |
| `Program/generate_report.py` | Generate analysis report | Consumes ablation summary/log/prediction files. |
| `Program/test_losses.py` | Manual loss tests | Needs torch; not pytest style. |
| `Program/test_models.py` | Manual model forward tests | Only covers Resnet18, ShallowCNN, ShallowResNet. |
| `Program/output/` | Default training outputs | `best_model.pth`, logs, plots, stats may be written here. |
| `Program/result/` | Placeholder result docs | Existing md files are empty. |

## `Program/src/`

| Path | Role | Notes |
| --- | --- | --- |
| `Program/src/dataset.py` | Dataset subsystem | Sample collection, modality pairing, splits, normalization, rotation, handcrafted features. |
| `Program/src/trainer.py` | Training epoch function | Supports classification/regression and optional handcrafted features. |
| `Program/src/evaluater.py` | Evaluation function | Typo retained in file name; supports classification/regression. |
| `Program/src/losses.py` | Losses and angle metrics | CE wrapper, EMD loss, regression metrics. |
| `Program/src/logger.py` | CSV experiment logger | Not currently used by `main.py`; dynamic-column behavior is risky. |
| `Program/src/cluster_resnet18.py` | Feature extraction + KMeans clustering | Clusters samples from a class using ResNet18 embeddings, optional handcrafted features. |

## `Program/model/`

| Path | Role | Notes |
| --- | --- | --- |
| `Program/model/utils.py` | Model factory and parameter counting | Imports `model.<name>.Model`. Currently passes `task=` to all models. |
| `Program/model/Resnet18.py` | Primary modified ResNet18 | Supports ToT/ToA channels, handcrafted features, classification/regression. Uses ImageNet pretrained weights by default. |
| `Program/model/ShallowResNet.py` | Shallow residual CNN | Designed for sparse 50x50 tracks; supports handcrafted features and regression. |
| `Program/model/ShallowCNN.py` | Shallow CNN | Supports handcrafted classification, but constructor lacks `task`. |
| `Program/model/Densenet201.py` | DenseNet201 feature model | Constructor lacks `task`; classification only. |
| `Program/model/Resnet18MLP.py` | Older ResNet18 + MLP variant | Similar concept, not integrated as primary. |
| `Program/model/Resnet34.py` | Legacy torchvision ResNet34 | Not adapted to 1/2-channel input and lacks current interface. |
| `Program/model/Resnet50.py` | Legacy torchvision ResNet50 | Not adapted to 1/2-channel input; forward runs backbone twice. |
| `Program/model/Shufflenet.py` | Legacy ShuffleNet | Not adapted to 1/2-channel input and lacks current interface. |
| `Program/model/Efficientnetb0.py` | Legacy EfficientNet | Defines `Efficientnetb0`, not `Model`, so factory cannot load it. |
| `Program/model/CNN.py` | Empty placeholder | Advertised in some comments but unusable. |

## `ProcessProgram/A/`

| Path | Role | Notes |
| --- | --- | --- |
| `README.md` | Preprocessing workflow notes | Best source for this folder's intent. |
| `DataExploration.ipynb` | Data exploration | Stats by particle/angle/modality. |
| `TrajectoryExtraction.ipynb` | Extract tracks from raw matrices | Finds connected components and writes centered ToT/ToA samples. |
| `ToT_to_Image.ipynb` | Render ToT matrices to images | Visual QA. |
| `ToA_to_Image.ipynb` | Render ToA matrices to images | Visual QA. |
| `AngleBatch_ActivatedStats.ipynb` | Activated-pixel statistics by angle | Exploratory statistics. |
| `merge_modalities_by_categories.py` | General dataset merger | Supports modality-first and category-first layouts. |
| `merge_alpha_0_1.py` | Specific class 0/1 alpha merger | Dry-run by default; custom paths assume older AlphaAnalysis layout. |

## `ProcessProgram/C/`

| Path | Role | Notes |
| --- | --- | --- |
| `DetectorData_FilterExport.ipynb` | C/proton data filtering/export | Current C/proton training data should be treated as ToT-only. |
| `TrajectoryExtraction_ToA.ipynb` | ToA track extraction | Legacy/experimental notebook name; current project note says C/proton has no ToA modality. |
| `ToT_Txt_To_Images.ipynb` | ToT text to image rendering | Visual QA. |
| `SampleSubset.ipynb` | Sample subset creation | Notebook workflow. |
| `CropTracks.ipynb` | Track cropping | Notebook workflow. |
| `AngleBatch_ActivatedStats.ipynb` | Activated-pixel statistics by angle | Large notebook. |
| `paper.py` | Older single-particle plotting/export script | Hard-coded external Windows paths; not portable. |
| `原始分析.html` | Exported analysis HTML | Reference artifact. |
