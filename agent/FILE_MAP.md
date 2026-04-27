# File Map

## Repository Root

| Path | Role | Notes |
| --- | --- | --- |
| `.github/` | GitHub metadata | Not inspected deeply. |
| `.vscode/` | Editor settings | Not part of runtime. |
| `.gitignore` | Ignore rules | Excludes local data, cache, and generated outputs. |
| `Data/Alpha_Clean/` | Local cleaned alpha dataset | Angle-first layout with `ToA` and `ToT`. |
| `Data/Alpha_Original/` | Local original alpha data | Likely raw or less processed angle folders. |
| `Document/` | Project documents | Not part of training code. |
| `Thesis/` | Thesis materials | Contains `images/`. |
| `representation/` | Presentation/report artifacts | Not inspected deeply. |
| `output/` | Root-level analysis outputs | Used by near-vertical feature analysis and PPT generation. |
| `timepix/` | New experiment-driven training package | First-stage refactor target; old `Program/` is preserved. |
| `configs/` | YAML dataset and experiment configs | Main user-facing way to define experiments. |
| `scripts/` | CLI entry points | `train.py`, `run_grid.py`, `summarize.py`. |
| `requirements.txt` | Refactored runtime dependencies | Minimal new-system dependencies including Optuna. |
| `generate_presentation.py` | Builds analysis PPT | Uses handcrafted feature CSV/plots for near-vertical analysis. |
| `generate_ppt.py` | Builds a PPT deck | General presentation generation helper. |
| `generate_literature_ppt.py` | Builds literature review PPT | Untracked at inspection time. |
| `generate_midterm_report.py` | Builds midterm report DOCX | Untracked at inspection time. |
| `near_vertical_analysis.py` | Feature analysis for near-vertical angles | Extracts features, plots distributions, RF, PCA/t-SNE-like views, report. |
| `near_vertical_analysis_v2.py` | Expanded near-vertical analysis | More advanced features and reporting. |

## `Program/`

`Program/` is now legacy/reference code during the refactor. It is still useful
for comparison and for functions not yet migrated, but new experiments should
prefer `configs/` + `scripts/` + `timepix/`.

## `timepix/`

| Path | Role | Notes |
| --- | --- | --- |
| `timepix/config.py` | YAML loading and override helpers | Supports environment placeholders such as `${TIMEPIX_DATA_ROOT:-Data}`. |
| `timepix/config_validation.py` | Config validation | Checks common schema errors before training or grid runs. |
| `timepix/data/` | New dataset subsystem | Modality validation, pairing, splits, normalization, handcrafted features. |
| `timepix/models/` | New model subsystem | Unified interface for ResNet18 variants, `shallow_resnet`, and `shallow_cnn`. |
| `timepix/losses.py` | New loss module | CrossEntropy and EMD in first stage. |
| `timepix/training/` | New training subsystem | Runner, epoch loops, metrics, logging. |
| `timepix/utils/` | Utility helpers | Seed and output path helpers. |

## `configs/`

| Path | Role | Notes |
| --- | --- | --- |
| `configs/datasets/alpha_clean.yaml` | Alpha dataset description | Declares alpha supports `ToT` and `ToA`. |
| `configs/datasets/proton_c_tot.yaml` | C/proton dataset description | Declares proton/C supports only `ToT`. |
| `configs/experiments/alpha_resnet18_tot.yaml` | Alpha ToT baseline | Single-modality baseline. |
| `configs/experiments/alpha_resnet18_tot_toa.yaml` | Alpha ToT+ToA baseline | Multimodal alpha experiment. |
| `configs/experiments/alpha_resnet18_tot_handcrafted_concat.yaml` | Handcrafted concat experiment | Uses ToT `total_energy`. |
| `configs/experiments/alpha_resnet18_tot_handcrafted_gated.yaml` | Handcrafted gated experiment | Uses gated feature fusion. |
| `configs/experiments/proton_resnet18_tot.yaml` | C/proton ToT baseline | ToT-only by dataset constraint. |
| `configs/experiments/a1_resnet18_original_baseline.yaml` | A1 original ResNet18 baseline | Alpha ToT, CE, no handcrafted features; original 7x7/stride-2/maxpool stem. |
| `configs/experiments/a1_structure_adaptation.yaml` | A1 ResNet18 structure grid | Alpha ToT, CE, no handcrafted features; compares maxpool, conv1 kernel/stride, and dropout. |
| `configs/experiments/compare_losses.yaml` | Grid config | Compares CE and EMD variants. |
| `configs/experiments/compare_models.yaml` | Grid config | Compares first-stage model set. |
| `configs/experiments/compare_mixed_precision.yaml` | Grid config | Compares FP32 and CUDA AMP under otherwise identical Alpha ToT ResNet18 settings. |
| `configs/search/alpha_resnet18_tot_training.yaml` | Optuna search config | Searches representative Alpha ToT ResNet18 training hyperparameters. |

## `scripts/`

| Path | Role | Notes |
| --- | --- | --- |
| `scripts/train.py` | Run one experiment | Supports `--data-root`, `--output-root`, `--set`, and `--resume`. |
| `scripts/run_grid.py` | Run grid experiments | Uses a YAML `grid` mapping; supports dry-run, skip-existing, continue-on-error, and manifest CSVs. |
| `scripts/search_hparams.py` | Run Optuna search | Uses `configs/search/*.yaml`; writes trials, best params, and best config. |
| `scripts/summarize.py` | Summarize outputs | Supports `--all`, `--group`, and explicit `--root`; writes CSV summaries with `experiment_group`, model hyperparameters, mixed-precision state, and timing fields. |

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
