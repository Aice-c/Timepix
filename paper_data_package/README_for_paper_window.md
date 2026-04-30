# Timepix 论文数据包说明

这个文件夹用于把原始实验结果整理给论文分析窗口。核心规则是：

1. 模型和设置选择只依据 validation 指标。
2. test 指标只用于最终泛化表现报告，不能反向用于挑选最优模型。
3. seed42 screening 和 three-seed verification 分开叙述。
4. `excluded` 和 `legacy` 结果不得进入论文主结果表。

## 文件

- `00_experiment_index.csv`：实验组划分索引。后续所有结果表都应依据这里的 `role`、`stage_type` 和 `include_level` 过滤。
- `01_main_results_summary.csv`：论文主表候选结果，包含 validation/test 的均值和标准差。
- `02_run_level_results.csv`：正式三 seed 和关键 A4b 融合实验的逐 run 指标。
- `03_per_class_results.csv`：主要方法的 validation/test 分类别 precision、recall、F1。
- `04_error_structure.csv`：主结果的 MAE、P90、高角度 F1、远距离错误率等误差结构字段。
- `05_modality_and_gate_diagnostics.csv`：A4b/A4c 多模态、selector、oracle、gate、FiLM 诊断数据。
- `06_handcrafted_feature_results.csv`：A5/B2/A7 手工特征筛选、消融和三 seed 验证数据的合并宽表。
- `06a_handcrafted_classical_metrics.csv`：A5a 手工特征经典模型指标。
- `06b_handcrafted_feature_importance.csv`：A5a 手工特征组重要性。
- `06c_handcrafted_cnn_fusion.csv`：A5/B2 手工特征与 CNN 融合实验，优先给论文分析窗口使用。
- `07_loss_strategy_results.csv`：A2/A6 与 B1/B3 的损失函数基线、筛选和三 seed 验证数据。
- `08_excluded_or_diagnostic_runs.csv`：排除、诊断或仅用于讨论的实验组。
- `09_missing_value_audit.csv`：空单元格审计表，区分 `not_applicable`、`source_missing`、`pending` 等情况。
- `10_final_decision_summary.csv`：面向论文分析窗口的最终决策摘要表，只总结关键采纳/否定结论，不替代原始指标表。
- `build_tables.py`：从现有 `outputs/` 结果重新生成上述数据表的脚本。
- `timepix_paper_data_package.xlsx` / `timepix_paper_data_package_updated.xlsx`：把上述 CSV 合并到一个 Excel 工作簿中，便于人工筛选和复制表格；如果前者被 Excel 占用，脚本会写入后者。
- `build_workbook.mjs`：从 CSV 重新生成 Excel 工作簿的脚本。

## role 说明

| role | 含义 |
| --- | --- |
| `dataset_context` | 数据集结构、样本形态、类别分布等背景信息。 |
| `baseline` | 正式基线，通常是三 seed 验证。 |
| `main_result` | 可进入论文主表的正式实验结果。 |
| `ablation` | 可进入正文或附录的消融实验。 |
| `screening` | 用于选择候选设置的筛选实验，通常不能作为最终结论。 |
| `diagnostic` | 机制解释或失败原因分析，不用于最终模型排名。 |
| `excluded` | 不进入论文分析，只记录排除原因。 |
| `planned` | 尚未完成的未来实验。 |

## 当前主线建议

### Alpha

- 正式 baseline：`A2-best`
- 多模态主线：`A4b-5`、`A4b-6`、`A4c-1-3`
- 手工物理特征消融：`A5d`
- A6 已完成：A6b 证明 `CE+EMD lambda=0.02` 不稳定且弱于 A2 CE baseline，Alpha 后续保持 CE one-hot。
- A7 已完成：`GMU_aux + main_5feat gated` 不进入最终模型；Alpha 最终端到端多模态主模型为 `dual_stream_gmu_aux + ToT/relative_minmax ToA + CE one-hot + no handcrafted`。

### Proton_C_7

- 正式 baseline：`B1-best`
- 手工特征诊断：`B2a`、`B2b`
- 损失函数主线：`B3a` 筛选，`B3b-main` 三 seed 验证
- 当前推荐 Proton 结果：`B3b-main CE+ExpectedMAE lambda=0.05`

## 需要特别避免的误用

- 不要使用 `B1-best-old` 作为 Proton baseline；它是 patience=5 的旧早停版本。
- 不要使用 `A3-legacy` 或 `A4-legacy` 的 `_50` 后缀结果作为正式分析。
- 不要把 `A5d` 中 test accuracy 更好的设置反向称为最优；A5d 内部选择必须看 validation。
- 不要把 `A6a` 的 seed42 tie-break 候选写成最终正结果；A6b 三 seed 验证后不采用 CE+EMD。
- 不要把 `A7` 的 `main_5feat` 写进 Alpha 最终 GMU 主模型；A7 是负/诊断性组件确认。
- 不要把 A7 的 `gate_tot/gate_toa` 解释为手工特征 gated fusion 的权重；它们是 GMU 图像分支 ToT/ToA gate。
- 不要把 oracle 或 complementarity 诊断写成可部署模型表现。
- 不要把 `run_ablation` 临时脚本写进正式实验体系。

## 使用注意

- `01_main_results_summary.csv` 中的 selection 说明只描述实验选择依据，不允许用 test 结果反向改写“最优”结论。
- `03_per_class_results.csv` 只整理 validation/test，不放 train，避免论文分析窗口混入训练集表现。
- A5d 的 `main_5feat` 已通过 Windows 长路径方式读取 `metrics.json`，validation macro-F1 和 per-class 细节已补齐。
- A4b post-hoc fusion 没有完整逐样本预测或混淆矩阵，因此 far-error 仍为空；这是 `source_missing`，不是模型没有跑。
- `06_handcrafted_feature_results.csv` 是合并宽表，空格较多；论文分析优先使用 `06a`、`06b`、`06c` 三张拆分表。
- `10_final_decision_summary.csv` 只用于快速理解实验结论；写论文表格时仍应回到 `01`、`03`、`05`、`06c`、`07` 查看原始指标。
- 重新生成数据包可运行：

```bash
python paper_data_package/build_tables.py
node paper_data_package/build_workbook.mjs
```
