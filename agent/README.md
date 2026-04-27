# Timepix 项目 Agent 交接文档

这个目录用于帮助后续 AI agent 快速理解毕业论文项目。文档按两个层面组织：

1. **物理意义层面**：这个课题研究什么、数据代表什么、为什么 ToT/ToA 和轨迹形态能反映入射极角，以及 alpha 与 C/质子数据集的模态差异。
2. **代码工程层面**：仓库怎么组织、训练流程怎么跑、哪些文件负责数据、模型、损失函数、实验脚本和结果生成。

建议后续 agent 按下面顺序阅读：

1. `PHYSICS_CONTEXT.md`  
   先理解课题背景、物理目标、数据含义和建模假设。

2. `CODE_CONTEXT.md`  
   再理解代码如何把物理任务转成深度学习分类/回归流程。

3. `ARCHITECTURE.md`  
   英文版训练架构细节，适合查 pipeline 和接口。

4. `FILE_MAP.md`  
   全仓库文件地图，适合定位脚本和模块。

5. `REVIEW_NOTES.md`  
   已发现的 bug、风险和重构建议。

6. `REFACTOR_DIRECTION.md`  
   当前和用户确认过的重构取舍：`run_ablation.py` 的临时性质、实验记录体系、sweep 是否重构、指标优化原则等。

7. `NEW_SYSTEM_GUIDE.md`  
   第一阶段新自动化实验系统的使用说明：如何写配置、如何处理本地/服务器路径、如何跑单实验/网格实验/汇总结果。

8. `EXPERIMENT_GROUPS.md`  
   实验组功能说明：如何用 `experiment_group` 分组保存实验，如何按组或汇总全部结果。

9. `SERVER_TRAINING.md`  
   服务器持久化训练说明：如何用 `tmux/nohup` 防止 VSCode 断开导致训练中断，以及如何从 `last_checkpoint.pth` 恢复。

## 项目一句话概括

本项目用深度学习识别带电粒子入射 Timepix/Timepix3 探测器的极角。输入是粒子事件在探测器像素平面上形成的 ToT 和/或 ToA 二维矩阵，输出是该事件所属的入射角类别。

重要数据约束：

- alpha 粒子数据集有双模态数据：`ToT` 和 `ToA`。
- C/质子数据集只有 `ToT` 模态，没有 `ToA` 模态。
- 因此，多模态 `['ToT', 'ToA']` 只适用于 alpha 数据集；C/质子数据集应使用 `['ToT']`。

## 数据一句话概括

每个样本是一个局部像素阵列文本文件，矩阵元素为空间像素上的激活值；0 表示未激活，非 0 表示该像素被粒子轨迹或能量沉积激活。数据目录用角度文件夹作为类别标签，角度文件夹下按模态分目录；alpha 通常有 `ToT/` 和 `ToA/`，C/质子只有 `ToT/`。

## 代码一句话概括

主训练链路是：

```text
configs/*.yaml
  -> scripts/train.py / scripts/run_grid.py / scripts/search_hparams.py
  -> timepix/data
  -> timepix/models
  -> timepix/losses.py
  -> timepix/training
  -> outputs/experiments
```

旧 `Program/` 目录暂时保留为 legacy 参考，不在第一阶段重构中删除。新训练链路支持 YAML 配置、进度条、checkpoint 恢复、网格实验、Optuna/TPE 超参数搜索、结果汇总和可选 CUDA AMP 混合精度。

## 当前本地状态

截至 2026-04-28 的本地检查：

- 主训练代码已迁移到 `timepix/` + `configs/` + `scripts/`。
- 数据预处理和探索 Notebook/脚本在 `ProcessProgram/`。
- 本地可见的清洗后 alpha 数据在 `Data/Alpha_Clean/`。
- `Data/Alpha_Clean/` 包含 `15`、`30`、`45`、`60` 四个角度。
- 旧 `Program/Config.py` 仍是 legacy 参考；新实验应优先使用 YAML 配置和命令行覆盖数据路径。
- 当前桌面 Python 环境缺少 `torch`，因此本地只能做语法、配置和 dry-run 检查；实际训练和 AMP 性能需要在服务器环境验证。

## 已做验证

最近通过：

```powershell
python -m compileall -q timepix scripts
python scripts\search_hparams.py --config configs\search\alpha_resnet18_tot_training.yaml --dry-run
python scripts\run_grid.py --config configs\experiments\compare_mixed_precision.yaml --dry-run
python scripts\run_grid.py --config configs\experiments\a1_structure_adaptation.yaml --dry-run
python scripts\summarize.py --root outputs\experiments\__missing__ --out outputs\__tmp_summary_amp.csv
```

未能运行：

```powershell
python -c "import torch"
```

原因：当前 Python 环境缺少 `torch`。
