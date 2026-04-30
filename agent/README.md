# Timepix 项目文档入口

本文档是后续研究、代码维护和论文写作的入口索引。当前训练主线已经从 legacy `Program/` 迁移到 `timepix/`、`configs/`、`scripts/` 组成的新实验系统；所有新的实验配置、命令和结论应以新系统文档为准。

> 归档说明：旧版 `agent/README.md` 因编码损坏已归档为 `agent/README.old.md`。归档文件仅用于追溯，不作为当前状态依据。

## 推荐阅读顺序

1. `agent/RESEARCH_HANDOFF_5_5_PRO.md`  
   面向研究型模型和论文协作者的总交接文档，包含课题目标、数据集、实验阶段、主要结论和当前开放问题。

2. `agent/EXPERIMENT_LOG.md`  
   当前权威实验日志。记录 A/B/D 系列实验编号、阶段目的、完成状态、关键结果、模型选择口径和决策依据。

3. `configs/README.md`  
   当前权威配置与命令索引。所有新实验配置都应同时给出服务器训练命令和汇总命令。

4. `agent/NEW_SYSTEM_GUIDE.md`  
   新实验系统使用指南，说明数据路径、单次训练、网格实验、汇总、tmux 持久化和恢复训练流程。

5. `agent/CODE_CONTEXT.md`  
   代码工程上下文，说明 `timepix/`、`scripts/`、`configs/` 的职责划分和主要扩展点。

6. `agent/FILE_MAP.md`  
   仓库文件地图，用于快速定位代码、配置、文档、分析脚本和 legacy 目录。

7. `agent/DATA_ANALYSIS_GUIDE.md` 与 `agent/DATA_ANALYSIS_HANDOFF_5_5_PRO.md`  
   论文数据分析链路文档。注意该链路默认使用全量 `Proton_C`，训练主线使用 `Proton_C_7`。

8. `agent/SERVER_TRAINING.md`  
   Linux 服务器持久化训练、`tmux`、断点恢复、AMP 和运行环境说明。

## 当前实验主线

- Alpha 正式训练主线使用 `Alpha_100`，不再使用 `Alpha_50` 作为正式结果线。
- Proton/C 正式训练主线使用 `Proton_C_7`，全量 `Proton_C` 仅用于论文数据分析。
- Alpha A4 系列已经形成多模态结论：A4b-6 是 expert-level 后处理融合系统；A4c-2 `dual_stream_gmu_aux` 是论文主推的端到端 ToT/ToA 多模态架构。
- Alpha A5 已完成，说明低维手工物理标量具备解释性和一定 MAE/F1 辅助价值，但没有稳定提升 test accuracy。
- Alpha A6 正在推进，定位为 Alpha 版 B3，只比较 angle-ordinal loss / label strategy；CE one-hot baseline 复用 A2-best。
- Proton B3 已完成，`CE+ExpectedMAE lambda=0.05` 是当前推荐的 Proton_C_7 有序角度损失。

## 文档维护原则

- 新增或修改实验配置时，必须同步更新 `agent/EXPERIMENT_LOG.md` 和相关指南。
- 每个对比实验必须同时记录训练命令、汇总命令；三 seed 实验还要记录 mean/std 聚合命令。
- 论文和实验结论不得使用 test set 反向选择模型、阈值、特征组或超参数。
- 若某个文档需要大改，应先将原文件归档为 `.old.md`，再重写当前版。
