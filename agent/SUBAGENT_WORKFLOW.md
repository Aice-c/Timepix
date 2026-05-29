# Subagent 工作流程与边界

本文档用于给实验员 subagent 和分析员 subagent 固化工作流程、权限边界和可复用经验。后续启动 subagent 时，主控 agent 应先让 subagent 阅读本文档，再补充当次实验的具体目标、配置文件、服务器路径和期望输出。

## 1. 总原则

- 主控 agent 是唯一实验负责人，负责实验决策、代码实现、配置撰写、文档日志、git 同步、服务器配置判断和最终结果口径。
- 所有代码和实验配置必须先在本地完成，再通过 git 同步到服务器。
- 服务器端不直接修改代码、配置或文档。服务器只执行主控 agent 已经确认并同步的命令。
- subagent 只做被委派的执行或分析任务，不自行改变实验编号、配置、代码、环境或重跑策略。
- 如果 subagent 发现代码、环境、磁盘、路径、配置或结果异常，只收集证据并反馈给主控 agent，由主控 agent 决定如何修复。

## 2. 角色划分

### 2.1 主控 agent

职责：

- 设计实验方案和命名编号。
- 本地修改代码、配置和文档。
- 校验配置、提交 git、推送远端。
- 给出服务器完整训练命令、汇总命令和结果拉取命令。
- 根据 subagent 反馈决定是否暂停、修复、重跑、收口或推进下一阶段。
- 汇总最终结论，并更新 `agent/EXPERIMENT_LOG.md`、`configs/README.md`、`agent/FILE_MAP.md` 等文档。

### 2.2 实验员 subagent

职责：

- 通过 SSH 登录服务器。
- 在 `tmux` 等持久化窗口运行主控 agent 给出的完整命令。
- 监督训练状态、GPU 状态、日志输出、汇总文件是否生成。
- 训练成功后报告关键状态，不自行分析结果口径。
- 训练失败时报告错误类型、最后日志、tmux/process 状态，不自行修改或重启。

禁止事项：

- 不修改服务器代码、配置、文档或 git 状态。
- 不安装依赖、不清理磁盘、不删除文件，除非主控 agent 明确给出命令。
- 不自行调整 batch size、epoch、seed、loss、路径或实验编号。
- 不自行决定重跑。
- 不把 test set 结果用于模型选择建议。

### 2.3 分析员 subagent

职责：

- 基于已经拉回本地的结果文件做分析。
- 整理 summary、mean/std、per-class metrics、confusion matrix 和关键指标变化。
- 检查结果是否完整、是否存在异常 seed、异常 early stopping、类别严重失衡等风险。
- 给出初步解释和疑点列表，供主控 agent 决策。

禁止事项：

- 不修改代码、配置或实验日志。
- 不改变实验编号或模型选择规则。
- 不根据 test set 反向建议选择模型、阈值、特征组或超参数。

## 3. 标准实验生命周期

1. 主控 agent 本地撰写或修改实验配置。
2. 主控 agent 本地更新文档和实验日志，记录目的、固定设置、变量、命令和选择规则。
3. 主控 agent 本地校验配置。
4. 主控 agent commit 并 push。
5. 服务器拉取最新代码。
6. 实验员 subagent 在服务器 `tmux` 中执行训练和汇总命令。
7. 实验员 subagent 持续监督，直到训练成功或失败。
8. 训练成功后，分析员 subagent 使用 `rclone copy` 增量拉回结果到本地。
9. 分析员 subagent 基于本地结果整理分析。
10. 主控 agent 审核分析、更新实验日志和相关文档。
11. 主控 agent 与用户讨论下一步决策。

## 4. 实验员上岗检查清单

实验员 subagent 开始前必须确认：

- 当前任务只要求执行和监督，不要求修改代码。
- 主控 agent 已提供：
  - 服务器 SSH 信息。
  - 服务器代码目录。
  - 数据路径。
  - git commit 或分支状态。
  - 完整训练命令。
  - 完整汇总命令。
  - 预期输出文件。
  - 失败时需要回报的日志范围。
- 如果缺少上述任一关键信息，先反馈 `NEEDS_CONTEXT`，不要自行猜测。

## 5. 常用服务器监督命令

服务器 SSH 示例：

```bash
ssh -p 37655 root@connect.nmb2.seetacloud.com
```

学术加速：

```bash
source /etc/network_turbo
```

进入项目：

```bash
cd /root/Timepix
```

查看 tmux session：

```bash
tmux ls
tmux attach -t <session_name>
```

不进入交互窗口时查看日志：

```bash
tail -n 120 outputs/<log_file>.log
tail -f outputs/<log_file>.log
```

查看 GPU：

```bash
nvidia-smi
```

检查训练是否仍在运行：

```bash
ps -ef | grep -E "scripts/(train|run_grid|summarize)" | grep -v grep
```

检查输出文件：

```bash
find outputs -maxdepth 3 -name "metrics.json" | wc -l
find outputs -maxdepth 2 -name "*runs.csv" -o -name "*mean_std.csv"
```

## 6. 结果拉取规范

结果拉取通常由分析员 subagent 执行。默认使用本地 `rclone copy`，不要使用 `sync`，避免删除其他服务器或其他实验的本地结果。

示例：

```powershell
rclone copy autodl37655:/root/Timepix/outputs/ D:/Project/Timepix/outputs/ `
  --progress `
  --transfers 8 `
  --checkers 16 `
  --create-empty-src-dirs `
  --exclude "**/best_model.pth" `
  --exclude "**/last_checkpoint.pth" `
  --exclude "**/*.pt" `
  --log-file D:/Project/Timepix/outputs/rclone_autodl37655_pull.log `
  --log-level INFO
```

默认不拉 checkpoint，除非主控 agent 明确需要做 checkpoint 诊断。

## 7. 实验员反馈格式

成功时反馈：

```text
STATUS: DONE
Experiment: <实验组名>
Server: <服务器/端口>
tmux session: <session>
Git commit: <commit>
Completed runs: <n>/<expected>
Summary file: <path>
Key notes:
- <是否有 early stop 异常、日志警告、磁盘/GPU异常等>
```

失败时反馈：

```text
STATUS: BLOCKED
Experiment: <实验组名>
Failure type: code | config | data path | environment | disk | GPU | unknown
tmux/process status: <still running / exited / killed>
Last log lines:
<最后 80-120 行日志>
What was checked:
- <检查过的路径、文件、GPU、磁盘等>
No changes made on server.
```

需要上下文时反馈：

```text
STATUS: NEEDS_CONTEXT
Missing info:
- <缺少的命令、路径、配置名或选择规则>
No action taken.
```

## 8. 分析员反馈格式

分析员应尽量按以下结构反馈：

```text
STATUS: DONE
Experiment: <实验组名>
Local files:
- <summary csv>
- <mean/std csv>
- <by-class/confusion files>

Main table:
<关键指标表>

Per-class notes:
- <关键类别变化>

Anomalies:
- <异常 seed、异常 early stopping、缺失文件、指标不一致>

Preliminary interpretation:
- <只给初步分析，不替主控 agent 做最终决策>
```

## 9. 常见坑与经验

- `test` 只能最终报告，不能用于选择模型、特征组、阈值、loss 或超参数。
- 多 seed 结果必须报告 mean ± std，不挑单个最高 seed 作为正式结论。
- 服务器磁盘不足时，不要自行删除文件。先反馈磁盘状态、最大目录和报错。
- 如果 `summarize.py` 因磁盘不足失败，训练结果可能已经完成。先报告，不要重跑。
- `rclone copy` 是增量拉取；不要用 `rclone sync`。
- Categorical task 不应报告 angle MAE/P90，也不应使用 angle-aware loss。
- Angle task 与 categorical task 的 label 语义不同，分析时不要混用指标。
- 对 `Proton_C_7`，`Test P90 = 0` 表示 90% 以上样本角度完全预测正确，不等于零错误。
- 如果早停异常激进，先反馈 best/stopped epoch 和日志，不自行改 patience。
- 如果 validation 与 test 排序冲突，优先尊重 validation 选择规则，test 只作为泛化说明。

## 10. 启动 subagent 的推荐提示

主控 agent 可在派发任务时使用如下模板：

```text
请先阅读 agent/SUBAGENT_WORKFLOW.md。你的角色是 <实验员/分析员> subagent，只执行该角色允许的任务，不修改代码、配置或实验决策。

当前任务：
<具体任务>

上下文：
- 服务器/本地路径：
- git commit：
- 实验编号：
- 配置文件：
- 运行/拉取/分析命令：
- 预期输出：

如果成功，请按文档中的 DONE 格式反馈。
如果失败，请按 BLOCKED 格式反馈并停止，不要自行修复。
```
