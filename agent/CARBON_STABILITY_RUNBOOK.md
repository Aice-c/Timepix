# 碳离子 T7 普通卷积稳定性复核（2026-09-16）

## 状态与关键决策

状态：本地45项定向测试及独立复核通过（另12项专项测试与9个模拟场景通过）；服务器只读预检通过，待Git部署后开始诊断。尚未诊断/新训练。
实验员 Laplace 是唯一服务器执行者；主控本地修改、审核、Git同步，分析员在回传后独立复核。工作区 `D:/Project/.deploy-worktrees/Timepix-carbon-39951`，分支 `codex/carbon-stability-55870`；不动主工作区其他任务。

1. 用户批准先诊断旧 R42，再对普通卷积减小学习率作小规模对照。
2. 用户随后取消 `3e-4` 同期复现：本轮只新增 **S42：lr1e-4、seed42**，无43/44、无额外学习率、无CDC/APDC/V6训练。
3. **用户明确恢复 Val Acc 优先**：Acc最大 → 实际角度argmax MAE最小 → Macro-F1最大 → 更早epoch。保存最佳模型和patience8的改善判断使用同一字典序。这是2026-09-16的新决策。
4. 旧R42、T7/V6及CDC/APDC仍按当时 **MAE → Macro-F1 → 更早epoch** 选模，保留原产物/结论口径，不回写成Acc选模。早期B1/B2/B3原本是Acc优先。
5. 给网页端Astra的交付必须置顶说明第2–4点：新旧比较同时改变学习率与选模/早停，且没有同期原学习率复现，不能把全部差异归因于学习率，也不是严格控制单一变量的结构比较。

## 不训练诊断

来源固定为旧 `20260909_014839_carbon_t7_tot_seed42` 的 best epoch3 和 last epoch11。仅有这两个状态，不能重建epoch6/9。输出到全新 `outputs/carbon_stability_20260916/diagnostic/`，以原子mkdir占用目录，已有目录一律拒绝覆盖（包括空目录）。

- 同一权重、同一事件顺序分别用FP32、AMP float16推理；验证集全10204事件。
- 训练原始视图按类别固定随机抽256个（seed20260916），共1792事件，作为固定train-eval子集。
- BN校准从train另用seed20260917每类抽512个，共3584事件；仅原始方向，不使用val/test，不新增筛选。保存完整样本键、角度、原帧键/路径和用途表。
- 在模型副本中保持dropout及其他模块eval，仅BN临时train，reset running stats、momentum=None，用FP32累计平均各batch的均值/方差；不反向传播、不建优化器、不修改参数。该均值是batch统计平均，不称为精确总体方差估计。完成后回到eval并验证所有非BN状态逐张量不变。
- BN重估的分数只作敏感性诊断，不替代正式模型，不声称证明根因；原始方向校准与训练四旋转视图有区别，必须在解释中保留。
- 源checkpoint/metadata/predictions前后SHA一致性、冻结split与标准化来源哈希、全部推理有限性均核查。
- 脚本结束后实验员先反馈；**只有主控审核后**才运行训练，诊断脚本不会自动启动训练。

## S42 固定配置

`configs/experiments/carbon_t7_stability_lr1e4_seed42.yaml` 继承原carbon公共配置，但不改公共文件。

Carbon_T7，111 MeV/u碳离子、100μm Si、Timepix；历史目录`Proton_C`不改变粒子身份。七类10/20/30/45/50/60/70°，90°垂直。原帧分组split.seed42不变，train/val/test=82162/10204/10572。

ToT 50×50、stem2/1/0、resnet18_no_maxpool、无手工特征、无新裁剪/插值/mask。冻结R42统计mean637.2197424470774/std1189.814612768605，全像素含零背景执行(x−mean)/std。训练固定四90°视图，validation原方向。

随机初始化（不加载旧权重）、CE onehot、Adam **lr1e-4**、wd1e-4、batch128、dropout0.1、cosine eta_min1e-7、25epoch、patience8、AMP float16。test禁推理，按上述Acc-first选模。新增诊断记录不作梯度裁剪、不替换BN、不关AMP，不改变优化器算法。

每batch记录未裁剪且还原scale后的梯度L2范数、有限性、scale前后值和真实optimizer post-step事件。每epoch保存仅模型state_dict和BN摘要及验证指标；best/last仍沿用正式保存机制。约额外1.2GB逐轮模型，启动前要求数据盘至少5GiB空闲。历史R42的14次batch/Adam更新差不能追溯轮次，本轮观测用于补齐该缺口。

不自动恢复或删除半成品，不自动重跑；进程锁由子进程继承，防止重复启动。完整完成项精确匹配配置才允许跳过。单run失败由实验员反馈主控。

部署前审查修复：配置schema增加严格的诊断开关；诊断目录原子占用；训练配置在锁内以独占方式新建；汇总核验批准配置与实际metadata/唯一run一致；每epoch batch编号连续完整且总数吻合metadata；汇总再次计算历史源文件SHA。45项本地定向测试通过，真实CUDA诊断尚待执行。

## 完整服务器命令

服务器 `ssh -p 55870 root@connect.westb.seetacloud.com`；主控完成本地commit/push后同步。服务器不直接编辑源文件。

```bash
cd /root/autodl-tmp/Timepix
source /etc/network_turbo >/dev/null
git -c http.version=HTTP/1.1 fetch origin codex/carbon-stability-55870
git switch -c codex/carbon-stability-55870 FETCH_HEAD
git status --short
git rev-parse HEAD
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/miniconda3/bin/python -m pytest tests/test_stability_diagnostics.py tests/test_carbon_protocol.py tests/test_difference_protocol.py tests/test_difference_queue.py tests/test_difference_convolution.py -q
mkdir -p outputs/carbon_stability_20260916
tmux new-session -d -s carbon_stability_diag_20260916 -c /root/autodl-tmp/Timepix \
  'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; /root/miniconda3/bin/python -u scripts/diagnose_carbon_stability.py --data-root /root/autodl-tmp/Proton_C > outputs/carbon_stability_20260916/diagnostic.console.log 2>&1; rc=$?; printf "%s\n" "$rc" > outputs/carbon_stability_20260916/diagnostic.exitcode'
```

诊断完成并经主控审核后，唯一的一组训练及自动汇总：

```bash
cd /root/autodl-tmp/Timepix
tmux new-session -d -s carbon_stability_train_20260916 -c /root/autodl-tmp/Timepix \
  'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; /root/miniconda3/bin/python -u scripts/run_carbon_stability.py --data-root /root/autodl-tmp/Proton_C --diagnostic-approved > outputs/carbon_stability_20260916/driver.console.log 2>&1; rc=$?; printf "%s\n" "$rc" > outputs/carbon_stability_20260916/training.exitcode'
```

只汇总，不训练：

```bash
cd /root/autodl-tmp/Timepix
/root/miniconda3/bin/python scripts/summarize_carbon_stability.py
```

`--diagnostic-approved` 是主控审核后的放行标记，不授权实验员自己判断放行。分支已存在时先检查HEAD，不重复照抄创建命令。两阶段均无训练墙钟限制，不因指标暂时低而人工结束。Laplace需要持续等待，阶段完成后报告。

## 回传与交付

完整回传含best/last及每轮权重，不套用通用排除checkpoint规则；本地只copy不sync。诊断阶段先只拉summary目录；训练全部成功后再拉完整run与新summary。旧同名文件不覆盖，源端仍在写的文件不提前作为最终结果。

```powershell
rclone copy autodl37655:/root/autodl-tmp/Timepix/outputs/carbon_stability_20260916 D:/Project/Timepix/outputs/carbon_stability_20260916 --sftp-host connect.westb.seetacloud.com --sftp-port 55870 --sftp-user root --sftp-disable-hashcheck --sftp-shell-type unix --transfers 2 --checkers 2 --contimeout 15s --timeout 60s --retries 2 --low-level-retries 2 --ignore-existing
rclone copy autodl37655:/root/autodl-tmp/Timepix/outputs/experiments/carbon_stability_20260916 D:/Project/Timepix/outputs/experiments/carbon_stability_20260916 --sftp-host connect.westb.seetacloud.com --sftp-port 55870 --sftp-user root --sftp-disable-hashcheck --sftp-shell-type unix --transfers 2 --checkers 2 --contimeout 15s --timeout 60s --retries 2 --low-level-retries 2 --ignore-existing
```

逐文件清单/size/SHA验证完成再做独立分析。输出`diagnostic_report.md`、`diagnostic_summary.csv`、`precision_comparison.csv`、每状态NPZ预测和源样本表，训练`validation_comparison.csv`、`epoch_diagnostics.csv`、per-class及混淆、`astra_handoff.md`和轻量`carbon_stability_review.zip`。压缩包不含checkpoint，完整模型另在run保存。

分析要求：选模复算、完整曲线（可记录相邻epoch Acc下降≥10pp的次数，定义仅作诊断，不作为提前终止规则）、BN状态、AMP跳步/梯度有限性、45↔50和60↔70错误，有限观测不能证明根因。新实验n=1不作三seed稳定性声明。新旧lr+选模双重变化的限制必须出现在给Astra的首段，不能只展示有利分数。
