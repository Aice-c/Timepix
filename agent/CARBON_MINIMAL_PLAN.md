# 碳离子最小对照：本地适配计划

日期：2026-09-09。本文件保留最初本地适配阶段记录；下文“未训练/未SSH/未推送”仅描述原阶段。用户随后批准的新服务器部署见`CARBON_SERVER_39951.md`，已完成的四组训练及最新结果状态见`CARBON_CONTROLS_RUNBOOK.md`。

## 范围与决策

本轮依据用户确认的 `carbon_near_vertical_codex_task.md`，仅完成本地代码、配置、原帧划分和验证。历史 `Proton_C*` 路径不改名，不覆盖旧诊断、划分或结果，不提交其他任务的改动。

用户补充：同一原帧提取的粒子事件有不同末尾后缀；不同角度可能有相同文件名前缀。因此样本键必须包含角度，原帧键必须包含角度与经验证的完整原帧来源路径。不能按全局短前缀分组。路径别名只能依据来源证据合并，不能把 C1/C2 当独立批次。

新背景由用户附件提供：111 MeV/u 碳离子、100 μm Si、Timepix、ToT only、90°垂直入射。不是从其他粒子项目推断的参数。

## 实施清单

- [x] 原帧分组：新增 `timepix/data/frame_groups.py`，独立测试跨角度重名前缀、同帧后缀、路径歧义、覆盖与跨split检查；生成独立T7/V6 manifest，旧文件只读。
- [x] 输入与结构：新增可选 `data.input_representation=hit_mask` 和 `model.preserve_late_resolution=true`，测试0/1输入与增强、Base/HiRes前向尺寸和相同参数量；旧默认不变。
- [x] 训练协议：支持显式关闭test、Val MAE/F1字典序选模、保存最佳模型的验证预测及来源键；用合成数据和替身训练步骤做控制流测试，不运行模型优化。
- [x] 本地准备与运行入口：四份单seed配置、来源/分组统计、机会基线、无训练前向检查、汇总脚本与完整服务器命令；Mask训练需先复核T7-ToT，HiRes训练前服务器显存检查。
- [x] 回归测试与独立审核：运行本轮测试及现有测试；核对最终diff，更新实验日志、CODE_CONTEXT、FILE_MAP和配置索引。

完成证据：`python -m pytest tests -q`共23项通过；只读审核员用timepix-local复跑本轮协议测试18项通过，复核发现的问题全部闭合。`local_preflight.json`记录真实无训练前向，`preparation_verification.json`记录固定划分计数与校验值。协议包4,785,812字节，逐成员SHA256核验通过。没有训练、SSH、提交或推送；原有其他任务改动保留。

## 共同协议

T7角度为10/20/30/45/50/60/70；V6为80/82/84/86/88/90。各自按角度内原帧分组0.8/0.1/0.1，split seed42；同任务两配置共享manifest与真实batch。首轮train seed42，随机初始化、CE one-hot、lr3e-4、wd1e-4、dropout0.1、cosine eta_min1e-7、25epoch、patience8、batch128，若HiRes显存不够，两组统一改batch后才启动。主指标Val argmax角度MAE，平局Macro-F1，再平局保留较早epoch。test禁评估。

ToT沿用训练集非零值拟合的全局z-score；零背景也被变换为常量负值。Mask从原矩阵x>0生成，保持0/1，不应用幅值归一化。几何增强沿用每事件四个90°旋转训练视图，不是每epoch随机抽一种旋转。

所有代码/配置由主控本地修改。分析员只读核查；本轮不SSH、不训练、不git推送。
