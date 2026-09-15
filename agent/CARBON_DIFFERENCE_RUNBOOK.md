# 碳离子 T7 CDC / 混合 APDC 顺序实验

## 状态与范围（2026-09-16）

附件2026-09-15三seed版已批准。代码提交`47dc2e0`已部署，15项新改动测试及两模型GPU单批检查通过；六项于2026-09-16 02:18:56至05:28:40+08顺序完成，各run及队列退出码均为0。Rawls已完整回传并独立复核：115文件/1,448,620,831bytes/12个checkpoint的大小及SHA均匹配，290检查通过，1919数值对照无实质差异。主控完成结果审核，旧T7/V6及服务器回传原件不覆盖。新分支`codex/carbon-difference-55870`从隔离工作区`b56ad5a`建立，不夹带主工作区论文或数据处理改动。

新克隆服务器：`ssh -p 55870 root@connect.westb.seetacloud.com`。实验员Gibbs唯一执行/监控，Pascal仅做本地只读代码审查，Rawls在训练结束后负责回传与复算；主控负责修正、Git和最终结果口径。本轮不设运行时限，不因一项成绩好坏取消后续项，不自动启动V6。

训练验收：A42/B42/A43/B43/A44/B44的best/stop依次为24/25、17/25、21/25、24/25、20/25、23/25。B42在第25轮恰好满足patience8，正常早停；六项均只执行一次，无中断恢复或重跑，test均未评估。实验员未发现OOM、异常堆栈或非有限日志值；05:29:46检查无训练/队列/tmux残留，GPU 1MiB/0%，数据盘约40.60GiB。

关机约定：用户要求完整完成后通过Computer Use关闭55870实例。浏览器接口连接失败，桌面接口又因无法可靠确认浏览器URL被安全检查停止，未执行关机。用户随后明确选择：完整回传核验后另开新对话尝试关机；本轮不绕过该限制、不调用其他关机方式。

只读预检：Ubuntu22.04.5、RTX4090、驱动595.71.05、Python3.12.3、torch2.8.0+cu128、torchvision0.23.0+cu128、NumPy2.3.2、PyYAML6.0.2、SciPy1.16.1、sklearn1.7.1、pytest8.4.2、tmux3.2a。CUDA可用、GPU空闲、数据盘约41.95GiB。原T7 split与R42 metadata文件SHA逐一匹配本地旧核验值。缺optuna/pandas，但本轮入口不依赖二者，主控决定不安装无关包。非交互PATH未含Python，故命令全部使用绝对路径并显式传数据根。新端口ED25519指纹`SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc`，首次accept-new登记，无冲突。

## 固定设置

| 项目 | 值 |
| --- | --- |
| 数据 | 碳离子111 MeV/u、100μm Si、Timepix；历史目录`Proton_C` |
| 类别 | 10/20/30/45/50/60/70°，90°垂直 |
| 输入 | 原完整50×50 ToT；无新裁剪、mask、log、q或插值 |
| 划分 | 原帧按角度+完整原帧路径分组，split.seed42；82162/10204/10572 |
| 标准化 | 读取旧R42 metadata，mean637.2197424470774、std1189.814612768605；全像素含零背景执行(x−mean)/std |
| 增强 | 每个训练事件四个固定90°视图；validation原视图 |
| 训练 | Adam、lr3e-4、wd1e-4、batch128、dropout0.1、cosine eta_min1e-7、25轮、patience8、AMP float16 |
| 选模 | Val实际角度argmax MAE最小→Macro-F1最大→更早epoch |
| 参数与分辨率 | 11,433,863；50→49→49→25→13→7→1 |
| 变化 | 仅`backbone.model.layer1.{0,1}.conv{1,2}`四层换CDC或混合APDC，theta0.7 |
| 不做 | test推理、旧基线补seed、V6、调参、额外增强/特征/分支、旧链路大回归 |

两模型先完成相同普通ResNet全模型初始化，再保留原参数对象替换运算；不消耗额外随机数，不读取旧权重作初始化。APDC权重用逆置换`[3,0,1,6,4,2,7,8,5]`，bias一次，W_eff可微且不注册为第二套参数。旧PyTorch DataLoader采样/worker种子机制保留；记录初始原始state/RNG哈希，checkpoint新增可兼容weights-only加载的RNG状态。

## 配置与队列

模板：`carbon_t7_cdc_layer1.yaml`、`carbon_t7_apdc_layer1.yaml`，继承`carbon_difference_common.yaml`和旧公共协议。

| 顺序 | ID | 完整配置（configs/experiments下） |
| --- | --- | --- |
| 参考 | R42 | 复用`20260909_014839_carbon_t7_tot_seed42`，不重训 |
| 1 | A42 | carbon_t7_cdc_layer1_theta07_seed42.yaml |
| 2 | B42 | carbon_t7_apdc_layer1_theta07_seed42.yaml |
| 3 | A43 | carbon_t7_cdc_layer1_theta07_seed43.yaml |
| 4 | B43 | carbon_t7_apdc_layer1_theta07_seed43.yaml |
| 5 | A44 | carbon_t7_cdc_layer1_theta07_seed44.yaml |
| 6 | B44 | carbon_t7_apdc_layer1_theta07_seed44.yaml |

六项均为独立进程，不继承上一项参数或优化器。输出`outputs/experiments/carbon_difference_controls_20260915/<timestamp>_<name>`；队列/汇总`outputs/carbon_difference_controls_20260915/`。全解析配置保存在该目录的`resolved_configs/`。

## 完整服务器命令

以下由主控确认环境与提交后交唯一实验员执行。SSH首次端口只允许`accept-new`；冲突不得删除已存密钥。Python使用服务器已核实的绝对路径。

```bash
ssh -p 55870 root@connect.westb.seetacloud.com
cd /root/autodl-tmp/Timepix
source /etc/network_turbo >/dev/null
git -c http.version=HTTP/1.1 fetch origin codex/carbon-difference-55870
git switch -c codex/carbon-difference-55870 FETCH_HEAD
git status --short
git rev-parse HEAD

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/root/miniconda3/bin/python -m pytest tests/test_difference_convolution.py tests/test_difference_protocol.py tests/test_difference_queue.py -q
/root/miniconda3/bin/python scripts/run_carbon_difference_queue.py --data-root /root/autodl-tmp/Proton_C --prepare-only
/root/miniconda3/bin/python scripts/check_carbon_difference.py

tmux new-session -d -s carbon_difference_20260915 -c /root/autodl-tmp/Timepix \
  'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; /root/miniconda3/bin/python -u scripts/run_carbon_difference_queue.py --data-root /root/autodl-tmp/Proton_C > outputs/carbon_difference_controls_20260915/queue.console.log 2>&1; rc=$?; printf "%s\n" "$rc" > outputs/carbon_difference_controls_20260915/queue.exitcode'
```

队列已自动执行训练、汇总、打包。单独重建汇总（不训练）：

```bash
cd /root/autodl-tmp/Timepix
/root/miniconda3/bin/python scripts/summarize_carbon_difference.py --output outputs/carbon_difference_controls_20260915
```

监督：读取queue_manifest、各ID.console.log、training_log、GPU与磁盘。不得在同一队列仍活跃时另启动实验员或手动train。进程级锁阻止重复执行。普通run错误记录后继续下一项；共同GPU/数据不可用或余量小于2GiB则报告阻碍，不擅自清理。外部中断恢复使用相同队列命令：跳过精确匹配且完整完成项，兼容checkpoint恢复原run的模型/优化器/scheduler/RNG/patience与累计轮数；无checkpoint半成品、重复目录或配置冲突必须主控审核，不自动删除从头训练。

本次部署记录：旧克隆的`remote.origin.fetch`仅映射旧分支。指定fetch已成功取得`47dc2e0c4066a5d8d74858baf5e17f7f062a5025`，但`--track origin/新分支`因没有对应引用而失败。主控改为从已核验完整提交建立新本地分支，不修改全局fetch配置，不重新训练或改变协议。上述命令用FETCH_HEAD兼容此类窄克隆；分支已存在时不得再次照抄创建命令，先核对当前HEAD。

## 回传和分析

本地仅`rclone copy`，不mirror删除。先确认远端退出、无训练残留，再按新组增量拉取两目录，完整包含best/last checkpoint；不重复拉原始数据或旧模型。

```powershell
rclone copy autodl37655:/root/autodl-tmp/Timepix/outputs/experiments/carbon_difference_controls_20260915 D:/Project/Timepix/outputs/experiments/carbon_difference_controls_20260915 --sftp-host connect.westb.seetacloud.com --sftp-port 55870 --sftp-user root --sftp-disable-hashcheck --sftp-shell-type unix --transfers 2 --checkers 2 --contimeout 15s --timeout 60s --retries 2 --low-level-retries 2 --ignore-existing
rclone copy autodl37655:/root/autodl-tmp/Timepix/outputs/carbon_difference_controls_20260915 D:/Project/Timepix/outputs/carbon_difference_controls_20260915 --sftp-host connect.westb.seetacloud.com --sftp-port 55870 --sftp-user root --sftp-disable-hashcheck --sftp-shell-type unix --transfers 2 --checkers 2 --contimeout 15s --timeout 60s --retries 2 --low-level-retries 2 --ignore-existing
```

传输超时用于失败重试，不是训练截止。既有同名文件冲突先报告；完整回传核对文件清单/大小与新结果哈希，不要求重哈希旧数据。分析员只读本地结果，可在新增`analysis_review/`写复核材料，不改代码/配置/主日志。

输出逐run六指标、七类Recall/F1与混淆、原验证预测及源帧键；按方法先逐run后mean±sample std(ddof1)，每个seed一票，n<2 std空。配对主比较B_s−A_s，报告均差/差值std/改善seed数；R42仅单seed历史参考，不伪造三条基线。高角度统计保留全部七类预测，不把子集重定义成四分类；平均混淆先逐run按真实类别归一化后求均值。不作bootstrap/显著性检验或集成。

原轻量包`carbon_difference_review.zip`不含数据或checkpoint，模型在独立run目录保留。新增主控报告`final_review/experiment_report.md`与`carbon_difference_final_review.zip`包含独立复核和最终结论，不覆盖服务器原包。CDC三seed MAE0.572978±0.008116°，APDC0.582451±0.012409°；B−A配对+0.009473±0.018942°、B改善1/3。CDC均值略优，不宣称显著胜负；与R42只作历史单seed对照，并保留R42早停/曲线异常及APDC短暂验证尖峰。theta不是信息贡献率，差分参数化不增加测量信息，不等于已证明簇内部物理梯度机制，也不外推V6。本轮到此收口，后续须另行决策。
