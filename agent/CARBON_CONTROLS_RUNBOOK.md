# 碳离子角度最小对照：T7 / V6

日期：2026-09-09。当前阶段：四组训练、完整结果回传和独立配对核验均完成，test未评估，未追加训练。最终报告为本地`outputs/carbon_angle_controls_20260909/final_review/reuse_and_experiment_report.md`，旧准备报告保留。部署验收见`CARBON_SERVER_39951.md`。

## 实验范围

用户本次确认的背景为111 MeV/u碳离子、100 μm Si、Timepix、ToT单模态，90°垂直。历史`Proton_C*`名称仅作路径兼容；不是另一个质子束或Alpha/粒子类型任务。C1/C2是同一次实验的容量分目录，不当成独立批次。

| 标识 | 角度 | 输入 | 网络 | 执行状态 |
| --- | --- | --- | --- | --- |
| T7-ToT | 10/20/30/45/50/60/70 | 正式ToT表示 | ResNet18 no-maxpool | best3/stop11，回传与一致性核验完成；验证有剧烈波动 |
| T7-Mask | 同T7-ToT | 原保存矩阵x>0的0/1掩膜 | 同T7-ToT | best14/stop22，exit0，回传与配对核验完成 |
| V6-Base | 80/82/84/86/88/90 | 正式ToT表示 | ResNet18 no-maxpool | best12/stop20，完成并回传核验；训练未充分拟合 |
| V6-HiRes | 同V6-Base | 同V6-Base | 只取消layer3/4首块主支和shortcut stride2 | best1/stop9，已回传核验；未见一致的分类改善 |

执行顺序更新：T7-ToT完成后，T7分析与V6两组训练并行进行；Mask已经单独放行，但不能与V6争抢同一GPU。该调整仅改变执行顺序，不改变任何配对协议或训练预算。

正式启动批次沿用上述四个配置，不新增实验变体。实验员须在启动前确认同名tmux会话、训练进程和日志均不存在，日志不覆盖；用显式`/root/miniconda3/bin/python -u`执行。完整命令中的训练pipeline使用`set -o pipefail`，退出后立即记录`rc=$?`到相应`.exitcode`文件。只有退出码0且metrics/checkpoint/validation预测齐全才标记完成。T7-ToT完成后暂停队列，等待主控放行Mask；不要自动运行所有四组。

首轮各1个seed42，不自动扩展三seed、梯度分支或超参搜索。旧B1/B3权重和成绩保留历史参考，不用于新的按原帧隔离基线或初始化。旧TinyCNN不是正式V6基线，不重跑。

## 已完成结果

| 格子 | Val Acc | Val BalAcc | Val Macro-F1 | Val Weighted-F1 | Val MAE / ° | Val P90 / ° | Best/Stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T7-ToT | 88.1321% | 91.2522% | 0.913131 | 0.878776 | 1.115739 | 5 | 3/11 |
| T7-Mask | 64.9451% | 71.9428% | 0.716857 | 0.642310 | 3.714230 | 10 | 14/22 |
| V6-Base | 19.0905% | 19.1702% | 0.138460 | 0.140690 | 4.074730 | 8 | 12/20 |
| V6-HiRes | 18.1736% | 18.1308% | 0.124319 | 0.125603 | 4.000530 | 8 | 1/9 |

T7-ToT−Mask：Acc+23.1870pp、MAE−2.598491°，支持本七分类协议下幅值有额外价值，不证明梯度机制或近垂直收益。V6-HiRes−Base：MAE−0.074200°但Acc−0.9169pp、F1−0.014141，两组MAE均差于常数84°参照2.920182°，未构成一致改善。ToT七分类验证有明显崩落，V6两组在线训练也未充分拟合，不能称不可分或物理分辨上限。所有比较单seed，仅validation。

完整本地run目录均在`outputs/experiments/carbon_angle_controls_20260909/`：

- T7-ToT：`20260909_014839_carbon_t7_tot_seed42`，训练commit `55905ff`。
- T7-Mask：`20260909_034138_carbon_t7_mask_seed42`，训练commit `a8bbf56`。
- V6-Base：`20260909_020948_carbon_v6_base_seed42`，训练commit `748fbd4`。
- V6-HiRes：`20260909_025048_carbon_v6_hires_seed42`，训练commit `748fbd4`。

提交差异仅阶段文档，训练代码及配置相同。两个checkpoint已完整回传且SHA一致；逐事件复算与best选择核验通过。实际Adam更新数依次28234、56472、95637、43037，不用batch数冒充。详见`final_review/run_manifest.csv`及各分析目录。

## 数据与原帧隔离

- 本地事件根：`E:/C1Analysis/Proton_C`；服务器通过`CARBON_DATA_ROOT`指定同版本目录，只枚举配置中的指定类别。
- T7：102,938事件、3,690原帧；V6：191,426事件、1,617原帧。来源核查仅补映射，不重新提取或筛选。
- 样本键例：`80/C_r00000_0085_007.txt`。只移除末尾数字component后缀，得到原帧`C_r00000_0085.txt`，再用角度查原压缩包索引。
- 分组键例：`["80","C1/80/C_r00000_0085.txt"]`。同帧不同后缀同组；不同角度同名前缀不合并；完整路径不同不凭短名称合并。
- 以原帧为单位在每角度内seed42打乱，按帧数约80/10/10分配，事件数比例允许偏差；不是按标签随机拆事件。
- V6及四个参考角度复用旧验证映射；10/20/30°补24,210事件、1,256组索引匹配。索引唯一匹配不是逐事件像素来源全量证明。
- 未找到跨不同原始ZIP成员的已验证物理别名；仅规范化分隔符和角度类型。若后续确认别名，必须显式提供有证据的alias表并新建协议目录，不能静默改变已用划分。
- `<70°`筛选执行链、全量100→50裁剪执行链仍有未闭合部分。第二轮sum<10000规则不能泛化到T7所有角度。本轮不重新筛选。
- 新manifest对样本集合全覆盖、类序、跨split重复、原帧交集和内部SHA256均做检查；manifest缺失直接报错，禁止退回事件随机划分。
- test在本轮保留不评估，但同一套采集数据有既往分析/评估历史，不能称为从未见过的新独立测试数据。按帧隔离也不等于跨独立采集批次验证。

## 冻结协议

CE one-hot，Adam，lr=3e-4、wd=1e-4、dropout=0.1、batch=128、25epoch、patience=8、cosine eta_min=1e-7、AMP float16、随机初始化、train seed42。主指标为Val argmax角度MAE更低，平局Macro-F1更高，再平局保留较早epoch。角度使用真实数值，不按类别索引算误差。

ToT：50×50，不缩放/插值；训练集非零像素拟合全局mean/std，所有像素执行`(x-mean)/std`，因此背景为常量负值。无log1p、无逐事件标准化、无本轮新增截断。统计量写入metadata的`data_info.normalizer_stats`。

Mask：原矩阵`x>0`，保留0/1，跳过ToT归一化。训练几何增强两组相同，沿用每个事件的0/90/180/270°四视图，不是每epoch随机选一个角度。

Base尺寸50→49→49→25→13→7→1；HiRes为50→49→49→25→25→25→1。HiRes同时改变感受野和计算量，不能把改善全部解释为找回梯度。结构不添加dilation，不改变参数量。

## 本地准备与核验命令

PowerShell，已存在结果遇到相同内容直接复用，不同内容拒绝覆盖：

```powershell
& D:/Program/Anaconda/envs/timepix-local/python.exe scripts/prepare_carbon_controls.py --data-root E:/C1Analysis/Proton_C
& D:/Program/Anaconda/envs/timepix-local/python.exe scripts/check_carbon_controls.py --data-root E:/C1Analysis/Proton_C
& D:/Program/Anaconda/envs/timepix-local/python.exe -m pytest tests -q
```

准备输出：`outputs/carbon_angle_controls_20260909/`。两个JSON manifest和分组资料是协议产物，不是训练成绩。服务器使用同一份manifest，没有重新随机划分。代码与配置通过git同步，manifest压缩包作为数据产物另行传输。原准备阶段不启动训练；用户随后单独授权并已完成本轮四组训练。

四个新配置启用`training.require_cuda=true`，在无CUDA环境下正式训练入口会在数据扫描前停止，禁止自动退回笔记本CPU训练。CPU前向检查不调用训练入口。回归测试使用合成数据和替身优化过程，未实际更新模型参数。

协议数据打包命令（不含训练数据、checkpoint或代码，代码仍需git同步）：

```powershell
& D:/Program/Anaconda/envs/timepix-local/python.exe scripts/package_carbon_controls.py
```

生成`outputs/carbon_angle_controls_20260909/carbon_controls_protocol.zip`并逐成员校验SHA256。包内保留`outputs/carbon_angle_controls_20260909/`相对目录；未来在服务器项目根解压，已存在不同manifest时应先核查，禁止覆盖。

## 服务器预检命令

以下记录已通过的预检命令，项目入口`/root/Timepix`指向数据盘仓库，同版本事件根为`/root/autodl-tmp/Proton_C`。非交互SSH默认PATH不含Miniconda，显式指定Python。完整环境证据保留在`CARBON_SERVER_39951.md`。

```bash
cd /root/Timepix
export CARBON_DATA_ROOT=/root/autodl-tmp/Proton_C
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/root/miniconda3/bin/python scripts/check_carbon_controls.py --data-root "$CARBON_DATA_ROOT" --device cuda --backward --output outputs/carbon_angle_controls_20260909/server_memory_preflight.json
```

这只做合成batch前向/反向显存测量，不优化参数；不含Adam状态，因此还需预留优化器状态和运行余量，不能把峰值当完整训练绝对上界。显存不足应反馈主控；两个V6配置统一改为可执行实际batch后重新冻结，不能只改单侧、不能拿梯度累积冒充相同BN batch。

## 完整训练与汇总命令

以下为本批完整执行命令记录，四组已经完成，不要再次运行或覆盖现有日志。实际顺序为T7-ToT、V6-Base、V6-HiRes、T7-Mask；每次启动前实验员检查同名会话、日志和进程，单GPU不并发。未来若批准重复，应先另建实验/日志目录，不能直接复用下列输出名。T7-ToT与V6各有独立问题，不能用T7失败自动取消V6。

```bash
cd /root/Timepix
mkdir -p outputs/carbon_angle_controls_20260909/logs
tmux new-session -d -s carbon-t7-tot 'cd /root/Timepix && export PATH=/root/miniconda3/bin:$PATH CARBON_DATA_ROOT=/root/autodl-tmp/Proton_C OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && { set -o pipefail; /root/miniconda3/bin/python -u scripts/train.py --config configs/experiments/carbon_t7_tot_seed42.yaml 2>&1 | tee outputs/carbon_angle_controls_20260909/logs/t7_tot.log; rc=$?; printf "%s\n" "$rc" > outputs/carbon_angle_controls_20260909/logs/t7_tot.exitcode; exit "$rc"; }'
```

主控已检查T7-ToT的验证表现、机会参照与训练正确性，确认有对照价值后放行Mask，并等待V6结束再单独执行：

```bash
cd /root/Timepix
tmux new-session -d -s carbon-t7-mask 'cd /root/Timepix && export PATH=/root/miniconda3/bin:$PATH CARBON_DATA_ROOT=/root/autodl-tmp/Proton_C OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && { set -o pipefail; /root/miniconda3/bin/python -u scripts/train.py --config configs/experiments/carbon_t7_mask_seed42.yaml 2>&1 | tee outputs/carbon_angle_controls_20260909/logs/t7_mask.log; rc=$?; printf "%s\n" "$rc" > outputs/carbon_angle_controls_20260909/logs/t7_mask.exitcode; exit "$rc"; }'
```

在同一GPU上，等待上一训练结束后再启动下面的V6队列，避免并发争显存。HiRes显存预检必须事先完成：

```bash
cd /root/Timepix
tmux new-session -d -s carbon-v6 'cd /root/Timepix && export PATH=/root/miniconda3/bin:$PATH CARBON_DATA_ROOT=/root/autodl-tmp/Proton_C OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && { set -o pipefail; /root/miniconda3/bin/python -u scripts/train.py --config configs/experiments/carbon_v6_base_seed42.yaml 2>&1 | tee outputs/carbon_angle_controls_20260909/logs/v6_base.log; rc=$?; printf "%s\n" "$rc" > outputs/carbon_angle_controls_20260909/logs/v6_base.exitcode; if [ "$rc" -ne 0 ]; then exit "$rc"; fi; /root/miniconda3/bin/python -u scripts/train.py --config configs/experiments/carbon_v6_hires_seed42.yaml 2>&1 | tee outputs/carbon_angle_controls_20260909/logs/v6_hires.log; rc=$?; printf "%s\n" "$rc" > outputs/carbon_angle_controls_20260909/logs/v6_hires.exitcode; exit "$rc"; }'
```

完整汇总入口仅读取validation，必须显式指定已批准的run目录，禁止多个重跑中自动挑最好。将训练结束打印的四个目录输入以下交互式命令，无需改脚本：

```bash
cd /root/Timepix
read -r -p 'T7-ToT completed run directory: ' T7_TOT
read -r -p 'T7-Mask completed run directory: ' T7_MASK
read -r -p 'V6-Base completed run directory: ' V6_BASE
read -r -p 'V6-HiRes completed run directory: ' V6_HIRES
/root/miniconda3/bin/python scripts/summarize_carbon_controls.py --run "$T7_TOT" --run "$T7_MASK" --run "$V6_BASE" --run "$V6_HIRES" --output "outputs/carbon_angle_controls_20260909/summary_$(date +%Y%m%d_%H%M%S)"
```

若仅完成一个或一对配置，只传对应`--run`，其余保持未完成状态。汇总包含验证指标、同任务成对差值，不以不同任务原始准确率作差。

本次四组结果的可直接运行汇总命令（只读模型结果，创建新时间戳汇总，不启动训练）：

```bash
cd /root/Timepix
/root/miniconda3/bin/python scripts/summarize_carbon_controls.py \
  --run outputs/experiments/carbon_angle_controls_20260909/20260909_014839_carbon_t7_tot_seed42 \
  --run outputs/experiments/carbon_angle_controls_20260909/20260909_034138_carbon_t7_mask_seed42 \
  --run outputs/experiments/carbon_angle_controls_20260909/20260909_020948_carbon_v6_base_seed42 \
  --run outputs/experiments/carbon_angle_controls_20260909/20260909_025048_carbon_v6_hires_seed42 \
  --output "outputs/carbon_angle_controls_20260909/summary_all_$(date +%Y%m%d_%H%M%S)"
```

## 输出与交接

训练run保存`config.yaml`、`metadata.json`、`metrics.json`、`training_log.csv`、`best_model.pth`、`last_checkpoint.pth`。新增`validation_predictions.csv`包含任务/实验/seed/sample_key/原帧键/真实角度/预测角度/各类概率；另输出验证混淆计数、行归一化矩阵、相邻类混淆、训练集确定的多数类与中位角机会参照。`test={}`、`test_evaluated=false`明确表示未评估，不显示为0分。

实验员只执行、监督和反馈，不改代码或配置。结果增量回传本地`outputs/`后由分析员只读分析，主控决定下一步；训练数据和大型checkpoint不混入轻量交接包。单seed结果不宣称显著或稳定。

本次源码核查确认旧TinyCNN使用事件随机划分、逐事件log1p标准化，与本协议不同；有提前达标退出却写early_stopping=off的历史元数据缺陷，但已有四组均跑满200epoch；未发现它被正式训练链路调用，因此不因旧低准确率重复运行TinyCNN。
