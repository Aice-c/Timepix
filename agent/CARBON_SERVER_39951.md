# 碳离子新服务器部署记录

日期：2026-09-09。仅环境、代码和数据准备；本轮不启动模型训练。

## 隔离与分工

- SSH：`ssh -p 39951 root@connect.westb.seetacloud.com`，使用本机已有密钥。
- 主控在本地隔离工作区`D:/Project/.deploy-worktrees/Timepix-carbon-39951`编辑并推送分支`deploy/carbon-controls-39951`。基线`origin/main`为`0cac19a`；不带入原main的两个论文提交，不提交无关工作区改动。
- Mill仅按主控命令配置环境、克隆已推送代码并校验数据，异常反馈，不自行修改代码或实验配置。
- 仓库已放`/root/autodl-tmp/Timepix`，`/root/Timepix`为指向它的软链接；输出随仓库落数据盘，不占系统盘。

## 初检与环境决策

Ubuntu 22.04.5；RTX 4090 24GB；驱动595.71.05；系统盘30G、数据盘50G初始为空。已安装Python3.12.3、PyTorch2.8.0+cu128、torchvision0.23.0+cu128、NumPy2.3.2，CUDA可用。复用现有环境，不重装驱动或CUDA、不另建conda环境。PyTorch/torchvision/cu128配对与[官方历史安装说明](https://pytorch.org/get-started/previous-versions/)一致。

非交互SSH默认PATH不包含`/root/miniconda3/bin`。执行时显式设置PATH，不能假定shell自动激活conda。补齐tmux、SciPy1.16.1、scikit-learn1.7.1、pytest8.4.2。安装前后保留pip freeze、pip check和GPU检查证据；本轮不安装无关的全量可视化依赖。

## 数据传输

不复用内容未经核实的历史Processed ZIP。`scripts/package_carbon_dataset.py`只读取冻结T7/V6 manifest中的事件路径，逐文件保存SHA256；不改变矩阵，不重新划分或筛选。包内保留`Proton_C/<角度>/ToT/<文件>`，以及`carbon_transfer_manifest.json`。输出已存在或遗留partial时拒绝覆盖。

本地PowerShell打包命令：

```powershell
& D:/Program/Anaconda/envs/timepix-local/python.exe scripts/package_carbon_dataset.py --data-root E:/C1Analysis/Proton_C --manifest D:/Project/Timepix/outputs/carbon_angle_controls_20260909/T7_frame_split.json --manifest D:/Project/Timepix/outputs/carbon_angle_controls_20260909/V6_frame_split.json --output E:/C1Analysis/transfers/carbon_controls_20260909/carbon_controls_events.zip
```

事件包和已生成的`carbon_controls_protocol.zip`独立于Git传输。事件包目标为数据盘，协议包在仓库根解压。解压前核对完整ZIP SHA256并确认目标尚不存在，不覆盖旧数据。解压后执行：

```bash
cd /root/Timepix
export PATH=/root/miniconda3/bin:$PATH
python scripts/package_carbon_dataset.py --verify-extracted /root/autodl-tmp
python -m pytest tests -q
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_carbon_controls.py --data-root /root/autodl-tmp/Proton_C --device cuda --backward --output outputs/carbon_angle_controls_20260909/server_memory_preflight.json
```

最后一条仅合成batch前向/反向，无optimizer step，不是训练；峰值不含Adam状态，需保留余量。服务器验证日志保存到`/root/autodl-tmp/server_setup_20260909`并回传本地`outputs/carbon_server_setup_20260909/`。配置/数据验证完成不代表已获批启动训练。

## 状态

- 本地隔离测试：25 passed。代码提交`6ee007f39319fbb90ae997541dd343a3d9e13a5f`，已用`git ls-remote`确认远端同一SHA。远端main未改变。
- 环境补齐完成：tmux3.2a、SciPy1.16.1、scikit-learn1.7.1、pytest8.4.2；pytest依赖使pluggy1.0.0升至1.6.0。`pip check`通过，CUDA小张量计算通过，torch/torchvision/numpy保持原版本。
- 本地事件包：294364事件，未压缩6643575746字节，ZIP168081490字节，逐成员哈希通过；SHA256=`b61199dcdf3c11f2de77a5b721e901e9deb09b10272c508e7c393fc9b5b2b74c`。
- 协议包：4785812字节，SHA256=`e63df001d50d1fc4d5c7e6c22f3146786c20e18172bdc252e739c5689ca2636c`。包内文件SHA与manifest内部SHA属于不同层次，分别核验。
- 首次Git clone出现HTTP/2 framing错误，HTTP/1.1直连重试超时；主控批准在同一shell中`source /etc/network_turbo`后，HTTP/1.1浅克隆成功。未改全局Git配置，未绕过TLS。
- 服务器仓库SHA与指定提交一致，工作区干净；服务器25项测试通过，`train.py --help`通过。
- 事件包、协议包和两份split JSON文件级SHA核验通过；解压后的294364事件全部逐文件SHA一致，事件范围完全匹配。未重新筛选、改写矩阵或划分。
- GPU预检通过：四组均实际batch128、AMP合成前后向、optimizer_steps=0。峰值allocated为T7-ToT 871473664字节、T7-Mask 871703040字节、V6-Base 871698944字节、V6-HiRes 2039223296字节；HiRes约1.90GiB，不含优化器状态。本轮无需改batch。
- Base尺寸50→49→49→25→13→7→1，HiRes为50→49→49→25→25→25→1。V6两模型参数量均11433350。
- 数据盘解压后约7.2G已用、43G可用；最终检查无训练进程，GPU显存回落至1MiB。
- 日志已回传`D:/Project/Timepix/outputs/carbon_server_setup_20260909/`。预检JSON已回传原协议目录，文件SHA256=`1feec4dca1fd81a3f5f830d2e25f63c2f8ddbc7d0787421282969c882edd84ff`，与服务器一致。
- 训练：未启动。
