"""
数据探查脚本 — 在服务器端运行，检查 9 类角度数据集。

用法:
    python check_data.py                                    # 默认路径
    python check_data.py --data_dir /root/autodl-tmp/full-angle  # 指定路径
"""

import argparse
import os
import sys
import numpy as np


def check_data(data_dir: str):
    print(f"\n{'='*60}")
    print(f"  数据目录探查: {data_dir}")
    print(f"{'='*60}\n")

    if not os.path.isdir(data_dir):
        print(f"错误: 目录不存在 — {data_dir}")
        return

    # 列出所有子目录
    entries = sorted(os.listdir(data_dir))
    print(f"目录内容 ({len(entries)} 项):")
    for e in entries:
        full = os.path.join(data_dir, e)
        if os.path.isdir(full):
            print(f"  📁 {e}/")
        else:
            print(f"  📄 {e}")

    # 查找角度子目录
    print(f"\n{'─'*40}")
    print("角度子目录分析:")
    print(f"{'─'*40}")

    target_angles = [10, 20, 30, 45, 50, 60, 70, 80, 90]
    angle_dirs = {}
    
    for e in entries:
        full = os.path.join(data_dir, e)
        if not os.path.isdir(full):
            continue
        # 尝试提取角度数字
        angle_str = e.replace('deg', '').replace('°', '').strip()
        try:
            angle = int(angle_str)
            angle_dirs[angle] = full
        except ValueError:
            print(f"  ⚠  无法解析为角度: {e}")

    print(f"\n找到 {len(angle_dirs)} 个角度目录:")
    for angle in sorted(angle_dirs.keys()):
        print(f"  {angle}° → {os.path.basename(angle_dirs[angle])}/")

    # 检查目标角度是否都存在
    missing = [a for a in target_angles if a not in angle_dirs]
    if missing:
        print(f"\n⚠  缺失的目标角度: {missing}")
    else:
        print(f"\n✓ 全部 9 个目标角度均已找到")

    # 统计各角度样本数量
    print(f"\n{'─'*40}")
    print("各角度样本数量统计:")
    print(f"{'─'*40}")
    print(f"{'角度':>6s}  {'样本数':>8s}  {'子目录结构'}")
    print(f"{'─'*6}  {'─'*8}  {'─'*30}")

    total = 0
    counts = {}
    sample_shapes = {}
    
    for angle in sorted(angle_dirs.keys()):
        angle_path = angle_dirs[angle]
        sub_entries = sorted(os.listdir(angle_path))
        
        # 检查子目录结构
        sub_dirs = [s for s in sub_entries if os.path.isdir(os.path.join(angle_path, s))]
        sub_files = [s for s in sub_entries if os.path.isfile(os.path.join(angle_path, s))]
        
        if sub_dirs:
            # 有子目录（如 ToT/, ToA/）
            structure = f"子目录: {', '.join(sub_dirs[:5])}"
            # 统计 ToT 子目录中的文件数
            tot_dir = os.path.join(angle_path, 'ToT')
            if os.path.isdir(tot_dir):
                n_files = len([f for f in os.listdir(tot_dir) if os.path.isfile(os.path.join(tot_dir, f))])
            else:
                # 尝试第一个子目录
                first_sub = os.path.join(angle_path, sub_dirs[0])
                n_files = len([f for f in os.listdir(first_sub) if os.path.isfile(os.path.join(first_sub, f))])
        else:
            # 直接是文件
            structure = f"文件: {len(sub_files)} 个"
            n_files = len(sub_files)

        counts[angle] = n_files
        total += n_files
        print(f"{angle:>5d}°  {n_files:>8d}  {structure}")

        # 读取一个样本查看 shape
        if angle not in sample_shapes:
            try:
                if sub_dirs:
                    first_dir = os.path.join(angle_path, sub_dirs[0])
                    files = [f for f in os.listdir(first_dir) if os.path.isfile(os.path.join(first_dir, f))]
                else:
                    first_dir = angle_path
                    files = sub_files
                if files:
                    sample_path = os.path.join(first_dir, files[0])
                    arr = np.loadtxt(sample_path)
                    sample_shapes[angle] = arr.shape
            except Exception as e:
                sample_shapes[angle] = f"读取失败: {e}"

    print(f"{'─'*6}  {'─'*8}")
    print(f"{'总计':>6s}  {total:>8d}")

    # 数据均衡性分析
    if counts:
        max_count = max(counts.values())
        min_count = min(counts.values())
        ratio = max_count / max(min_count, 1)
        print(f"\n{'─'*40}")
        print("数据均衡性分析:")
        print(f"{'─'*40}")
        print(f"  最多: {max_count} (角度 {max(counts, key=lambda x: counts[x])}°)")
        print(f"  最少: {min_count} (角度 {min(counts, key=lambda x: counts[x])}°)")
        print(f"  最大/最小比: {ratio:.2f}")
        if ratio > 3:
            print(f"  ⚠  数据不均衡（比例 > 3），建议使用加权采样或 class_weight='balanced'")
        else:
            print(f"  ✓ 数据比较均衡（比例 ≤ 3）")

    # 样本 shape
    if sample_shapes:
        print(f"\n{'─'*40}")
        print("样本数据格式:")
        print(f"{'─'*40}")
        for angle in sorted(sample_shapes.keys()):
            print(f"  {angle}°: shape = {sample_shapes[angle]}")

    print(f"\n{'='*60}")
    print("探查完成")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/full-angle')
    args = parser.parse_args()
    check_data(args.data_dir)
