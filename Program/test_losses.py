"""
Wasserstein (EMD) 损失函数单元测试。

验证内容：
1. EMD 损失基本计算正确性
2. 完美预测时损失接近 0
3. 相邻类别预测损失 < 远处类别预测损失（有序性）
4. 高斯软标签编码正确性
5. CrossEntropyLossWrapper 兼容性
6. build_loss_function 工厂函数
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.losses import (
    EarthMoverDistanceLoss,
    CrossEntropyLossWrapper,
    build_loss_function,
    compute_angle_mae,
)


def test_emd_loss_basic():
    """验证 1：EMD 损失基本计算"""
    print("=" * 60)
    print("测试 1：EMD 损失基本计算")
    print("=" * 60)

    num_classes = 46  # 角度 0,2,4,...,90 共 46 类
    angle_values = [float(i * 2) for i in range(num_classes)]

    logits = torch.randn(4, num_classes)
    target_onehot = torch.zeros(4, num_classes)
    for i in range(4):
        target_onehot[i, i * 10] = 1.0

    emd_loss_fn = EarthMoverDistanceLoss(
        num_classes=num_classes, p=2,
        label_encoding='onehot', angle_values=angle_values,
    )
    loss = emd_loss_fn(logits, target_onehot)
    print(f"  EMD loss (p=2, random logits): {loss.item():.4f}")
    assert loss.item() > 0, "随机预测的损失应 > 0"
    print("  ✓ 随机预测损失 > 0")


def test_emd_loss_perfect():
    """验证 2：完美预测时损失接近 0"""
    print("\n" + "=" * 60)
    print("测试 2：完美预测时损失接近 0")
    print("=" * 60)

    num_classes = 46
    angle_values = [float(i * 2) for i in range(num_classes)]

    target_onehot = torch.zeros(4, num_classes)
    for i in range(4):
        target_onehot[i, i * 10] = 1.0

    # 构造几乎完美的 logits（softmax 后接近 one-hot）
    perfect_logits = torch.zeros(4, num_classes)
    for i in range(4):
        perfect_logits[i, i * 10] = 100.0

    emd_loss_fn = EarthMoverDistanceLoss(
        num_classes=num_classes, p=2,
        label_encoding='onehot', angle_values=angle_values,
    )
    loss = emd_loss_fn(perfect_logits, target_onehot)
    print(f"  EMD loss (perfect prediction): {loss.item():.6f}")
    assert loss.item() < 1e-4, f"完美预测的损失应接近 0，实际为 {loss.item()}"
    print("  ✓ 完美预测损失接近 0")


def test_emd_loss_ordinal():
    """验证 3：有序性——相邻类别损失 < 远处类别损失"""
    print("\n" + "=" * 60)
    print("测试 3：有序性检验（近 vs 远）")
    print("=" * 60)

    num_classes = 46
    angle_values = [float(i * 2) for i in range(num_classes)]

    emd_loss_fn = EarthMoverDistanceLoss(
        num_classes=num_classes, p=2,
        label_encoding='onehot', angle_values=angle_values,
    )

    # 真实类别在第 20 类（40°）
    target_single = torch.zeros(1, num_classes)
    target_single[0, 20] = 1.0

    # 预测偏到第 21 类（42°，相邻）
    logits_near = torch.full((1, num_classes), -100.0)
    logits_near[0, 21] = 100.0

    # 预测偏到第 40 类（80°，远处）
    logits_far = torch.full((1, num_classes), -100.0)
    logits_far[0, 40] = 100.0

    loss_near = emd_loss_fn(logits_near, target_single)
    loss_far = emd_loss_fn(logits_far, target_single)

    print(f"  EMD loss (near miss, 差2°): {loss_near.item():.4f}")
    print(f"  EMD loss (far miss, 差40°):  {loss_far.item():.4f}")
    print(f"  比值 far/near: {loss_far.item() / loss_near.item():.1f}x")
    assert loss_far.item() > loss_near.item(), "远处预测的损失应大于近处预测"
    print("  ✓ 远处预测损失 > 近处预测损失")


def test_emd_with_integer_labels():
    """验证 4：EMD 损失接受整数标签"""
    print("\n" + "=" * 60)
    print("测试 4：EMD 损失接受整数标签")
    print("=" * 60)

    num_classes = 4
    angle_values = [15.0, 30.0, 45.0, 60.0]

    emd_loss_fn = EarthMoverDistanceLoss(
        num_classes=num_classes, p=2,
        label_encoding='onehot', angle_values=angle_values,
    )

    logits = torch.randn(8, num_classes)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    loss = emd_loss_fn(logits, labels)
    print(f"  EMD loss (integer labels): {loss.item():.4f}")
    assert loss.item() > 0, "损失应 > 0"
    print("  ✓ 整数标签正常工作")


def test_gaussian_labels():
    """验证 5：高斯软标签编码"""
    print("\n" + "=" * 60)
    print("测试 5：高斯软标签编码")
    print("=" * 60)

    num_classes = 46
    angle_values = [float(i * 2) for i in range(num_classes)]

    emd_loss_fn = EarthMoverDistanceLoss(
        num_classes=num_classes, p=2,
        label_encoding='gaussian', angle_values=angle_values,
        gaussian_sigma=2.0,
    )

    # 真实角度 = 10°（对应第 5 个类，索引 5）
    targets = torch.tensor([5])
    encoded = emd_loss_fn._encode_targets(targets)

    print(f"  编码后的分布 shape: {encoded.shape}")
    print(f"  分布总和: {encoded.sum().item():.6f}")
    print(f"  最大值位置: 第 {encoded.argmax().item()} 类 ({angle_values[encoded.argmax().item()]}°)")
    print(f"  最大概率: {encoded.max().item():.6f}")

    # 显示 10° 附近的分布
    print(f"  8° (idx=4): {encoded[0, 4].item():.6f}")
    print(f"  10° (idx=5): {encoded[0, 5].item():.6f}")
    print(f"  12° (idx=6): {encoded[0, 6].item():.6f}")
    print(f"  60° (idx=30): {encoded[0, 30].item():.10f}")
    print(f"  90° (idx=45): {encoded[0, 45].item():.10f}")

    assert abs(encoded.sum().item() - 1.0) < 1e-5, "分布总和应为 1"
    assert encoded.argmax().item() == 5, f"最大概率位置应为第5类（10°），实际为 {encoded.argmax().item()}"
    assert encoded[0, 4].item() > encoded[0, 30].item(), "相邻8°概率应大于远处60°"
    print("  ✓ 高斯软标签编码正确")


def test_emd_gaussian_combination():
    """验证 6：EMD + gaussian 组合"""
    print("\n" + "=" * 60)
    print("测试 6：EMD + gaussian 推荐方案")
    print("=" * 60)

    num_classes = 4
    angle_values = [15.0, 30.0, 45.0, 60.0]

    emd_loss_fn = EarthMoverDistanceLoss(
        num_classes=num_classes, p=2,
        label_encoding='gaussian', angle_values=angle_values,
        gaussian_sigma=5.0,
    )

    logits = torch.randn(8, num_classes)
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    loss = emd_loss_fn(logits, labels)
    print(f"  EMD loss (emd+gaussian): {loss.item():.4f}")

    # 验证梯度存在
    logits.requires_grad = True
    loss2 = emd_loss_fn(logits, labels)
    loss2.backward()
    print(f"  梯度范数: {logits.grad.norm().item():.6f}")
    assert logits.grad is not None, "应有梯度"
    assert logits.grad.norm().item() > 0, "梯度应非零"
    print("  ✓ EMD+gaussian 可正常计算损失和反向传播")


def test_ce_wrapper():
    """验证 7：CrossEntropyLossWrapper"""
    print("\n" + "=" * 60)
    print("测试 7：CrossEntropyLossWrapper 兼容性")
    print("=" * 60)

    num_classes = 4
    ce_wrapper = CrossEntropyLossWrapper()
    ce_original = torch.nn.CrossEntropyLoss()

    logits = torch.randn(8, num_classes)
    labels_int = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])

    # 整数标签
    loss_wrapper = ce_wrapper(logits, labels_int)
    loss_original = ce_original(logits, labels_int)
    print(f"  CE wrapper (int labels): {loss_wrapper.item():.6f}")
    print(f"  CE original (int labels): {loss_original.item():.6f}")
    assert abs(loss_wrapper.item() - loss_original.item()) < 1e-5
    print("  ✓ 整数标签结果一致")

    # 分布标签（应取 argmax 后与整数标签结果一致）
    labels_dist = torch.zeros(8, num_classes)
    for i in range(8):
        labels_dist[i, labels_int[i]] = 1.0
    loss_dist = ce_wrapper(logits, labels_dist)
    print(f"  CE wrapper (one-hot dist): {loss_dist.item():.6f}")
    assert abs(loss_dist.item() - loss_original.item()) < 1e-5
    print("  ✓ One-hot 分布标签结果一致")


def test_build_loss_function():
    """验证 8：build_loss_function 工厂函数"""
    print("\n" + "=" * 60)
    print("测试 8：build_loss_function 工厂函数")
    print("=" * 60)

    label_map = {0: '15', 1: '30', 2: '45', 3: '60'}

    # 模拟 config 对象
    class MockConfig:
        loss_type = 'cross_entropy'
        label_encoding = 'onehot'
        weight = [2.8, 7.1, 6.5, 2.8]
        emd_p = 2
        gaussian_sigma = 2.0

    cfg = MockConfig()
    criterion_ce = build_loss_function(cfg, num_classes=4, label_map=label_map)
    print(f"  CE criterion type: {type(criterion_ce).__name__}")
    assert isinstance(criterion_ce, CrossEntropyLossWrapper)
    print("  ✓ cross_entropy 配置正确构建")

    cfg.loss_type = 'emd'
    cfg.label_encoding = 'gaussian'
    criterion_emd = build_loss_function(cfg, num_classes=4, label_map=label_map)
    print(f"  EMD criterion type: {type(criterion_emd).__name__}")
    assert isinstance(criterion_emd, EarthMoverDistanceLoss)
    print("  ✓ emd 配置正确构建")


def test_compute_angle_mae():
    """验证 9：MAE 计算"""
    print("\n" + "=" * 60)
    print("测试 9：角度 MAE 计算")
    print("=" * 60)

    num_classes = 4
    angle_values = torch.tensor([15.0, 30.0, 45.0, 60.0])

    # 完美预测
    logits_perfect = torch.zeros(4, num_classes)
    for i in range(4):
        logits_perfect[i, i] = 100.0
    labels = torch.tensor([0, 1, 2, 3])

    mae = compute_angle_mae(logits_perfect, labels, angle_values)
    print(f"  完美预测 MAE (argmax): {mae['ae_argmax'] / mae['count']:.2f}°")
    print(f"  完美预测 MAE (weighted): {mae['ae_weighted'] / mae['count']:.4f}°")
    assert mae['ae_argmax'] / mae['count'] < 0.1
    print("  ✓ 完美预测 MAE ≈ 0")

    # 偏移一个类别
    logits_off_by_one = torch.zeros(4, num_classes)
    logits_off_by_one[0, 1] = 100.0  # 真实 15°, 预测 30°
    logits_off_by_one[1, 2] = 100.0  # 真实 30°, 预测 45°
    logits_off_by_one[2, 3] = 100.0  # 真实 45°, 预测 60°
    logits_off_by_one[3, 0] = 100.0  # 真实 60°, 预测 15°

    mae2 = compute_angle_mae(logits_off_by_one, labels, angle_values)
    mae_argmax = mae2['ae_argmax'] / mae2['count']
    print(f"  偏移一类 MAE (argmax): {mae_argmax:.2f}° (期望 22.5°)")
    # 期望: (15+15+15+45)/4 = 22.5
    assert abs(mae_argmax - 22.5) < 0.1
    print("  ✓ MAE 计算正确")


def test_p1_vs_p2():
    """验证 10：p=1 和 p=2 的对比"""
    print("\n" + "=" * 60)
    print("测试 10：p=1 vs p=2 对比")
    print("=" * 60)

    num_classes = 46
    angle_values = [float(i * 2) for i in range(num_classes)]

    emd_p1 = EarthMoverDistanceLoss(num_classes=num_classes, p=1, angle_values=angle_values)
    emd_p2 = EarthMoverDistanceLoss(num_classes=num_classes, p=2, angle_values=angle_values)

    logits = torch.randn(8, num_classes)
    labels = torch.tensor([0, 5, 10, 15, 20, 25, 30, 35])

    loss_p1 = emd_p1(logits, labels)
    loss_p2 = emd_p2(logits, labels)
    print(f"  EMD loss (p=1): {loss_p1.item():.4f}")
    print(f"  EMD loss (p=2): {loss_p2.item():.4f}")
    print("  ✓ 两种阶数均可正常计算")


if __name__ == '__main__':
    test_emd_loss_basic()
    test_emd_loss_perfect()
    test_emd_loss_ordinal()
    test_emd_with_integer_labels()
    test_gaussian_labels()
    test_emd_gaussian_combination()
    test_ce_wrapper()
    test_build_loss_function()
    test_compute_angle_mae()
    test_p1_vs_p2()

    print("\n" + "=" * 60)
    print("全部测试通过！ ✓")
    print("=" * 60)
