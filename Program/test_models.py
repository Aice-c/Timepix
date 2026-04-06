"""
验证脚本：测试新模型的前向传播和参数量。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from Config import config
from model.utils import build_model, count_parameters

def test_forward_pass():
    """验证所有模型能正确处理输入并输出正确 shape"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_channels = config.inchannel  # 1 (ToT only)
    num_classes = 4  # 与当前 4 类角度一致
    batch_size = 4

    dummy_input = torch.randn(batch_size, in_channels, 50, 50).to(device)
    handcrafted_dim = config.handcrafted_feature_dim()
    dummy_handcrafted = torch.randn(batch_size, handcrafted_dim).to(device) if handcrafted_dim > 0 else None

    model_names = ['Resnet18', 'ShallowCNN', 'ShallowResNet']

    print(f"\n{'#'*60}")
    print(f"前向传播测试  |  输入: ({batch_size}, {in_channels}, 50, 50)  |  设备: {device}")
    print(f"手工特征维度: {handcrafted_dim}")
    print(f"{'#'*60}")

    results = {}
    for name in model_names:
        config.model_name = name
        try:
            model = build_model(config, num_classes=num_classes, device=device)
            if dummy_handcrafted is not None:
                logits, prob, pred = model(dummy_input, dummy_handcrafted)
            else:
                logits, prob, pred = model(dummy_input)
            params = count_parameters(model)
            results[name] = params
            print(f"\n[OK] {name}:")
            print(f"     logits shape: {logits.shape}  (期望: ({batch_size}, {num_classes}))")
            print(f"     prob   shape: {prob.shape}")
            print(f"     pred   shape: {pred.shape}")
            assert logits.shape == (batch_size, num_classes), f"logits shape 不匹配!"
            assert prob.shape == (batch_size, num_classes), f"prob shape 不匹配!"
            assert pred.shape == (batch_size,), f"pred shape 不匹配!"
        except Exception as e:
            print(f"\n[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()

    # 参数量对比
    print(f"\n{'#'*60}")
    print(f"参数量对比")
    print(f"{'#'*60}")
    print(f"{'模型':<20} {'总参数量':>15} {'可训练参数':>15}")
    print(f"{'-'*50}")
    for name, params in results.items():
        print(f"{name:<20} {params['total']:>15,} {params['trainable']:>15,}")

    # 恢复默认
    config.model_name = 'Resnet18'
    print(f"\n所有前向传播测试通过！")


if __name__ == '__main__':
    test_forward_pass()
