"""
模型工具函数：模型工厂、参数统计。

用法：
    from model.utils import build_model, count_parameters
    model = build_model(config, num_classes=4)
"""

import importlib
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict:
    """
    统计模型参数量。

    Returns
    -------
    dict
        {'total': int, 'trainable': int, 'non_trainable': int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
    }


def print_model_summary(model: nn.Module, model_name: str, input_shape: tuple = None):
    """
    打印模型架构摘要和参数量。

    Parameters
    ----------
    model : nn.Module
        模型实例
    model_name : str
        模型名称
    input_shape : tuple, optional
        输入 shape (C, H, W)，用于打印信息
    """
    params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"模型架构: {model_name}")
    print(f"总参数量: {params['total']:,}")
    print(f"可训练参数: {params['trainable']:,}")
    print(f"不可训练参数: {params['non_trainable']:,}")
    if input_shape is not None:
        print(f"输入 shape: {input_shape}")
    print(f"{'='*60}")
    print(model)
    print(f"{'='*60}\n")


def build_model(config, num_classes: int, device=None) -> nn.Module:
    """
    模型工厂函数：根据 config 实例化模型并打印摘要。

    Parameters
    ----------
    config : Config
        配置对象，需包含 model_name 属性
    num_classes : int
        输出类别数
    device : torch.device, optional
        目标设备

    Returns
    -------
    nn.Module
        实例化的模型（已移动到指定设备）
    """
    model_name = config.model_name
    module_name = f"model.{model_name}"
    class_name = "Model"

    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载模型 {model_name}: {e}")

    model = model_class(num_classes=num_classes)

    if device is not None:
        model = model.to(device)

    # 打印模型摘要
    input_shape = (config.inchannel, 50, 50)
    print_model_summary(model, model_name, input_shape=input_shape)

    return model
