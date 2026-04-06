"""
ShallowResNet —— 带残差连接的浅层 CNN，专为稀疏小团簇数据设计。

在 ShallowCNN 基础上为卷积块添加残差连接（shortcut），有助于梯度流动和训练稳定性。

下采样过程（输入 50×50）：
  50×50 →(Stem)→ 50×50 →(ResBlock1)→ 50×50
  →(ResBlock2, stride=2)→ 25×25 →(ResBlock3)→ 25×25 → GAP → 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import config

print("Loading ShallowResNet model...")


class FC_Attention(nn.Module):
    """对拼接后的特征做加权的全连接注意力模块"""
    def __init__(self, in_features, reduction=8):
        super(FC_Attention, self).__init__()
        self.fc1 = nn.Linear(in_features, in_features // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features // reduction, in_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.fc1(x)
        att = self.relu(att)
        att = self.fc2(att)
        att = self.sigmoid(att)
        return x * att


class ResidualBlock(nn.Module):
    """
    残差块：Conv(3×3) → BN → ReLU → Conv(3×3) → BN → + shortcut → ReLU

    当输入输出通道数不同或需要下采样时，shortcut 使用 Conv(1×1) + BN 调整。

    Parameters
    ----------
    in_channels : int
        输入通道数
    out_channels : int
        输出通道数
    stride : int
        第二个卷积的步幅（用于下采样），默认 1
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut: 当通道数或空间尺寸改变时，用 1×1 卷积 + BN 调整
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ShallowResNet(nn.Module):
    """
    带残差连接的浅层 CNN 特征提取器。

    架构：
      Stem(in→32) → ResBlock1(32→64, s=1) → ResBlock2(64→128, s=2) → ResBlock3(128→256, s=1) → GAP

    感受野：
      Stem: 3×3 → ResBlock1: 7×7 → ResBlock2: 11×11 → ResBlock3: 15×15

    Parameters
    ----------
    in_channels : int
        输入通道数
    """
    def __init__(self, in_channels):
        super(ShallowResNet, self).__init__()

        # Stem: 输出 (batch, 32, 50, 50)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ResBlock1: 32→64, stride=1, 输出 (batch, 64, 50, 50), 感受野 7×7
        self.resblock1 = ResidualBlock(32, 64, stride=1)

        # ResBlock2: 64→128, stride=2, 输出 (batch, 128, 25, 25), 感受野 11×11 — 唯一下采样
        self.resblock2 = ResidualBlock(64, 128, stride=2)

        # ResBlock3: 128→256, stride=1, 输出 (batch, 256, 25, 25), 感受野 15×15
        self.resblock3 = ResidualBlock(128, 256, stride=1)

        # 全局平均池化: 输出 (batch, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 权重初始化：卷积层 Kaiming，BN 层默认
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (batch, in_channels, 50, 50)
        x = self.stem(x)          # (batch, 32, 50, 50)
        x = self.resblock1(x)     # (batch, 64, 50, 50)
        x = self.resblock2(x)     # (batch, 128, 25, 25)
        x = self.resblock3(x)     # (batch, 256, 25, 25)
        x = self.gap(x)           # (batch, 256, 1, 1)
        x = torch.flatten(x, 1)   # (batch, 256)
        return x


class Model(nn.Module):
    """
    ShallowResNet 完整模型（骨干 + 注意力 + 分类/回归头），接口与 Resnet18.Model 保持一致。

    Parameters
    ----------
    num_classes : int
        输出类别数（分类任务时使用）
    task : str
        'classification' 或 'regression'
    """
    def __init__(self, num_classes, task=None):
        super(Model, self).__init__()
        self.task = task or getattr(config, 'task', 'classification')

        # 骨干网络
        self.backbone = ShallowResNet(in_channels=config.inchannel)

        # 注意力机制（支持可选的手工特征拼接）
        self.handcrafted_dim = config.handcrafted_feature_dim()
        feature_dim = 256 + self.handcrafted_dim
        self.attention = FC_Attention(feature_dim)

        # 输出头
        if self.task == 'regression':
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(128, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(128, num_classes),
            )

        self.supports_handcrafted_features = True

        # 全连接层 Xavier 初始化
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, samples, handcrafted_features=None):
        # 特征提取
        feature = self.backbone(samples)  # (batch, 256)

        # 手工特征拼接（若配置启用）
        if self.handcrafted_dim > 0:
            if handcrafted_features is None:
                raise ValueError("手工特征未提供，但配置要求使用它们")
            if handcrafted_features.dim() == 1:
                handcrafted_features = handcrafted_features.unsqueeze(0)
            feature = torch.cat((feature, handcrafted_features), dim=1)

        # 注意力加权
        feature = self.attention(feature)

        # 输出
        output = self.classifier(feature)

        if self.task == 'regression':
            output = torch.sigmoid(output)  # 归一化到 [0, 1]
            return output.squeeze(-1), None, None
        else:
            logits = output
            prob = F.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            return logits, prob, pred
