"""
ShallowCNN —— 专为稀疏小团簇数据设计的浅层 CNN。

设计原则：
  1. 最少下采样：整个网络仅 1 次空间下采样（Block 3, stride=2）
  2. 浅层：4 个卷积块，避免对 ~12 个活跃像素的数据过拟合
  3. 感受野覆盖团簇：3 层 3×3 → 感受野 7×7，刚好覆盖活跃区域
  4. 通道数适中：32 → 64 → 128 → 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import config

print("Loading ShallowCNN model...")


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


class ShallowCNN(nn.Module):
    """
    浅层 CNN 特征提取器。

    下采样过程（输入 50×50）：
      50×50 →(Block1, stride=1)→ 50×50 →(Block2, stride=1)→ 50×50
      →(Block3, stride=2)→ 25×25 →(Block4, stride=1)→ 25×25 → GAP → 256

    活跃区域变化（约 7×7 的团簇）：
      7×7 → 7×7 → 7×7 → 4×4 → 4×4 → 保留！

    Parameters
    ----------
    in_channels : int
        输入通道数
    """
    def __init__(self, in_channels):
        super(ShallowCNN, self).__init__()

        # Block 1: 输出 (batch, 32, 50, 50), 感受野 3×3
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Block 2: 输出 (batch, 64, 50, 50), 感受野 5×5
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Block 3: 输出 (batch, 128, 25, 25), 感受野 7×7 — 唯一一次下采样
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Block 4: 输出 (batch, 256, 25, 25), 感受野 11×11
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

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
        x = self.block1(x)          # (batch, 32, 50, 50)
        x = self.block2(x)          # (batch, 64, 50, 50)
        x = self.block3(x)          # (batch, 128, 25, 25)
        x = self.block4(x)          # (batch, 256, 25, 25)
        x = self.gap(x)             # (batch, 256, 1, 1)
        x = torch.flatten(x, 1)     # (batch, 256)
        return x


class Model(nn.Module):
    """
    ShallowCNN 完整模型（骨干 + 注意力 + 分类头），接口与 Resnet18.Model 保持一致。

    Parameters
    ----------
    num_classes : int
        输出类别数
    """
    def __init__(self, num_classes):
        super(Model, self).__init__()
        # 骨干网络
        self.backbone = ShallowCNN(in_channels=config.inchannel)

        # 注意力机制（支持可选的手工特征拼接）
        self.handcrafted_dim = config.handcrafted_feature_dim()
        feature_dim = 256 + self.handcrafted_dim
        self.attention = FC_Attention(feature_dim)

        # 分类头: Linear(feature_dim→128) + ReLU + Dropout + Linear(128→num_classes)
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

        # 分类
        logits = self.classifier(feature)       # (batch, num_classes)
        prob = F.softmax(logits, dim=1)          # (batch, num_classes)
        pred = torch.argmax(prob, dim=1)         # (batch,)
        return logits, prob, pred
