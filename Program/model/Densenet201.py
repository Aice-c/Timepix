import torch
import torch.nn as nn
from torchvision.models import densenet201
from torch.nn import functional as F
from Config import config

print("Loading DenseNet201 model...")


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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 加载DenseNet201骨干
        self.model = densenet201(pretrained=False)

        # 适配输入通道与卷积核，且去除首层下采样（与ResNet18保持一致的改造思路）
        self.model.features.conv0 = nn.Conv2d(
            in_channels=config.input_channels(),
            out_channels=self.model.features.conv0.out_channels,
            kernel_size=config.kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        # 跳过初始池化层（原本为 3x3, stride=2 的 pool0），与 ResNet18 的“去掉maxpool”一致
        self.model.features.pool0 = nn.Identity()

        # 将分类器替换为特征投影层，输出固定的卷积特征维度
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, config.cnn_feature_size
        )

    def forward(self, x):
        # DenseNet 的标准 forward（手动展开以确保与上面的改造兼容）
        x = self.model.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x  # 返回中间卷积特征


class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        # 初始化CNN模型
        self.CNN = CNN()

        # 初始化注意力机制（按是否启用手工特征动态确定输入维度）
        self.handcrafted_dim = config.handcrafted_feature_dim()
        feature_dim = config.cnn_feature_size + self.handcrafted_dim
        self.attention = FC_Attention(feature_dim)

        # 初始化分类器
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, config.out_feature_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.out_feature_size, num_classes),
        )

        self.supports_handcrafted_features = True

    def forward(self, samples, handcrafted_features=None):
        # 卷积特征
        cnn_feature = self.CNN(samples)

        # 手工特征拼接（若配置启用）
        if self.handcrafted_dim > 0:
            if handcrafted_features is None:
                raise ValueError("手工特征未提供，但配置要求使用它们")
            if handcrafted_features.dim() == 1:
                handcrafted_features = handcrafted_features.unsqueeze(0)
            feature = torch.cat((cnn_feature, handcrafted_features), dim=1)
        else:
            feature = cnn_feature

        # 注意力加权
        feature = self.attention(feature)

        # 分类头
        logits = self.classifier(feature)
        prob = F.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        return logits, prob, pred