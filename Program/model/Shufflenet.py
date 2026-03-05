import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_5
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.model = shufflenet_v2_x1_5(pretrained=False)  # 加载预训练的ShuffleNetV2模型
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes)  # 替换为适合当前分类类别的输出层
        )
    
    def forward(self, x):
        logits = self.model(x)  # 获取模型输出
        prob = F.softmax(logits, dim=1)  # 获取模型分类概率
        pred = torch.argmax(prob, dim=1)  # 获取预测类别
        return logits, prob, pred  # 返回 logits, 概率和预测类别
