import torch
import torch.nn as nn
from torchvision.models import resnet50
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.model = resnet50(pretrained=False)  # 加载预训练的ResNet50模型
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes) # 替换为为适合当前分类类别的输出层
            # nn.Softmax(dim=1)  
            # 训练阶段，通常不需要在模型中显式添加 Softmax
            # 因为大多数损失函数（如 CrossEntropyLoss）会自动将 logits 转换为概率。
            # 如果在训练阶段使用了 Softmax，可能会导致重复计算，从而影响损失值。
        )
    
    def forward(self, x):
        logits = self.model(x) # 获取模型输出
        prob = F.softmax(self.model(x), dim=1) # 获取模型分类概率
        pred = torch.argmax(prob, dim=1) # 获取预测类别
        return logits, prob, pred # 返回 logits, 概率和预测类别