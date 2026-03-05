import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
from Config import config
print("Loading ResNet18 model...")

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
        self.model = resnet18(pretrained=False)  # 加载预训练的ResNet18模型
        self.model.conv1 = nn.Conv2d(in_channels=config.inchannel, out_channels=64, kernel_size=config.kernel_size, stride=1, padding=0, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, config.cnn_feature_size)  # 替换为适合当前分类类别的输出层 

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        #x = self.maxpool(x) 删去池化层，即略过池化层输出
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        
        middle_feature = x  # 获取中间特征
        return middle_feature

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        ## 初始化CNN模型 
        self.CNN = CNN()

        ## 初始化注意力机制
        self.handcrafted_dim = config.handcrafted_feature_dim()
        feature_dim = config.cnn_feature_size + self.handcrafted_dim
        self.attention = FC_Attention(feature_dim)  # 注意力机制输入大小根据手工特征动态调整

        ## 初始化分类器  
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, config.out_feature_size),  # CNN特征和手工特征拼接后的输入大小
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),  
            nn.Linear(config.out_feature_size, num_classes)  # 输出层，num_classes为分类数量
        )

        self.supports_handcrafted_features = True

    def forward(self, samples, handcrafted_features=None):
        ## 特征提取
        cnn_feature = self.CNN(samples)
        if self.handcrafted_dim > 0:
            if handcrafted_features is None:
                raise ValueError("手工特征未提供，但配置要求使用它们")
            if handcrafted_features.dim() == 1:
                handcrafted_features = handcrafted_features.unsqueeze(1)
            feature = torch.cat((cnn_feature, handcrafted_features), dim=1)  # 将CNN特征和手工特征拼接
        else:
            feature = cnn_feature
        
        ## 注意力机制
        feature = self.attention(feature)  

        ## 分类
        x = self.classifier(feature)  
        
        logits = x # 获取模型输出的logits
        prob = F.softmax(logits, dim=1) # 获取模型分类概率
        pred = torch.argmax(prob, dim=1) # 获取预测类别
        return logits, prob, pred # 返回 logits, 概率和预测类别