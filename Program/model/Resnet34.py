import torch 
import torch.nn as nn
from torchvision.models import resnet34
from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.model = resnet34(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        return logits, prob, pred