import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import torch.nn.functional as F

class Efficientnetb0(nn.Module):
    def __init__(self, num_classes=10):
        super(Efficientnetb0, self).__init__()
        self.base_model = efficientnet_b0(pretrained=False)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(self.base_model.classifier.in_features, num_classes)
        )

    def forward(self, x):
        logits = self.base_model(x)
        prob = F.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        return logits, prob, pred