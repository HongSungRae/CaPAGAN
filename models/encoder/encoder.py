import torch.nn as nn
import torchvision
from torchvision.models import resnet34, resnet50




class Encoder(nn.Module):
    def __init__(self, num_classes=2, width=34):
        super().__init__()
        if width == 34:
            self.net = nn.Sequential(resnet34(torchvision.models.ResNet34_Weights),
                                     nn.ReLU(),
                                     nn.Linear(1000, 256))
        elif width == 50:
            self.net = nn.Sequential(resnet50(torchvision.models.ResNet50_Weights),
                                     nn.ReLU(),
                                     nn.Linear(1000, 256))
        self.linear = nn.Sequential(nn.ReLU(),
                                    nn.Linear(256,64),
                                    nn.ReLU(),
                                    nn.Linear(64,num_classes)
                                    )

    def forward(self, x):
        return self.linear(self.net(x))