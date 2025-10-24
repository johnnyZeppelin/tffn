import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

class ResNet50TPF(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # not used (stage1)
        self.layer2 = resnet.layer2  # stage2 → used
        self.layer3 = resnet.layer3  # stage3
        self.layer4 = resnet.layer4  # stage4

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        s2 = self.layer2(x)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return [s2, s3, s4]  # multi-scale features