
import torch.nn as nn
import torchvision.models as models
import torch

def resnet50_stages(pretrained=True):
    # returns a module that extracts stage2, stage3, stage4 features from ResNet50
    res = models.resnet50(pretrained=pretrained)
    # stage0: conv+bn+relu+maxpool handled by res.layer1.. etc
    # we'll use layer2, layer3, layer4 as stage2,3,4
    class ResStages(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4
        def forward(self, x):
            x = self.conv1(x)   # stage0
            s1 = self.layer1(x) # stage1
            s2 = self.layer2(s1) # stage2
            s3 = self.layer3(s2) # stage3
            s4 = self.layer4(s3) # stage4
            return {'s2': s2, 's3': s3, 's4': s4}
    return ResStages(res)
