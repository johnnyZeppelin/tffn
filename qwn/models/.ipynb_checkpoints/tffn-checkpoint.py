import torch
import torch.nn as nn
from .resnet50_tpf import ResNet50TPF
from .vmamba_wrapper import VMambaWrapper
from timm.models.layers import LayerNorm2d
from einops import rearrange

class BDModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sw_msa = nn.Identity()  # Placeholder; replace with real SW-MSA if needed
        self.norm_left = LayerNorm2d(dim)
        self.norm_right = LayerNorm2d(dim)

    def forward(self, F_left, F_right):
        # Simplified: no real SW-MSA due to complexity; use identity
        F_left = F_left + torch.relu(self.norm_left(F_left))
        F_right = F_right + torch.relu(self.norm_right(F_right))
        return F_left - F_right

class BSModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, F_left, F_right):
        return F_left + F_right

class TPFBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet50TPF(pretrained=True)
        self.bd = BDModule(2048)  # stage4 dim
        self.bs = BSModule()
        self.fc = nn.Linear(2048, 512)

    def forward(self, left, right):
        # left, right: [B, 20, C, H, W] → process each viewport
        B, V, C, H, W = left.shape
        left = left.view(B*V, C, H, W)
        right = right.view(B*V, C, H, W)

        feats_left = self.resnet(left)[-1]  # [B*V, 2048, h, w]
        feats_right = self.resnet(right)[-1]

        bd_feat = self.bd(feats_left, feats_right)  # [B*V, 2048, h, w]
        bs_feat = self.bs(feats_left, feats_right)

        # Global avg pool
        bd_feat = bd_feat.mean(dim=[2,3])  # [B*V, 2048]
        bs_feat = bs_feat.mean(dim=[2,3])

        bd_feat = self.fc(bd_feat).view(B, V, -1).mean(dim=1)  # [B, 512]
        bs_feat = self.fc(bs_feat).view(B, V, -1).mean(dim=1)
        return bd_feat, bs_feat

class PDIEBlock(nn.Module):
    def __init__(self, vmamba_ckpt):
        super().__init__()
        self.vmamba = VMambaWrapper(vmamba_ckpt, num_classes=512)

    def forward(self, diff_maps):
        # diff_maps: [B, 20, C, H, W] → concat along batch
        B, V, C, H, W = diff_maps.shape
        x = diff_maps.view(B*V, C, H, W)
        feats = self.vmamba(x)  # [B*V, 512]
        return feats.view(B, V, -1).mean(dim=1)  # [B, 512]

class FFBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim * 3, 1)

    def forward(self, bd, bs, mf):
        fused = torch.cat([bd, bs, mf], dim=1)  # [B, 1536]
        return self.fc(fused).squeeze(-1)

class TFFN(nn.Module):
    def __init__(self, vmamba_ckpt):
        super().__init__()
        self.tpf = TPFBlock()
        self.pdie = PDIEBlock(vmamba_ckpt)
        self.ff = FFBlock()

    def forward(self, left, right, diff):
        bd, bs = self.tpf(left, right)
        mf = self.pdie(diff)
        score = self.ff(bd, bs, mf)
        return score