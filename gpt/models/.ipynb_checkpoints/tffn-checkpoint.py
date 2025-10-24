
import torch.nn as nn
import torch
from .backbones import resnet50_stages
from .sw_msa import WindowedSelfAttention
from .vmamba_wrapper import load_vmamba

class BDModule(nn.Module):
    def __init__(self, in_channels, hidden=256, window_size=7):
        super().__init__()
        self.left_attn = WindowedSelfAttention(in_channels, num_heads=8, window_size=window_size)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(in_channels)
        self.fc = nn.Linear(in_channels*2, hidden)
    def forward(self, FL, FR, prev_bd=None):
        # FL, FR: B x C x H x W
        FL_t = FL + self.gelu(self.left_attn(FL))
        FR_t = FR + self.gelu(self.ln(FR.permute(0,2,3,1)).permute(0,3,1,2)) if FR is not None else FR
        # ensure same spatial dims
        if FL_t.shape != FR_t.shape:
            # simple resize FR to FL
            FR_t = nn.functional.interpolate(FR_t, size=(FL_t.shape[2], FL_t.shape[3]), mode='bilinear', align_corners=False)
        diff = FL_t - FR_t
        if prev_bd is None:
            concat = diff.view(diff.size(0), -1).unsqueeze(1)
        else:
            prev_vec = prev_bd.view(prev_bd.size(0), -1).unsqueeze(1)
            concat = torch.cat([prev_vec, diff.view(diff.size(0), -1).unsqueeze(1)], dim=1)
        # FC applied to last dim
        out = self.fc(concat.view(concat.size(0), -1))
        return out

class BSModule(nn.Module):
    def __init__(self, in_channels, hidden=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_channels, hidden)
    def forward(self, FL, FR, prev_bs=None):
        gL = self.pool(FL).view(FL.size(0), -1)
        gR = self.pool(FR).view(FR.size(0), -1)
        s = gL + gR
        out = self.fc(s)
        if prev_bs is not None:
            out = torch.cat([prev_bs, out], dim=1)
        return out

class FFBlock(nn.Module):
    def __init__(self, bd_dim, mf_dim, bs_dim, out_dim=128):
        super().__init__()
        self.sw = WindowedSelfAttention(bd_dim, num_heads=8, window_size=7)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(bd_dim)
        self.fc = nn.Linear(bd_dim + mf_dim + bs_dim, out_dim)
    def forward(self, BD, MF, BS):
        BD_t = BD + self.gelu(self.sw(BD.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)) if BD.ndim==2 else BD
        BS_t = BS + self.gelu(self.ln(BS))
        F = torch.cat([BD_t, MF, BS_t], dim=1)
        return self.fc(F)

class TFFN(nn.Module):
    def __init__(self, root_vmamba=None, pretrained_resnet=True):
        super().__init__()
        # backbones for left/right viewports
        self.backbone = resnet50_stages(pretrained=pretrained_resnet)
        # BD and BS for stage2,3,4 channels; for simplicity, we set in_channels equal to stage channels
        self.bd2 = BDModule(in_channels=512, hidden=128)
        self.bd3 = BDModule(in_channels=1024, hidden=128)
        self.bd4 = BDModule(in_channels=2048, hidden=128)
        self.bs2 = BSModule(in_channels=512, hidden=64)
        self.bs3 = BSModule(in_channels=1024, hidden=64)
        self.bs4 = BSModule(in_channels=2048, hidden=64)
        # VMamba for PDIE
        self.vmamba = load_vmamba(root_vmamba or ".")
        # MF dim: try to infer 512; fallback 512
        self.mf_dim = 512
        # final fusion
        self.ff = FFBlock(bd_dim=128*3, mf_dim=self.mf_dim, bs_dim=64*3, out_dim=256)
        self.reg = nn.Sequential(nn.LayerNorm(256), nn.Linear(256,1))
    def forward(self, left, right, restored_right):
        # left,right,restored_right: Bx3xHxxW (already aggregated across viewports in dataset design)
        # extract features per view
        L_feats = self.backbone(left)   # dict s2,s3,s4
        R_feats = self.backbone(right)
        # BD multi-scale
        bd2 = self.bd2(L_feats['s2'], R_feats['s2'])
        bd3 = self.bd3(L_feats['s3'], R_feats['s3'], prev_bd=bd2.unsqueeze(-1) if bd2 is not None else None)
        bd4 = self.bd4(L_feats['s4'], R_feats['s4'], prev_bd=bd3.unsqueeze(-1) if bd3 is not None else None)
        BD_concat = torch.cat([bd2, bd3, bd4], dim=1)
        # BS multi-scale
        bs2 = self.bs2(L_feats['s2'], R_feats['s2'])
        bs3 = self.bs3(L_feats['s3'], R_feats['s3'])
        bs4 = self.bs4(L_feats['s4'], R_feats['s4'])
        BS_concat = torch.cat([bs2, bs3, bs4], dim=1)
        # PDIE: monocular feature from difference maps (right - restored_right)
        diff = right - restored_right
        MF = self.vmamba(diff.unsqueeze(0) if diff.ndim==3 else diff)
        if MF.ndim==1:
            MF = MF.unsqueeze(0)
        # ensure MF is [B x mf_dim]
        if MF.size(1) != self.mf_dim:
            # if vmamba returned a flattened vec, project
            MF = MF.view(MF.size(0), -1)
            if MF.size(1) > self.mf_dim:
                MF = MF[:,:self.mf_dim]
            else:
                pad = torch.zeros(MF.size(0), self.mf_dim - MF.size(1), device=MF.device)
                MF = torch.cat([MF, pad], dim=1)
        # fuse
        F = self.ff(BD_concat, MF, BS_concat)
        q = self.reg(F).squeeze(1)
        return q
