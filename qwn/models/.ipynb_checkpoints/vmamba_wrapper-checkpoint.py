import torch
import torch.nn as nn
from vmamba import VSSM  # assuming you have vmamba.py in PYTHONPATH

class VMambaWrapper(nn.Module):
    def __init__(self, ckpt_path, num_classes=512):
        super().__init__()
        # Load VMamba-S (50M params) as in paper
        self.vmamba = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            depths=[2, 2, 15, 2],
            dims=[96, 192, 384, 768],
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=4.0,
            downsample_version="v3",
            patchembed_version="v2",
        )
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            self.vmamba.load_state_dict(ckpt, strict=False)
        # We'll use global avg pooling on final feature map
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, C, H, W] → VMamba expects [B, C, H, W]
        # VMamba returns [B, L, D] if return_all_tokens=True, else [B, num_classes]
        # We want feature map, so modify to return features
        # But original VMamba returns cls token only.
        # So we assume you modified VMamba to return patch tokens.
        # Alternatively, use feature extraction mode.
        # For simplicity, assume we get [B, D, H', W'] from backbone
        feats = self.vmamba.forward_features(x)  # [B, D, H', W']
        feats = feats.flatten(2)  # [B, D, N]
        feats = self.avgpool(feats).squeeze(-1)  # [B, D]
        return feats