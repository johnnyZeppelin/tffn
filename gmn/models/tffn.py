import torch
import torch.nn as nn
from .backbones import get_resnet_extractor, get_vmamba_extractor
from .blocks import BD_Block, BS_Block, FF_Block

class TFFN(nn.Module):
    """
    The main Three-branch Feature Fusion Network (TFFN) model[cite: 2].
    """
    def __init__(self, num_heads=8, window_size=7):
        super().__init__()
        
        # --- 1. Backbones ---
        
        # TPF Backbone (ResNet50)
        self.resnet_extractor = get_resnet_extractor()
        # ResNet feature dimensions
        s2_dim, s3_dim, s4_dim = 512, 1024, 2048
        
        # PDIE Backbone (VMamba)
        self.vmamba_extractor = get_vmamba_extractor()
        mf_dim = 768 # vssm_small stage 4 output dim
        
        # --- 2. Hierarchical Fusion Blocks (TPF) ---
        # We define output dims for the fusion blocks
        f_dim = 256 # Fused feature dim
        
        # Stage 2 (from ResNet s2)
        self.bd_s2 = BD_Block(s2_dim, None, num_heads, window_size, f_dim)
        self.bs_s2 = BS_Block(s2_dim, None, f_dim)
        
        # Stage 3 (from ResNet s3 + prev)
        self.bd_s3 = BD_Block(s3_dim, f_dim, num_heads, window_size, f_dim)
        self.bs_s3 = BS_Block(s3_dim, f_dim, f_dim)
        
        # Stage 4 (from ResNet s4 + prev)
        self.bd_s4 = BD_Block(s4_dim, f_dim, num_heads, window_size, f_dim)
        self.bs_s4 = BS_Block(s4_dim, f_dim, f_dim)
        
        # --- 3. Final Feature Fusion (FF) ---
        self.ff_block = FF_Block(
            bd_dim=f_dim, 
            mf_dim=mf_dim, 
            bs_dim=f_dim,
            num_heads=num_heads,
            window_size=window_size
        )
        final_feature_dim = f_dim + mf_dim + f_dim
        
        # --- 4. Quality Score Regression ---
        # (Formula 10) [cite: 291]
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Avg Pooling
            nn.Flatten(),
            nn.LayerNorm(final_feature_dim),
            nn.Linear(final_feature_dim, 1)
        )

    def forward(self, l_vps, r_vps, r_restored_vps):
        """
        Inputs:
        - l_vps: Left viewports [B, N_v, C, H, W] (N_v=20)
        - r_vps: Right viewports [B, N_v, C, H, W]
        - r_restored_vps: Restored right viewports [B, N_v, C, H, W]
        """
        B, N_v, C, H, W = l_vps.shape
        
        # Reshape from (B, N_v, ...) to (B * N_v, ...) to process all viewports
        l_vps_flat = l_vps.view(B * N_v, C, H, W)
        r_vps_flat = r_vps.view(B * N_v, C, H, W)
        r_restored_vps_flat = r_restored_vps.view(B * N_v, C, H, W)
        
        # --- TPF Branch ---
        # Extract ResNet features 
        f_l = self.resnet_extractor(l_vps_flat)
        f_r = self.resnet_extractor(r_vps_flat)
        # f_l, f_r are dicts {'s2': ..., 's3': ..., 's4': ...}
        
        # Hierarchical fusion [cite: 9]
        bd_2 = self.bd_s2(f_l['s2'], f_r['s2'], None)
        bs_2 = self.bs_s2(f_l['s2'], f_r['s2'], None)
        
        bd_3 = self.bd_s3(f_l['s3'], f_r['s3'], bd_2)
        bs_3 = self.bs_s3(f_l['s3'], f_r['s3'], bs_2)
        
        bd_4 = self.bd_s4(f_l['s4'], f_r['s4'], bd_3) # Final BD feature
        bs_4 = self.bs_s4(f_l['s4'], f_r['s4'], bs_3) # Final BS feature
        
        # --- PDIE Branch ---
        # Calculate pseudo-difference map [cite: 271]
        # Only uses right view 
        md_map = r_vps_flat - r_restored_vps_flat 
        
        # Extract VMamba features [cite: 277]
        mf = self.vmamba_extractor(md_map)[0] # [0] as it returns a list
        
        # --- Final Fusion ---
        fused_features = self.ff_block(bd_4, mf, bs_4) # [cite: 287]
        
        # --- Regression ---
        # Get score for each viewport
        scores_flat = self.regression_head(fused_features) # [B * N_v, 1]
        
        # Reshape back to (B, N_v, 1)
        scores_viewports = scores_flat.view(B, N_v, 1)
        
        # Final score is the mean of all viewport scores for the image
        final_score = torch.mean(scores_viewports, dim=1) # [B, 1]
        
        return final_score