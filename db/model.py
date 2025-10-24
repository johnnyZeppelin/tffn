import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import timm
from timm.models.swin_transformer import SwinTransformerBlock
from config import *

# -------------------------- Helper Blocks --------------------------
class SWMSA(nn.Module):
    """Shifted-Windows Multi-Head Self-Attention (SW-MSA) block"""
    def __init__(self, dim, num_heads=SW_MSA_NUM_HEADS, window_size=SW_MSA_WINDOW_SIZE):
        super().__init__()
        self.swin_block = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2,  # Shift by half window
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.1,
            attn_drop=0.1
        )
    
    def forward(self, x):
        # x: (B, N, C) → Swin block input format (batch, num_patches, dim)
        return self.swin_block(x)

# -------------------------- TPF Block (Two-stream Parallel Fusion) --------------------------
class TPFBlock(nn.Module):
    """TPF Block: Extracts binocular difference (BD) and summation (BS) features"""
    def __init__(self):
        super().__init__()
        # Load ResNet50 (modify to output stage2/stage3/stage4 features)
        self.resnet = resnet50(pretrained=True)  # Use ImageNet pretrained ResNet50
        self.resnet.layer1 = nn.Identity()  # Stage1 not used (paper uses stage2-4)
        
        # BD Block (Binocular Difference)
        self.bd_swin = nn.ModuleList([
            SWMSA(dim=dim) for dim in RESNET50_OUT_DIMS
        ])  # SW-MSA for each stage's left feature
        self.bd_norm = nn.ModuleList([
            nn.LayerNorm(dim) for dim in RESNET50_OUT_DIMS
        ])  # LayerNorm for right feature
        self.bd_fc = nn.ModuleList([
            nn.Linear(dim, dim) for dim in RESNET50_OUT_DIMS
        ])  # FC for feature refinement
        
        # BS Block (Binocular Summation)
        self.bs_global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global pooling for each stage
        self.bs_fc = nn.ModuleList([
            nn.Linear(dim, dim) for dim in RESNET50_OUT_DIMS
        ])  # FC for feature refinement
    
    def _resnet_forward(self, x):
        """Forward pass of ResNet50 to get stage2/stage3/stage4 features"""
        # x: (B, V, 3, H, W) → (B×V, 3, H, W) (V=NUM_VIEWPORTS_PER_IMAGE)
        B, V, C, H, W = x.shape
        x = x.view(B*V, C, H, W)
        
        # Stage0: conv1 + bn1 + relu + maxpool
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # Stage2 (layer2), Stage3 (layer3), Stage4 (layer4)
        stage2 = self.resnet.layer2(x)  # (B×V, 512, H/8, W/8)
        stage3 = self.resnet.layer3(stage2)  # (B×V, 1024, H/16, W/16)
        stage4 = self.resnet.layer4(stage3)  # (B×V, 2048, H/32, W/32)
        
        # Reshape back to (B, V, C, H, W)
        stage2 = stage2.view(B, V, *stage2.shape[1:])
        stage3 = stage3.view(B, V, *stage3.shape[1:])
        stage4 = stage4.view(B, V, *stage4.shape[1:])
        
        return [stage2, stage3, stage4]
    
    def bd_forward(self, left_stages, right_stages):
        """Forward pass for BD Block: left_stages/right_stages = [stage2, stage3, stage4]"""
        bd_features = []
        for i, (left, right) in enumerate(zip(left_stages, right_stages)):
            B, V, C, H, W = left.shape
            
            # Reshape to (B×V, H×W, C) for SW-MSA (flatten spatial dims)
            left_flat = left.view(B*V, C, H*W).permute(0, 2, 1)  # (B×V, H×W, C)
            right_flat = right.view(B*V, C, H*W).permute(0, 2, 1)
            
            # SW-MSA on left feature + GELU
            left_swin = self.bd_swin[i](left_flat)
            left_act = F.gelu(left_swin)
            
            # LayerNorm + GELU on right feature
            right_norm = self.bd_norm[i](right_flat)
            right_act = F.gelu(right_norm)
            
            # Binocular difference: left_act - right_act
            bd = left_act - right_act  # (B×V, H×W, C)
            
            # Residual connection (left_flat) + FC + GELU
            bd_res = F.gelu(self.bd_fc[i](bd + left_flat))  # (B×V, H×W, C)
            
            # Reshape back to (B, V, C, H, W)
            bd_res = bd_res.permute(0, 2, 1).view(B, V, C, H, W)
            
            # Global pooling to (B, V, C)
            bd_pooled = F.adaptive_avg_pool2d(bd_res, (1, 1)).squeeze(-1).squeeze(-1)
            bd_features.append(bd_pooled)
        
        # Concatenate features from all stages (B, V, 512+1024+2048)
        return torch.cat(bd_features, dim=-1)
    
    def bs_forward(self, left_stages, right_stages):
        """Forward pass for BS Block: left_stages/right_stages = [stage2, stage3, stage4]"""
        bs_features = []
        for i, (left, right) in enumerate(zip(left_stages, right_stages)):
            B, V, C, H, W = left.shape
            
            # Global pooling for left/right (B, V, C)
            left_pool = F.adaptive_avg_pool2d(left, (1, 1)).squeeze(-1).squeeze(-1)
            right_pool = F.adaptive_avg_pool2d(right, (1, 1)).squeeze(-1).squeeze(-1)
            
            # Binocular summation: left_pool + right_pool
            bs = left_pool + right_pool  # (B, V, C)
            
            # FC + GELU
            bs_act = F.gelu(self.bs_fc[i](bs))
            bs_features.append(bs_act)
        
        # Concatenate features from all stages (B, V, 512+1024+2048)
        return torch.cat(bs_features, dim=-1)
    
    def forward(self, left_imgs, right_imgs):
        """
        Input: left_imgs (B, V, 3, H, W), right_imgs (B, V, 3, H, W)
        Output: bd_feat (B, V, 3584), bs_feat (B, V, 3584) → 512+1024+2048=3584
        """
        # Extract ResNet stages for left/right
        left_stages = self._resnet_forward(left_imgs)
        right_stages = self._resnet_forward(right_imgs)
        
        # BD and BS features
        bd_feat = self.bd_forward(left_stages, right_stages)
        bs_feat = self.bs_forward(left_stages, right_stages)
        
        # Average over viewports (B, 3584)
        bd_feat_avg = bd_feat.mean(dim=1)
        bs_feat_avg = bs_feat.mean(dim=1)
        
        return bd_feat_avg, bs_feat_avg

# -------------------------- PDIE Block (Pseudo-difference Information Extraction) --------------------------
class PDIEBlock(nn.Module):
    """PDIE Block: Uses VMamba to extract monocular features from pseudo-difference"""
    def __init__(self):
        super().__init__()
        # Load VMamba (from https://github.com/MzeroMiko/VMamba)
        self.vmamba = self._load_vmamba()
        self.vmamba.train()  # Not frozen (update parameters)
        
        # Difference map processing
        self.diff_norm = nn.LayerNorm(VMAMBA_OUT_DIM)
        self.fc = nn.Linear(VMAMBA_OUT_DIM, VMAMBA_OUT_DIM)
    
    def _load_vmamba(self):
        """Load pretrained VMamba model"""
        # Import VMamba from the official repo (adjust import based on user's vmamba folder)
        from vmamba.models.vmamba import VMamba
        
        # Initialize VMamba (match pretrained config)
        vmamba_config = {
            "img_size": IMAGE_SIZE[0],
            "in_chans": 3,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "drop_rate": 0.1,
        }
        model = VMamba(**vmamba_config)
        
        # Load pretrained weights
        checkpoint = torch.load(VMAMBA_PRETRAINED, map_location=DEVICE)
        model.load_state_dict(checkpoint["model"], strict=False)
        
        # Modify VMamba to output features (not classification)
        model.head = nn.Identity()  # Remove classification head
        return model
    
    def forward(self, distorted_right, restored_right):
        """
        Input: distorted_right (B, V, 3, H, W), restored_right (B, V, 3, H, W)
        Output: m_feat (B, VMAMBA_OUT_DIM)
        """
        B, V, C, H, W = distorted_right.shape
        
        # Compute pseudo-difference map: (distorted - restored)
        diff_map = distorted_right - restored_right  # (B, V, 3, H, W)
        
        # Reshape to (B×V, 3, H, W) for VMamba
        diff_map = diff_map.view(B*V, C, H, W)
        
        # Extract VMamba features
        vmamba_feat = self.vmamba(diff_map)  # (B×V, VMAMBA_OUT_DIM)
        
        # Reshape back to (B, V, VMAMBA_OUT_DIM) and average over viewports
        vmamba_feat = vmamba_feat.view(B, V, VMAMBA_OUT_DIM)
        m_feat = vmamba_feat.mean(dim=1)  # (B, VMAMBA_OUT_DIM)
        
        # Refine with LayerNorm + FC + GELU
        m_feat = F.gelu(self.fc(self.diff_norm(m_feat)))
        return m_feat

# -------------------------- FF Block (Feature Fusion) --------------------------
class FFBlock(nn.Module):
    """FF Block: Fuses BD, BS, and monocular features"""
    def __init__(self, bd_dim=3584, bs_dim=3584, m_dim=VMAMBA_OUT_DIM):
        super().__init__()
        self.bd_swin = SWMSA(dim=bd_dim)  # SW-MSA on BD feature
        self.bs_norm = nn.LayerNorm(bs_dim)  # LayerNorm on BS feature
        self.m_norm = nn.LayerNorm(m_dim)    # LayerNorm on monocular feature
        
        # Fusion FC layers
        self.fc1 = nn.Linear(bd_dim + bs_dim + m_dim, FF_HIDDEN_DIM)
        self.fc2 = nn.Linear(FF_HIDDEN_DIM, 1)  # Output quality score
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, bd_feat, bs_feat, m_feat):
        """
        Input: bd_feat (B, 3584), bs_feat (B, 3584), m_feat (B, VMAMBA_OUT_DIM)
        Output: score (B, 1)
        """
        # Process BD feature: SW-MSA (add dummy spatial dim for SW-MSA)
        bd_reshaped = bd_feat.unsqueeze(1)  # (B, 1, 3584)
        bd_processed = self.bd_swin(bd_reshaped).squeeze(1)  # (B, 3584)
        
        # Process BS feature: LayerNorm + GELU
        bs_processed = F.gelu(self.bs_norm(bs_feat))  # (B, 3584)
        
        # Process monocular feature: LayerNorm + GELU
        m_processed = F.gelu(self.m_norm(m_feat))  # (B, VMAMBA_OUT_DIM)
        
        # Concatenate all features
        fused = torch.cat([bd_processed, bs_processed, m_processed], dim=-1)  # (B, 3584+3584+VMAMBA_OUT_DIM)
        
        # Predict quality score
        score = F.gelu(self.fc1(fused))
        score = self.dropout(score)
        score = self.fc2(score)
        
        return score

# -------------------------- TFFN Full Model --------------------------
class TFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.tpf_block = TPFBlock()
        self.pdie_block = PDIEBlock()
        self.ff_block = FFBlock()
    
    def forward(self, left_imgs, right_imgs, restored_right_imgs):
        """
        Input:
            left_imgs: (B, V, 3, H, W) → Left view viewports
            right_imgs: (B, V, 3, H, W) → Right view viewports
            restored_right_imgs: (B, V, 3, H, W) → Restored right view viewports
        Output:
            score: (B, 1) → Predicted quality score
        """
        # Step 1: TPF Block → BD and BS features
        bd_feat, bs_feat = self.tpf_block(left_imgs, right_imgs)
        
        # Step 2: PDIE Block → Monocular feature
        m_feat = self.pdie_block(right_imgs, restored_right_imgs)
        
        # Step 3: FF Block → Quality score
        score = self.ff_block(bd_feat, bs_feat, m_feat)
        
        return score