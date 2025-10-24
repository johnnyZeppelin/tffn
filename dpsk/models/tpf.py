import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class BinocularDifferenceStream(nn.Module):
    def __init__(self, feature_dim):
        super(BinocularDifferenceStream, self).__init__()
        self.feature_dim = feature_dim
        
        # SW-MSA for left view (simplified implementation)
        self.sw_msa = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=8, batch_first=True)
        self.ln_left = nn.LayerNorm(feature_dim)
        self.ln_right = nn.LayerNorm(feature_dim)
        self.gelu = nn.GELU()
        
        # Fusion layers
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        
    def forward(self, left_feat, right_feat, prev_bd=None):
        # left_feat, right_feat: (batch, channels, height, width)
        batch, C, H, W = left_feat.shape
        
        # Flatten spatial dimensions
        left_flat = left_feat.view(batch, C, -1).transpose(1, 2)  # (batch, H*W, C)
        right_flat = right_feat.view(batch, C, -1).transpose(1, 2)
        
        # Left stream with SW-MSA
        left_attn, _ = self.sw_msa(left_flat, left_flat, left_flat)
        left_out = left_flat + self.gelu(left_attn)
        left_out = self.ln_left(left_out)
        
        # Right stream (only LayerNorm + GELU)
        right_out = self.gelu(self.ln_right(right_flat))
        
        # Binocular difference
        bd_feat = left_out - right_out  # (batch, H*W, C)
        
        # Global average pooling
        bd_feat = bd_feat.mean(dim=1)  # (batch, C)
        
        # Hierarchical fusion with previous BD features
        if prev_bd is not None:
            bd_feat = torch.cat([prev_bd, bd_feat], dim=1)
            bd_feat = self.fc(bd_feat)
            
        return bd_feat

class BinocularSummationStream(nn.Module):
    def __init__(self, feature_dim):
        super(BinocularSummationStream, self).__init__()
        self.feature_dim = feature_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feature_dim * 2, feature_dim)
        
    def forward(self, left_feat, right_feat, prev_bs=None):
        # Global pooling
        left_pooled = self.global_pool(left_feat).squeeze(-1).squeeze(-1)  # (batch, C)
        right_pooled = self.global_pool(right_feat).squeeze(-1).squeeze(-1)
        
        # Binocular summation
        bs_feat = left_pooled + right_pooled
        
        # Hierarchical fusion
        if prev_bs is not None:
            bs_feat = torch.cat([prev_bs, bs_feat], dim=1)
            bs_feat = self.fc(bs_feat)
            
        return bs_feat

class TPFBlock(nn.Module):
    def __init__(self, feature_dims=[512, 1024, 2048]):
        super(TPFBlock, self).__init__()
        self.feature_dims = feature_dims
        
        # Initialize ResNet50 backbone
        resnet = resnet50(pretrained=True)
        self.resnet_stage0 = nn.Sequential(*list(resnet.children())[:4])  # conv1, bn1, relu, maxpool
        self.resnet_stage1 = resnet.layer1  # output: 256
        self.resnet_stage2 = resnet.layer2  # output: 512
        self.resnet_stage3 = resnet.layer3  # output: 1024
        self.resnet_stage4 = resnet.layer4  # output: 2048
        
        # BD and BS streams for each stage
        self.bd_streams = nn.ModuleList([
            BinocularDifferenceStream(feature_dims[0]),  # stage2
            BinocularDifferenceStream(feature_dims[1]),  # stage3
            BinocularDifferenceStream(feature_dims[2])   # stage4
        ])
        
        self.bs_streams = nn.ModuleList([
            BinocularSummationStream(feature_dims[0]),
            BinocularSummationStream(feature_dims[1]),
            BinocularSummationStream(feature_dims[2])
        ])
        
        # Final fusion
        self.bd_fusion = nn.Linear(feature_dims[2], 512)
        self.bs_fusion = nn.Linear(feature_dims[2], 512)
        
    def forward(self, left_viewports, right_viewports):
        batch_size, num_viewports, C, H, W = left_viewports.shape
        all_bd_features = []
        all_bs_features = []
        
        # Process each viewport pair
        for i in range(num_viewports):
            left_vp = left_viewports[:, i]  # (batch, C, H, W)
            right_vp = right_viewports[:, i]
            
            # Extract features from ResNet stages
            left_feat0 = self.resnet_stage0(left_vp)
            left_feat1 = self.resnet_stage1(left_feat0)
            left_feat2 = self.resnet_stage2(left_feat1)
            left_feat3 = self.resnet_stage3(left_feat2)
            left_feat4 = self.resnet_stage4(left_feat3)
            
            right_feat0 = self.resnet_stage0(right_vp)
            right_feat1 = self.resnet_stage1(right_feat0)
            right_feat2 = self.resnet_stage2(right_feat1)
            right_feat3 = self.resnet_stage3(right_feat2)
            right_feat4 = self.resnet_stage4(right_feat3)
            
            # Multi-scale feature fusion
            bd_feat = None
            bs_feat = None
            
            # Stage 2
            bd_feat = self.bd_streams[0](left_feat2, right_feat2, None)
            bs_feat = self.bs_streams[0](left_feat2, right_feat2, None)
            
            # Stage 3
            bd_feat = self.bd_streams[1](left_feat3, right_feat3, bd_feat)
            bs_feat = self.bs_streams[1](left_feat3, right_feat3, bs_feat)
            
            # Stage 4
            bd_feat = self.bd_streams[2](left_feat4, right_feat4, bd_feat)
            bs_feat = self.bs_streams[2](left_feat4, right_feat4, bs_feat)
            
            all_bd_features.append(bd_feat)
            all_bs_features.append(bs_feat)
        
        # Concatenate features from all viewports
        bd_features = torch.stack(all_bd_features, dim=1)  # (batch, num_viewports, feature_dim)
        bs_features = torch.stack(all_bs_features, dim=1)  # (batch, num_viewports, feature_dim)
        
        # Average over viewports
        bd_features = bd_features.mean(dim=1)  # (batch, feature_dim)
        bs_features = bs_features.mean(dim=1)  # (batch, feature_dim)
        
        # Final projection
        bd_features = self.bd_fusion(bd_features)
        bs_features = self.bs_fusion(bs_features)
        
        return bd_features, bs_features