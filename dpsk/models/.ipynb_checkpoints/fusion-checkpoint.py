import torch
import torch.nn as nn

class FeatureFusion(nn.Module):
    def __init__(self, bd_dim, bs_dim, mf_dim, hidden_dim, output_dim=1):
        super(FeatureFusion, self).__init__()
        
        self.fusion_net = nn.Sequential(
            nn.Linear(bd_dim + bs_dim + mf_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, bd_features, bs_features, mf_features):
        # Concatenate all features
        fused_features = torch.cat([bd_features, bs_features, mf_features], dim=1)
        
        # Predict quality score
        quality_score = self.fusion_net(fused_features)
        
        return quality_score.squeeze()