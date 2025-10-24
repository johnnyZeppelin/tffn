import torch
import torch.nn as nn
from .tpf import TPFBlock
from .pdienet import PDIENet
from .fusion import FeatureFusion

class TFFN(nn.Module):
    def __init__(self, config):
        super(TFFN, self).__init__()
        self.config = config
        
        # Three branches
        self.tpf_block = TPFBlock()
        self.pdie_net = PDIENet(config.VMAMBA_PATH)
        self.fusion_block = FeatureFusion(
            bd_dim=512, 
            bs_dim=512, 
            mf_dim=512,
            hidden_dim=config.FF_HIDDEN_DIM
        )
        
    def forward(self, left_viewports, right_viewports, restored_viewports):
        # TPF Block: binocular features
        bd_features, bs_features = self.tpf_block(left_viewports, right_viewports)
        
        # PDIE Block: monocular features (using right view only as mentioned in paper)
        mf_features = self.pdie_net(right_viewports, restored_viewports)
        
        # Feature Fusion
        quality_score = self.fusion_block(bd_features, bs_features, mf_features)
        
        return quality_score