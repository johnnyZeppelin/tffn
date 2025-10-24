import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
import timm

def get_resnet_extractor():
    """
    Gets the ResNet50 backbone and configures it to return intermediate
    features from stages 2, 3, and 4 as specified in the paper[cite: 214, 217].
    
    Paper's 'stage 2' = torchvision's 'layer2'
    Paper's 'stage 3' = torchvision's 'layer3'
    Paper's 'stage 4' = torchvision's 'layer4'
    """
    # Load pretrained ResNet50
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    
    # We don't need the final fc layer
    body = nn.Sequential(*list(m.children())[:-2])
    
    # Extract features from 'layer2', 'layer3', and 'layer4'
    return_layers = {
        '5': 's2', # 'layer2' is the 5th module in body
        '6': 's3', # 'layer3' is the 6th module
        '7': 's4', # 'layer4' is the 7th module
    }
    
    # Create the feature extractor
    feature_extractor = IntermediateLayerGetter(body, return_layers=return_layers)
    
    # This will return a dict: {'s2': ..., 's3': ..., 's4': ...}
    # Feature map sizes (for 224x224 input):
    # s2: [B, 512, 28, 28]  (Paper calls this H/8, but ResNet default is H/16)
    # s3: [B, 1024, 14, 14] (H/32)
    # s4: [B, 2048, 7, 7]   (H/64)
    # Let's adjust return_layers to match the paper's diagram better (s2, s3, s4)
    # Let's use layer1, layer2, layer3 as s2, s3, s4
    
    body_new = nn.Sequential(
        m.conv1, m.bn1, m.relu, m.maxpool, # stage 0, 1
        m.layer1, # stage 2 (torchvision) -> s2 (paper)
        m.layer2, # stage 3 (torchvision) -> s3 (paper)
        m.layer3  # stage 4 (torchvision) -> s4 (paper)
    )
    
    # The paper's diagram [cite: 101] shows outputs after stage2, stage3, stage4.
    # Fig 4 [cite: 213] shows arrows from 'Stage 2', 'Stage 3', 'Stage 4'
    # These correspond to ResNet's layer2, layer3, and layer4.
    
    return_layers_paper = {
        'layer2': 's2',
        'layer3': 's3',
        'layer4': 's4',
    }
    
    # We use the full m (model) not just body
    feature_extractor = IntermediateLayerGetter(m, return_layers=return_layers_paper)
    
    # Feature map sizes (for 224x224 input):
    # s2 (layer2): [B, 512, 28, 28]
    # s3 (layer3): [B, 1024, 14, 14]
    # s4 (layer4): [B, 2048, 7, 7]
    return feature_extractor

def get_vmamba_extractor():
    """
    Gets the VMamba backbone as specified in the paper[cite: 275].
    We use 'vssm_small' from timm, which is a VMamba implementation.
    We load pretrained weights and configure it to return features compatible
    with the ResNet s4 output (e.g., [B, C, 7, 7]).
    """
    # Create the vssm_small model, pretrained
    # We will not freeze it, as requested by the user.
    vmamba = timm.create_model(
        'vssm_small', 
        pretrained=True, 
        features_only=True, # Return intermediate features
        out_indices=(3,) # We only want the last stage's output
    )
    
    # This will return a list of features. With out_indices=(3,),
    # it will return [features_stage_3].
    # Output shape for 224x224 input: [B, 768, 7, 7]
    # This is compatible with ResNet's s4 output (spatial size 7x7).
    return vmamba

if __name__ == '__main__':
    # Test backbones
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Test ResNet
    resnet_ext = get_resnet_extractor()
    resnet_feats = resnet_ext(dummy_input)
    print("ResNet Extractor Output Shapes:")
    for k, v in resnet_feats.items():
        print(f"{k}: {v.shape}")
        
    # Test VMamba
    vmamba_ext = get_vmamba_extractor()
    vmamba_feat = vmamba_ext(dummy_input)
    print("\nVMamba Extractor Output Shape:")
    print(f"s4-compatible: {vmamba_feat[0].shape}") # It returns a list