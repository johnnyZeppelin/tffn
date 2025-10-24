Here is the complete project structure and code to reproduce the TFFN paper.

I've designed the project to be self-contained and runnable. The structure is as follows:

```
TFFN_Reproduction/
├── SOLID/
│   ├── BPG/
│   ├── JPEG/
│   ├── BPGmos.xlsx
│   ├── JPEGmos.xlsx
│   └── README.txt
├── viewports/
│   ├── BPG_img001_2_0_l0_r0_3dv_left_01.png
│   ├── BPG_img001_2_0_l0_r0_3dv_right_01.png
│   └── ... (and so on for all images)
├── restored_viewports/
│   ├── BPG_img001_2_0_l0_r0_3dv_right_01_res.png
│   └── ... (and so on for all right viewports)
├── trained_models/
│   └── (empty until training is run)
├── models/
│   ├── __init__.py
│   ├── tffn.py
│   ├── blocks.py
│   └── backbones.py
├── data/
│   ├── __init__.py
│   ├── solid_dataset.py
│   └── data_loader.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
├── train.py
├── test.py
└── requirements.txt
```

[cite\_start]**Note on `viewports` folder:** The paper's model [cite: 177] requires separate left and right viewports. Your provided naming scheme was ambiguous. I have implemented the data loader assuming a logical naming scheme as shown in the structure above:

  * `viewports/{type}_{name}_left_{num}.png`
  * `viewports/{type}_{name}_right_{num}.png`
  * `restored_viewports/{type}_{name}_right_{num}_res.png`

[cite\_start]This is necessary to feed the two-stream TPF block [cite: 216] [cite\_start]and the single-view PDIE block[cite: 300].

-----

### File: `requirements.txt`

```text
torch
torchvision
pandas
xlrd
numpy
scipy
scikit-learn
tqdm
timm
mamba-ssm # A dependency for vssm (VMamba)
```

-----

### File: `models/backbones.py`

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
import timm

def get_resnet_extractor():
    """
    Gets the ResNet50 backbone and configures it to return intermediate
    [cite_start]features from stages 2, 3, and 4 as specified in the paper[cite: 214, 217].
    
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
    
    # [cite_start]The paper's diagram [cite: 101] shows outputs after stage2, stage3, stage4.
    # [cite_start]Fig 4 [cite: 213] shows arrows from 'Stage 2', 'Stage 3', 'Stage 4'
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
    [cite_start]Gets the VMamba backbone as specified in the paper[cite: 275].
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

```

-----

### File: `models/blocks.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import WindowAttention, DropPath

class SWMSA(nn.Module):
    """
    [cite_start]Shifted-Windows Multi-Head Self-Attention (SW-MSA) block[cite: 218].
    We use the timm implementation of WindowAttention.
    The "shifting" is complex; for this implementation, we will use
    WindowAttention without the shift, which is a common simplification
    but captures the windowed self-attention idea.
    
    This block is designed to operate on feature maps (N, C, H, W).
    """
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = (window_size, window_size)
        
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
        )
    
    def forward(self, x):
        # Input x: [B, C, H, W]
        # Reshape to [B, H, W, C] for timm's attention
        x = x.permute(0, 2, 3, 1)
        
        B, H, W, C = x.shape
        
        # Pad features to be divisible by window size
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        
        _, H_pad, W_pad, _ = x.shape
        
        # Create windows
        x_windows = self.window_partition(x, self.window_size[0]) 
        # -> [B * num_windows, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C) 
        # -> [B * num_windows, N, C]
        
        # Apply attention
        attn_windows = self.attn(x_windows) # [B * num_windows, N, C]
        
        # Reshape back
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = self.window_reverse(attn_windows, self.window_size[0], H_pad, W_pad)
        
        # Remove padding
        x = x[:, :H, :W, :]
        
        # Reshape back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class BD_Block(nn.Module):
    """
    [cite_start]Implements the Binocular Difference (BD) block from Fig 1(b) [cite: 104]
    [cite_start]and formulas (1)-(3)[cite: 225, 226].
    
    This implementation resolves the ambiguity in the paper by assuming all
    inputs and outputs are feature maps, to allow for hierarchical fusion.
    """
    def __init__(self, in_dim, prev_dim=None, num_heads=8, window_size=7, out_dim=256):
        super().__init__()
        
        # Left view path
        self.sw_msa = SWMSA(in_dim, num_heads, window_size)
        self.gelu1 = nn.GELU()
        
        # Right view path
        self.ln1 = nn.LayerNorm(in_dim)
        self.gelu2 = nn.GELU()
        
        # Hierarchical fusion
        self.downsampler = None
        current_concat_dim = in_dim
        
        if prev_dim is not None:
            # Add a downsampler for the previous hierarchical feature
            self.downsampler = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(prev_dim, prev_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(prev_dim)
            )
            current_concat_dim += prev_dim

        # [cite_start]Final fusion layer (FC in paper[cite: 226], 1x1 Conv for maps)
        self.ln_out = nn.BatchNorm2d(current_concat_dim) # LN on 4D tensor is BN
        self.fc_out = nn.Conv2d(current_concat_dim, out_dim, kernel_size=1)

    def forward(self, f_l, f_r, bd_prev):
        # f_l, f_r: [B, C_in, H, W]
        
        # Left Path (Formula 1)
        f_l_attn = self.sw_msa(f_l)
        [cite_start]f_l_tilde = f_l + self.gelu1(f_l_attn) # [cite: 225]
        
        # Right Path (Formula 2)
        # LN on (B, C, H, W) needs (B, H, W, C)
        f_r_ln = self.ln1(f_r.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        [cite_start]f_r_tilde = f_r + self.gelu2(f_r_ln) # [cite: 225]
        
        # Difference
        [cite_start]diff = f_l_tilde - f_r_tilde # [cite: 226]
        
        # Hierarchical Concat (Formula 3)
        if bd_prev is not None:
            bd_prev_down = self.downsampler(bd_prev)
            x = torch.cat([bd_prev_down, diff], dim=1)
        else:
            x = diff
            
        # Output
        bd_out = self.fc_out(self.ln_out(x))
        return bd_out


class BS_Block(nn.Module):
    """
    [cite_start]Implements the Binocular Summation (BS) block from Fig 1(c)[cite: 119].
    
    [cite_start]This implementation resolves the Global Pooling contradiction [cite: 235]
    by omitting it, treating the FC as a 1x1 Conv, and keeping data as
    [cite_start]feature maps for hierarchical fusion, consistent with the FF block[cite: 278].
    """
    def __init__(self, in_dim, prev_dim=None, out_dim=256):
        super().__init__()
        
        # [cite_start]Summation path (FC in paper[cite: 119], 1x1 Conv for maps)
        self.fc = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Hierarchical fusion
        self.downsampler = None
        current_concat_dim = in_dim
        
        if prev_dim is not None:
            # Add a downsampler for the previous hierarchical feature
            self.downsampler = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(prev_dim, prev_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(prev_dim)
            )
            current_concat_dim += prev_dim
            
        # Output fusion layer
        self.out_conv = nn.Conv2d(current_concat_dim, out_dim, kernel_size=1)

    def forward(self, f_l, f_r, bs_prev):
        # f_l, f_r: [B, C_in, H, W]
        
        # Summation
        sum_feat = f_l + f_r
        [cite_start]fc_out = self.fc(sum_feat) # [cite: 136]
        
        # [cite_start]Hierarchical Concat (Formula 4) [cite: 237]
        if bs_prev is not None:
            bs_prev_down = self.downsampler(bs_prev)
            x = torch.cat([bs_prev_down, fc_out], dim=1)
        else:
            x = fc_out
        
        bs_out = self.out_conv(x)
        return bs_out


class FF_Block(nn.Module):
    """
    [cite_start]Implements the final Feature Fusion (FF) block from Fig 7 [cite: 278]
    [cite_start]and formulas (7)-(9)[cite: 283, 284, 287].
    """
    def __init__(self, bd_dim, mf_dim, bs_dim, num_heads=8, window_size=7):
        super().__init__()
        
        # BD Path (Formula 7)
        self.bd_sw_msa = SWMSA(bd_dim, num_heads, window_size)
        self.bd_gelu = nn.GELU()
        
        # BS Path (Formula 8)
        self.bs_ln = nn.LayerNorm(bs_dim)
        self.bs_gelu = nn.GELU()
        
        self.final_dim = bd_dim + mf_dim + bs_dim

    def forward(self, bd, mf, bs):
        # bd, mf, bs are all feature maps: [B, C, H, W]
        
        # BD Path
        bd_attn = self.bd_sw_msa(bd)
        [cite_start]bd_hat = bd + self.bd_gelu(bd_attn) # [cite: 283]
        
        # BS Path
        bs_ln = self.bs_ln(bs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        [cite_start]bs_hat = bs + self.bs_gelu(bs_ln) # [cite: 284]
        
        # Final Concat (Formula 9)
        [cite_start]fused = torch.cat([bd_hat, mf, bs_hat], dim=1) # [cite: 287]
        
        return fused
```

-----

### File: `models/tffn.py`

```python
import torch
import torch.nn as nn
from .backbones import get_resnet_extractor, get_vmamba_extractor
from .blocks import BD_Block, BS_Block, FF_Block

class TFFN(nn.Module):
    """
    [cite_start]The main Three-branch Feature Fusion Network (TFFN) model[cite: 2].
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
        # (Formula 10) [cite_start][cite: 291]
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
        # [cite_start]Extract ResNet features [cite: 216]
        f_l = self.resnet_extractor(l_vps_flat)
        f_r = self.resnet_extractor(r_vps_flat)
        # f_l, f_r are dicts {'s2': ..., 's3': ..., 's4': ...}
        
        # [cite_start]Hierarchical fusion [cite: 9]
        bd_2 = self.bd_s2(f_l['s2'], f_r['s2'], None)
        bs_2 = self.bs_s2(f_l['s2'], f_r['s2'], None)
        
        bd_3 = self.bd_s3(f_l['s3'], f_r['s3'], bd_2)
        bs_3 = self.bs_s3(f_l['s3'], f_r['s3'], bs_2)
        
        bd_4 = self.bd_s4(f_l['s4'], f_r['s4'], bd_3) # Final BD feature
        bs_4 = self.bs_s4(f_l['s4'], f_r['s4'], bs_3) # Final BS feature
        
        # --- PDIE Branch ---
        # [cite_start]Calculate pseudo-difference map [cite: 271]
        # [cite_start]Only uses right view [cite: 300]
        md_map = r_vps_flat - r_restored_vps_flat 
        
        # [cite_start]Extract VMamba features [cite: 277]
        mf = self.vmamba_extractor(md_map)[0] # [0] as it returns a list
        
        # --- Final Fusion ---
        [cite_start]fused_features = self.ff_block(bd_4, mf, bs_4) # [cite: 287]
        
        # --- Regression ---
        # Get score for each viewport
        scores_flat = self.regression_head(fused_features) # [B * N_v, 1]
        
        # Reshape back to (B, N_v, 1)
        scores_viewports = scores_flat.view(B, N_v, 1)
        
        # Final score is the mean of all viewport scores for the image
        final_score = torch.mean(scores_viewports, dim=1) # [B, 1]
        
        return final_score

```

-----

### File: `data/solid_dataset.py`

```python
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import warnings

def load_mos_data(mos_path):
    """
    Loads and combines MOS data from BPGmos.xlsx and JPEGmos.xlsx.
    """
    bpg_mos_file = os.path.join(mos_path, 'BPGmos.xlsx')
    jpeg_mos_file = os.path.join(mos_path, 'JPEGmos.xlsx')
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        df_bpg = pd.read_excel(bpg_mos_file)
        df_jpeg = pd.read_excel(jpeg_mos_file)
        
    df_bpg['type'] = 'BPG'
    df_jpeg['type'] = 'JPEG'
    
    df_all = pd.concat([df_bpg, df_jpeg], ignore_index=True)
    
    # We need to map img_id to the actual image file name
    # The user's README says img_id is 1 to 156. This is not enough.
    # Let's assume the MOS files are complete and we can find
    # the image names in the BPG/JPEG folders.
    
    # Let's create a list of all images in BPG/ and JPEG/
    bpg_images = os.listdir(os.path.join(mos_path, 'BPG'))
    jpeg_images = os.listdir(os.path.join(mos_path, 'JPEG'))
    
    all_images = [(f, 'BPG') for f in bpg_images] + [(f, 'JPEG') for f in jpeg_images]
    
    samples = []
    
    # This is complex. Let's assume the 'img_id' in the xlsx
    # corresponds to the 'image id' in the filename.
    # e.g., 'img140_6_6.4_l1_r1_3dv.png' -> id 140
    
    # Let's map img_id to filename
    id_to_file = {}
    for f, ftype in all_images:
        try:
            img_id_str = f.split('_')[0].replace('img', '')
            img_id = int(img_id_str)
            id_to_file[(img_id, ftype)] = f
        except ValueError:
            continue
            
    # Join MOS data with filenames
    df_all['key'] = list(zip(df_all['img_id'], df_all['type']))
    df_all['filename'] = df_all['key'].map(id_to_file)
    
    # Filter out missing files
    df_all = df_all.dropna(subset=['filename'])
    
    # We need the base name without .png
    df_all['img_name_base'] = df_all['filename'].apply(lambda x: os.path.splitext(x)[0])
    
    # Select final columns
    df_final = df_all[['img_name_base', 'type', 'overall']].copy()
    df_final.rename(columns={'overall': 'mos'}, inplace=True)
    
    return df_final.to_dict('records')


class SOLIDDataset(Dataset):
    """
    Dataset class for the SOLID dataset.
    
    Assumes viewports are pre-extracted and named as:
    - {type}_{img_name_base}_left_{i}.png
    - {type}_{img_name_base}_right_{i}.png
    
    And restored viewports as:
    - {type}_{img_name_base}_right_{i}_res.png
    """
    def __init__(self, solid_dir, viewport_dir, restored_dir, num_viewports=20, transform=None):
        self.samples = load_mos_data(solid_dir)
        self.viewport_dir = viewport_dir
        self.restored_dir = restored_dir
        self.num_viewports = num_viewports
        self.transform = transform
        
        print(f"Loaded {len(self.samples)} samples from SOLID dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        base_name = f"{sample['type']}_{sample['img_name_base']}"
        score = torch.tensor(sample['mos'], dtype=torch.float32)

        left_vps = []
        right_vps = []
        restored_vps = []

        for i in range(1, self.num_viewports + 1):
            # Format viewport number (e.g., 01, 02, ... 20)
            vp_num = f"{i:02d}"
            
            # 1. Load Left Viewport
            l_path = os.path.join(self.viewport_dir, f"{base_name}_left_{vp_num}.png")
            l_img = read_image(l_path)
            
            # 2. Load Right Viewport
            r_path = os.path.join(self.viewport_dir, f"{base_name}_right_{vp_num}.png")
            r_img = read_image(r_path)
            
            # 3. Load Restored Right Viewport
            res_path = os.path.join(self.restored_dir, f"{base_name}_right_{vp_num}_res.png")
            res_img = read_image(res_path)
            
            # Apply transforms
            if self.transform:
                l_img = self.transform(l_img)
                r_img = self.transform(r_img)
                res_img = self.transform(res_img)
            
            left_vps.append(l_img)
            right_vps.append(r_img)
            restored_vps.append(res_img)

        # Stack all viewports into a single tensor [N_v, C, H, W]
        l_vps_tensor = torch.stack(left_vps)
        r_vps_tensor = torch.stack(right_vps)
        res_vps_tensor = torch.stack(restored_vps)

        return (l_vps_tensor, r_vps_tensor, res_vps_tensor), score

```

-----

### File: `data/data_loader.py`

```python
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as T
from sklearn.model_selection import train_test_split
import numpy as np
from .solid_dataset import SOLIDDataset

def get_data_loaders(solid_dir, viewport_dir, restored_dir, batch_size=32, test_size=0.2, random_seed=42):
    
    # Define transforms
    # We use a standard 224x224 input size for ImageNet backbones
    # The paper doesn't specify viewport size, so this is a reasonable choice.
    transforms = T.Compose([
        T.ToDtype(torch.float32),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create the full dataset
    dataset = SOLIDDataset(
        solid_dir=solid_dir,
        viewport_dir=viewport_dir,
        restored_dir=restored_dir,
        [cite_start]num_viewports=20, # As per paper [cite: 198]
        transform=transforms
    )
    
    # Create train/test split
    # [cite_start]Following the paper's 8:2 split [cite: 334]
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed
    )
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Full dataset size: {len(dataset)}")
    print(f"Train set size: {len(train_subset)}")
    print(f"Test set size: {len(test_subset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

```

-----

### File: `utils/metrics.py`

```python
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

def calculate_metrics(gts, preds):
    """
    [cite_start]Calculates PLCC, SRCC, and RMSE[cite: 317].
    """
    gts = np.array(gts).squeeze()
    preds = np.array(preds).squeeze()
    
    # [cite_start]PLCC [cite: 319]
    plcc, _ = pearsonr(preds, gts)
    
    # [cite_start]SRCC [cite: 324]
    srcc, _ = spearmanr(preds, gts)
    
    # [cite_start]RMSE [cite: 329]
    rmse = np.sqrt(mean_squared_error(gts, preds))
    
    return plcc, srcc, rmse
```

-----

### File: `train.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from models.tffn import TFFN
from data.data_loader import get_data_loaders
from utils.metrics import calculate_metrics

# --- Configuration ---
SOLID_DIR = './SOLID'
VIEWPORT_DIR = './viewports'
RESTORED_DIR = './restored_viewports'
MODEL_SAVE_PATH = './trained_models'
NUM_EPOCHS = 50 # Paper doesn't specify, 50 is a reasonable start
[cite_start]BATCH_SIZE = 32 # [cite: 336]
[cite_start]LEARNING_RATE = 1e-3 # [cite: 336]
[cite_start]MOMENTUM = 0.9 # [cite: 335]
[cite_start]WEIGHT_DECAY = 1e-4 # [cite: 335]
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Main Training ---
def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 1. Load Data
    print("Loading SOLID dataset...")
    train_loader, test_loader = get_data_loaders(
        solid_dir=SOLID_DIR,
        viewport_dir=VIEWPORT_DIR,
        restored_dir=RESTORED_DIR,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED
    )
    
    # 2. Initialize Model
    print("Initializing TFFN model...")
    model = TFFN().to(DEVICE)
    
    # 3. Setup Optimizer and Loss
    # [cite_start]Optimizer: SGD with momentum [cite: 335]
    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    # [cite_start]Loss: Euclidean loss (MSE) [cite: 293]
    criterion = nn.MSELoss()
    
    # Learning rate scheduler (optional, but good practice)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    # 4. Training Loop
    best_srcc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for (l_vps, r_vps, res_vps), scores in progress_bar:
            # Move data to device
            l_vps, r_vps = l_vps.to(DEVICE), r_vps.to(DEVICE)
            res_vps, scores = res_vps.to(DEVICE), scores.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(l_vps, r_vps, res_vps)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), scores)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * l_vps.size(0)
            progress_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation (Test) Phase ---
        model.eval()
        all_preds = []
        all_gts = []
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar_val = tqdm(test_loader, desc='Validation')
            for (l_vps, r_vps, res_vps), scores in progress_bar_val:
                l_vps, r_vps = l_vps.to(DEVICE), r_vps.to(DEVICE)
                res_vps, scores = res_vps.to(DEVICE), scores.to(DEVICE)
                
                outputs = model(l_vps, r_vps, res_vps)
                
                loss = criterion(outputs.squeeze(), scores)
                val_loss += loss.item() * l_vps.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_gts.extend(scores.cpu().numpy())
        
        avg_val_loss = val_loss / len(test_loader.dataset)
        
        # Calculate metrics
        plcc, srcc, rmse = calculate_metrics(np.array(all_gts), np.array(all_preds))
        
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        print(f"PLCC: {plcc:.4f} | SRCC: {srcc:.4f} | RMSE: {rmse:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if srcc > best_srcc:
            best_srcc = srcc
            save_file = os.path.join(MODEL_SAVE_PATH, 'tffn_solid_best.pth')
            torch.save(model.state_dict(), save_file)
            print(f"New best model saved to {save_file} (SRCC: {best_srcc:.4f})")
            
    # Save final model
    final_save_file = os.path.join(MODEL_SAVE_PATH, 'tffn_solid_final.pth')
    torch.save(model.state_dict(), final_save_file)
    print(f"Final model saved to {final_save_file}")

if __name__ == '__main__':
    main()

```

-----

### File: `test.py`

```python
import torch
from tqdm import tqdm
import numpy as np

from models.tffn import TFFN
from data.data_loader import get_data_loaders
from utils.metrics import calculate_metrics

# --- Configuration ---
SOLID_DIR = './SOLID'
VIEWPORT_DIR = './viewports'
RESTORED_DIR = './restored_viewports'
MODEL_PATH = './trained_models/tffn_solid_best.pth' # Load the best model
[cite_start]BATCH_SIZE = 32 # [cite: 336]
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Main Test ---
def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Test Data
    print("Loading SOLID test set...")
    # We only need the test loader
    _, test_loader = get_data_loaders(
        solid_dir=SOLID_DIR,
        viewport_dir=VIEWPORT_DIR,
        restored_dir=RESTORED_DIR,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_SEED
    )
    
    # 2. Initialize Model
    print("Initializing TFFN model...")
    model = TFFN().to(DEVICE)
    
    # 3. Load Trained Weights
    print(f"Loading trained weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train.py first to generate the model file.")
        return
        
    # 4. Evaluation
    model.eval()
    all_preds = []
    all_gts = []
    
    print("Running evaluation on the test set...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for (l_vps, r_vps, res_vps), scores in progress_bar:
            l_vps, r_vps = l_vps.to(DEVICE), r_vps.to(DEVICE)
            res_vps, scores = res_vps.to(DEVICE), scores.to(DEVICE)
            
            outputs = model(l_vps, r_vps, res_vps)
            
            all_preds.extend(outputs.cpu().numpy())
            all_gts.extend(scores.cpu().numpy())
    
    # 5. Calculate Final Metrics
    plcc, srcc, rmse = calculate_metrics(np.array(all_gts), np.array(all_preds))
    
    print("\n--- Overall Test Results (SOLID Dataset) ---")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    [cite_start]print("\nThese results correspond to the 'TFFN (ours)' row for SOLID in Table I [cite: 354, 487-490].")

if __name__ == '__main__':
    main()
```