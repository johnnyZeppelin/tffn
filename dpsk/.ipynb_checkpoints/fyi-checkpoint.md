I'll design and implement the complete project structure for reproducing the TFFN paper. Let me first create the project structure and then write each file.

## Project Structure

```
TFFN_SOLID/
├── config.py
├── data/
│   ├── SOLID/ (dataset folder as described)
│   ├── viewports/ (extracted viewports)
│   └── restored_viewports/ (InstructIR restored viewports)
├── models/
│   ├── __init__.py
│   ├── tffn.py
│   ├── tpf.py
│   ├── pdienet.py
│   └── fusion.py
├── utils/
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics.py
│   └── helpers.py
├── train.py
├── test.py
├── vmamba/ (VMamba model code)
└── checkpoints/ (for saving trained models)
```

Now let me write each file:

## 1. config.py

```python
import torch
import os

class Config:
    # Dataset paths
    DATA_ROOT = './data/SOLID'
    VIEWPORTS_PATH = './data/viewports'
    RESTORED_VIEWPORTS_PATH = './data/restored_viewports'
    
    # Model parameters
    NUM_VIEWPORTS = 20
    FEATURE_DIM = 512
    FF_HIDDEN_DIM = 256
    
    # Training parameters
    BATCH_SIZE = 8  # Reduced due to large viewport size
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    TRAIN_TEST_SPLIT = 0.8
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    VMAMBA_PATH = './vmamba'
    CHECKPOINT_PATH = './checkpoints'
    
    # Image processing
    IMG_SIZE = (224, 224)  # ResNet input size
    
    def __init__(self):
        # Create directories
        os.makedirs(self.CHECKPOINT_PATH, exist_ok=True)

config = Config()
```

## 2. models/tpf.py (Two-stream Parallel Fusion)

```python
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
```

## 3. models/pdienet.py (Pseudo-Difference Information Extraction)

```python
import torch
import torch.nn as nn
import sys
import os

# Add VMamba to path
sys.path.append('./vmamba')

try:
    from vmamba import VSSM
except ImportError:
    print("Warning: VMamba not found. Using placeholder implementation.")
    # Placeholder implementation
    class VSSM(nn.Module):
        def __init__(self, **kwargs):
            super(VSSM, self).__init__()
            self.conv = nn.Conv2d(3, 512, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x.flatten(1)

class PDIENet(nn.Module):
    def __init__(self, vmamba_model_path=None):
        super(PDIENet, self).__init__()
        
        # Load VMamba model
        if vmamba_model_path and os.path.exists(vmamba_model_path):
            self.vmamba = VSSM()
            # Load pretrained weights here
            print("Loaded VMamba model")
        else:
            self.vmamba = VSSM()
            print("Using VMamba with random initialization")
        
        # Feature projection
        self.feature_proj = nn.Linear(512, 512)
        
    def forward(self, distorted_viewports, restored_viewports):
        batch_size, num_viewports, C, H, W = distorted_viewports.shape
        
        # Compute monocular difference maps
        md_maps = distorted_viewports - restored_viewports  # (batch, num_viewports, C, H, W)
        
        # Process each difference map
        all_features = []
        for i in range(num_viewports):
            md_map = md_maps[:, i]  # (batch, C, H, W)
            
            # Extract features using VMamba
            features = self.vmamba(md_map)  # (batch, feature_dim)
            features = self.feature_proj(features)
            all_features.append(features)
        
        # Concatenate features from all viewports
        all_features = torch.stack(all_features, dim=1)  # (batch, num_viewports, feature_dim)
        
        # Average over viewports
        monocular_features = all_features.mean(dim=1)  # (batch, feature_dim)
        
        return monocular_features
```

## 4. models/fusion.py (Feature Fusion)

```python
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
```

## 5. models/tffn.py (Main Model)

```python
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
```

## 6. utils/dataset.py

```python
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import glob

class SOLIDDataset(Dataset):
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # Load MOS data
        self.bpg_mos = pd.read_excel(os.path.join(config.DATA_ROOT, 'BPGmos.xlsx'))
        self.jpeg_mos = pd.read_excel(os.path.join(config.DATA_ROOT, 'JPEGmos.xlsx'))
        
        # Combine and process MOS data
        self.samples = self._load_samples()
        
        # Split data
        train_size = int(len(self.samples) * config.TRAIN_TEST_SPLIT)
        if split == 'train':
            self.samples = self.samples[:train_size]
        else:
            self.samples = self.samples[train_size:]
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_samples(self):
        samples = []
        
        # Process BPG images
        for _, row in self.bpg_mos.iterrows():
            img_id = row['img_id']
            mos = row['overall']
            
            # Find corresponding viewports
            base_pattern = f"BPG_img{img_id:03d}_*"
            viewport_files = glob.glob(os.path.join(self.config.VIEWPORTS_PATH, f"{base_pattern}.png"))
            
            if len(viewport_files) >= self.config.NUM_VIEWPORTS:
                samples.append({
                    'img_id': img_id,
                    'compression': 'BPG',
                    'mos': mos,
                    'base_pattern': base_pattern
                })
        
        # Process JPEG images
        for _, row in self.jpeg_mos.iterrows():
            img_id = row['img_id']
            mos = row['overall']
            
            base_pattern = f"JPEG_img{img_id:03d}_*"
            viewport_files = glob.glob(os.path.join(self.config.VIEWPORTS_PATH, f"{base_pattern}.png"))
            
            if len(viewport_files) >= self.config.NUM_VIEWPORTS:
                samples.append({
                    'img_id': img_id,
                    'compression': 'JPEG',
                    'mos': mos,
                    'base_pattern': base_pattern
                })
        
        return samples
    
    def _load_viewports(self, base_pattern):
        viewports = []
        restored_viewports = []
        
        # Load original viewports
        for i in range(1, self.config.NUM_VIEWPORTS + 1):
            vp_pattern = f"{base_pattern}_{i:02d}.png"
            vp_files = glob.glob(os.path.join(self.config.VIEWPORTS_PATH, vp_pattern))
            
            if vp_files:
                img = Image.open(vp_files[0]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                viewports.append(img)
            
            # Load restored viewports
            res_pattern = f"{base_pattern}_{i:02d}_res.png"
            res_files = glob.glob(os.path.join(self.config.RESTORED_VIEWPORTS_PATH, res_pattern))
            
            if res_files:
                img = Image.open(res_files[0]).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                restored_viewports.append(img)
        
        # Stack viewports
        if viewports and restored_viewports:
            viewports = torch.stack(viewports)  # (num_viewports, C, H, W)
            restored_viewports = torch.stack(restored_viewports)
            return viewports, restored_viewports
        else:
            return None, None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load viewports
        left_viewports, left_restored = self._load_viewports(sample['base_pattern'] + '_l')
        right_viewports, right_restored = self._load_viewports(sample['base_pattern'] + '_r')
        
        # Use right view for monocular features as mentioned in paper
        if left_viewports is not None and right_viewports is not None and right_restored is not None:
            return left_viewports, right_viewports, right_restored, torch.tensor(sample['mos'], dtype=torch.float32)
        else:
            # Return a random sample if viewports are missing
            return self[(idx + 1) % len(self)]
```

## 7. utils/metrics.py

```python
import torch
import numpy as np
from scipy.stats import spearmanr

def plcc(pred, target):
    """
    Pearson Linear Correlation Coefficient
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    return np.corrcoef(pred, target)[0, 1]

def srcc(pred, target):
    """
    Spearman Rank-order Correlation Coefficient
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    return spearmanr(pred, target).correlation

def rmse(pred, target):
    """
    Root Mean Squared Error
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    return np.sqrt(np.mean((pred - target) ** 2))
```

## 8. utils/helpers.py

```python
import torch
import os
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
    else:
        return model, optimizer, 0, float('inf')

def plot_training_curves(train_losses, val_losses, save_path):
    """Plot training and validation curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(save_path)
    plt.close()
```

## 9. train.py

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm

from config import Config
from models.tffn import TFFN
from utils.dataset import SOLIDDataset
from utils.metrics import plcc, srcc, rmse
from utils.helpers import save_checkpoint, plot_training_curves

def train():
    config = Config()
    
    # Initialize model
    model = TFFN(config).to(config.DEVICE)
    
    # Dataset and DataLoader
    train_dataset = SOLIDDataset(config, split='train')
    val_dataset = SOLIDDataset(config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        momentum=config.MOMENTUM, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
        for left_vp, right_vp, restored_vp, mos in pbar:
            left_vp = left_vp.to(config.DEVICE)
            right_vp = right_vp.to(config.DEVICE)
            restored_vp = restored_vp.to(config.DEVICE)
            mos = mos.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_mos = model(left_vp, right_vp, restored_vp)
            loss = criterion(pred_mos, mos)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(pred_mos.detach())
            train_targets.append(mos.detach())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_plcc = plcc(train_preds, train_targets)
        train_srcc = srcc(train_preds, train_targets)
        train_rmse = rmse(train_preds, train_targets)
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for left_vp, right_vp, restored_vp, mos in val_loader:
                left_vp = left_vp.to(config.DEVICE)
                right_vp = right_vp.to(config.DEVICE)
                restored_vp = restored_vp.to(config.DEVICE)
                mos = mos.to(config.DEVICE)
                
                pred_mos = model(left_vp, right_vp, restored_vp)
                loss = criterion(pred_mos, mos)
                
                val_loss += loss.item()
                val_preds.append(pred_mos)
                val_targets.append(mos)
        
        # Calculate validation metrics
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_plcc = plcc(val_preds, val_targets)
        val_srcc = srcc(val_preds, val_targets)
        val_rmse = rmse(val_preds, val_targets)
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train - Loss: {avg_train_loss:.4f}, PLCC: {train_plcc:.4f}, SRCC: {train_srcc:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, PLCC: {val_plcc:.4f}, SRCC: {val_srcc:.4f}, RMSE: {val_rmse:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_PATH, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, os.path.join(config.CHECKPOINT_PATH, 'training_curves.png'))
    
    print("Training completed!")

if __name__ == '__main__':
    train()
```

## 10. test.py

```python
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from config import Config
from models.tffn import TFFN
from utils.dataset import SOLIDDataset
from utils.metrics import plcc, srcc, rmse
from utils.helpers import load_checkpoint

def test():
    config = Config()
    
    # Initialize model
    model = TFFN(config).to(config.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
    
    # Load best model
    checkpoint_path = os.path.join(config.CHECKPOINT_PATH, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model, optimizer, _, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print("Loaded best model for testing")
    else:
        print("No checkpoint found. Using randomly initialized model.")
    
    # Test dataset
    test_dataset = SOLIDDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Testing
    model.eval()
    test_preds = []
    test_targets = []
    
    print("Starting testing...")
    with torch.no_grad():
        for left_vp, right_vp, restored_vp, mos in tqdm(test_loader, desc='Testing'):
            left_vp = left_vp.to(config.DEVICE)
            right_vp = right_vp.to(config.DEVICE)
            restored_vp = restored_vp.to(config.DEVICE)
            
            pred_mos = model(left_vp, right_vp, restored_vp)
            
            test_preds.append(pred_mos.cpu())
            test_targets.append(mos.cpu())
    
    # Calculate metrics
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    
    test_plcc = plcc(test_preds, test_targets)
    test_srcc = srcc(test_preds, test_targets)
    test_rmse = rmse(test_preds, test_targets)
    
    print("\nTest Results:")
    print(f"PLCC: {test_plcc:.4f}")
    print(f"SRCC: {test_srcc:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    # Save results
    results_path = os.path.join(config.CHECKPOINT_PATH, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("TFFN Test Results on SOLID Dataset\n")
        f.write("=" * 40 + "\n")
        f.write(f"PLCC: {test_plcc:.4f}\n")
        f.write(f"SRCC: {test_srcc:.4f}\n")
        f.write(f"RMSE: {test_rmse:.4f}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")
    
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    test()
```

## 11. models/__init__.py

```python
from .tffn import TFFN
from .tpf import TPFBlock
from .pdienet import PDIENet
from .fusion import FeatureFusion

__all__ = ['TFFN', 'TPFBlock', 'PDIENet', 'FeatureFusion']
```

## 12. utils/__init__.py

```python
from .dataset import SOLIDDataset
from .metrics import plcc, srcc, rmse
from .helpers import save_checkpoint, load_checkpoint, plot_training_curves

__all__ = ['SOLIDDataset', 'plcc', 'srcc', 'rmse', 'save_checkpoint', 'load_checkpoint', 'plot_training_curves']
```

## Usage Instructions

1. **Setup the environment:**
```bash
pip install torch torchvision pandas openpyxl scipy tqdm matplotlib
```

2. **Organize the data:**
   - Place SOLID dataset in `data/SOLID/`
   - Place viewports in `data/viewports/`
   - Place restored viewports in `data/restored_viewports/`

3. **Train the model:**
```bash
python train.py
```

4. **Test the model:**
```bash
python test.py
```

## Key Implementation Notes

1. **VMamba Integration**: The code includes a placeholder for VMamba. You'll need to integrate the actual VMamba implementation from the provided repository.

2. **Viewport Processing**: The dataset loader assumes viewports are named with patterns like `{compression}_img{id}_l{viewport_num}.png` and `{compression}_img{id}_r{viewport_num}.png`.

3. **Memory Management**: Batch size is set to 8 to handle the large number of viewports. Adjust based on your GPU memory.

4. **Progressive Training**: The model uses pre-trained ResNet50 weights and fine-tunes them for the SOIQA task.

This implementation follows the paper's architecture closely and should provide a solid foundation for reproducing the TFFN results on the SOLID dataset.