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