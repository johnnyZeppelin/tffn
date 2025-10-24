import os
import re
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from config import *

# -------------------------- MOS Loader --------------------------
def load_mos_data():
    """Load MOS from BPGmos.xlsx and JPEGmos.xlsx, return {img_id: overall_mos}"""
    # Load BPG MOS
    bpg_mos_df = pd.read_excel(BPG_MOS_PATH)
    bpg_mos = dict(zip(bpg_mos_df["img_id"], bpg_mos_df["overall"]))
    
    # Load JPEG MOS
    jpeg_mos_df = pd.read_excel(JPEG_MOS_PATH)
    jpeg_mos = dict(zip(jpeg_mos_df["img_id"], jpeg_mos_df["overall"]))
    
    return {"BPG": bpg_mos, "JPEG": jpeg_mos}

# -------------------------- Viewport Parser --------------------------
def parse_viewport_filename(filename):
    """Parse viewport filename to get: compress_type, img_id, view_type (l/r), port_num"""
    # Example filename: BPG_img001_2_0_l0_r0_3dv_02.png
    pattern = r"^(\w+)_img(\d+)_.*_l(\d+)_r(\d+)_3dv_(\d+)\.png$"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid viewport filename: {filename}")
    
    compress_type = match.group(1)  # BPG/JPEG
    img_id = int(match.group(2))    # img001 → 1
    l_level = match.group(3)        # Left view level (l0/l1)
    r_level = match.group(4)        # Right view level (r0/r1)
    port_num = int(match.group(5))  # Viewport number (00-19)
    
    return {
        "compress_type": compress_type,
        "img_id": img_id,
        "left_view": f"l{l_level}",
        "right_view": f"r{r_level}",
        "port_num": port_num,
        "base_img_name": f"img{img_id:03d}_*_l{l_level}_r{r_level}_3dv.png"  # Base image name
    }

# -------------------------- Image Transforms --------------------------
def get_image_transform():
    """Image preprocessing pipeline"""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])

# -------------------------- SOLID Dataset --------------------------
class SOLIDDataset(Dataset):
    def __init__(self, viewport_files, restored_viewport_files, mos_dict, transform):
        self.viewport_files = viewport_files  # List of viewport paths
        self.restored_viewport_files = restored_viewport_files  # List of restored viewport paths
        self.mos_dict = mos_dict  # {compress_type: {img_id: mos}}
        self.transform = transform
        
        # Group viewports by base image (1 image → 20 viewports)
        self.image_groups = self._group_viewports_by_image()

    def _group_viewports_by_image(self):
        """Group viewports into {base_img_key: (left_ports, right_ports, restored_right_ports, mos)}"""
        image_groups = {}
        
        for vp_path in self.viewport_files:
            vp_filename = os.path.basename(vp_path)
            parsed = parse_viewport_filename(vp_filename)
            
            # Base key: (compress_type, img_id, left_view, right_view)
            base_key = (parsed["compress_type"], parsed["img_id"], parsed["left_view"], parsed["right_view"])
            
            # Get MOS for this image
            mos = self.mos_dict[parsed["compress_type"]].get(parsed["img_id"], None)
            if mos is None:
                raise ValueError(f"MOS not found for {parsed['compress_type']} img_id {parsed['img_id']}")
            
            # Check if base_key exists in groups
            if base_key not in image_groups:
                image_groups[base_key] = {
                    "left_ports": [],   # Left view viewports (sorted by port_num)
                    "right_ports": [],  # Right view viewports (sorted by port_num)
                    "restored_right_ports": [],  # Restored right viewports (sorted by port_num)
                    "mos": mos
                }
            
            # Assign to left/right port
            if parsed["port_num"] not in [p["port_num"] for p in image_groups[base_key]["left_ports"]]:
                if parsed["left_view"] in vp_filename:  # Left viewport
                    image_groups[base_key]["left_ports"].append({
                        "path": vp_path,
                        "port_num": parsed["port_num"]
                    })
                elif parsed["right_view"] in vp_filename:  # Right viewport
                    image_groups[base_key]["right_ports"].append({
                        "path": vp_path,
                        "port_num": parsed["port_num"]
                    })
        
        # Sort viewports by port_num and find corresponding restored viewports
        for base_key in image_groups:
            # Sort left/right ports by port_num
            image_groups[base_key]["left_ports"].sort(key=lambda x: x["port_num"])
            image_groups[base_key]["right_ports"].sort(key=lambda x: x["port_num"])
            
            # Get restored right viewports (match port_num and add "_res")
            restored_right = []
            for right_port in image_groups[base_key]["right_ports"]:
                right_filename = os.path.basename(right_port["path"])
                restored_filename = right_filename.replace(".png", "_res.png")
                restored_path = os.path.join(RESTORED_VIEWPORTS_DIR, restored_filename)
                
                if not os.path.exists(restored_path):
                    raise FileNotFoundError(f"Restored viewport not found: {restored_path}")
                restored_right.append(restored_path)
            
            image_groups[base_key]["restored_right_ports"] = restored_right
        
        # Convert to list of tuples (for Dataset __getitem__)
        return [
            (
                [p["path"] for p in group["left_ports"]],
                [p["path"] for p in group["right_ports"]],
                group["restored_right_ports"],
                group["mos"]
            )
            for group in image_groups.values()
            if len(group["left_ports"]) == NUM_VIEWPORTS_PER_IMAGE and 
               len(group["right_ports"]) == NUM_VIEWPORTS_PER_IMAGE
        ]

    def __len__(self):
        return len(self.image_groups)

    def __getitem__(self, idx):
        left_paths, right_paths, restored_right_paths, mos = self.image_groups[idx]
        
        # Load left viewports (NUM_VIEWPORTS_PER_IMAGE × 3 × H × W)
        left_imgs = torch.stack([
            self.transform(Image.open(path).convert("RGB")) 
            for path in left_paths
        ])
        
        # Load right viewports
        right_imgs = torch.stack([
            self.transform(Image.open(path).convert("RGB")) 
            for path in right_paths
        ])
        
        # Load restored right viewports (for PDIE)
        restored_right_imgs = torch.stack([
            self.transform(Image.open(path).convert("RGB")) 
            for path in restored_right_paths
        ])
        
        # Convert MOS to tensor
        mos_tensor = torch.tensor(mos, dtype=torch.float32)
        
        return left_imgs, right_imgs, restored_right_imgs, mos_tensor

# -------------------------- Data Loader Factory --------------------------
def get_solid_dataloaders():
    """Create train/test DataLoaders for SOLID dataset"""
    # Load all viewport files
    viewport_files = [
        os.path.join(VIEWPORTS_DIR, f) 
        for f in os.listdir(VIEWPORTS_DIR) 
        if f.endswith(".png") and not f.endswith("_res.png")
    ]
    
    # Load all restored viewport files (not used directly, but validated in Dataset)
    restored_viewport_files = [
        os.path.join(RESTORED_VIEWPORTS_DIR, f) 
        for f in os.listdir(RESTORED_VIEWPORTS_DIR) 
        if f.endswith("_res.png")
    ]
    
    # Load MOS data
    mos_dict = load_mos_data()
    
    # Create Dataset
    transform = get_image_transform()
    dataset = SOLIDDataset(viewport_files, restored_viewport_files, mos_dict, transform)
    
    # Split into train/test
    train_size = int(TRAIN_TEST_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"Train set size: {len(train_dataset)} | Test set size: {len(test_dataset)}")
    return train_loader, test_loader