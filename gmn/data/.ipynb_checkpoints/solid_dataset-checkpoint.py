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