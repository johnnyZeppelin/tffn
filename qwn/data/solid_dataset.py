import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class SOLIDViewportsDataset(Dataset):
    def __init__(self, img_ids, mos_dict, viewport_dir, restored_dir, transform=None):
        self.img_ids = img_ids
        self.mos_dict = mos_dict
        self.viewport_dir = viewport_dir
        self.restored_dir = restored_dir
        self.transform = transform or transforms.ToTensor()
        self.num_viewports = 20

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        mos = self.mos_dict[img_id]

        # Determine compression type (BPG or JPEG) from img_id range
        # Paper: img_id 1–156 → BPG: 1–78? JPEG: 79–156? But actually both have 1–156
        # Instead, we check existence in both folders
        comp_type = None
        base_name = f"img{img_id:03d}"
        # We assume you stored full name mapping elsewhere, but for simplicity:
        # Try BPG first, then JPEG
        if os.path.exists(os.path.join(self.viewport_dir, f"BPG_{base_name}_2_0_l0_r0_3dv_00.png")):
            comp_type = "BPG"
        else:
            comp_type = "JPEG"

        # Load all 20 viewports for left and right
        left_viewports = []
        right_viewports = []
        diff_maps = []

        for v in range(self.num_viewports):
            # Original viewport names (assume right view used for PDIE)
            vp_name = f"{comp_type}_{base_name}_2_0_l0_r0_3dv_{v:02d}.png"
            vp_path = os.path.join(self.viewport_dir, vp_name)
            rest_path = os.path.join(self.restored_dir, vp_name.replace(".png", "_res.png"))

            # Load right viewport (for PDIE) — note: paper uses right view only
            vp = cv2.imread(vp_path, cv2.IMREAD_COLOR)
            rest = cv2.imread(rest_path, cv2.IMREAD_COLOR)
            if vp is None or rest is None:
                raise FileNotFoundError(f"Missing viewport or restored: {vp_path}")

            # For TPF: we need both left and right, but your data only gives one image?
            # ⚠️ ASSUMPTION: The viewport already corresponds to left/right pair?
            # But paper splits SOI into left/right viewports separately.
            # Since you said "we have viewports", we assume:
            # - You have separate left/right viewport sets.
            # However, your naming only shows one image.
            #
            # WORKAROUND: Since SOI has two views, and you precomputed viewports,
            # we assume you stored:
            #   {comp}_{img}_left_{v}.png and {comp}_{img}_right_{v}.png
            # But your example doesn't show that.
            #
            # Given ambiguity, we assume the viewport is the **right view** (for PDIE),
            # and for TPF, we simulate left = right (or you must adjust data).
            #
            # For now, use same image for left/right (not ideal, but for code structure)
            left_vp = vp.copy()
            right_vp = vp.copy()

            left_viewports.append(self.transform(left_vp))
            right_viewports.append(self.transform(right_vp))

            # PDIE: difference map
            diff = cv2.absdiff(vp, rest).astype(np.float32) / 255.0
            diff_maps.append(self.transform(diff))

        left_viewports = torch.stack(left_viewports)    # [20, C, H, W]
        right_viewports = torch.stack(right_viewports)
        diff_maps = torch.stack(diff_maps)             # [20, C, H, W]

        return {
            "left": left_viewports,
            "right": right_viewports,
            "diff": diff_maps,
            "mos": torch.tensor(mos, dtype=torch.float32)
        }