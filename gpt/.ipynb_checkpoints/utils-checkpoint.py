
import os, random, math
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pandas as pd

def read_mos_table(root):
    # loads BPGmos.xlsx and JPEGmos.xlsx if exist, combine into mapping by img_id -> overall
    mos = {}
    for fname in ["BPGmos.xlsx","JPEGmos.xlsx"]:
        p = os.path.join(root, fname)
        if os.path.exists(p):
            df = pd.read_excel(p)
            for _, r in df.iterrows():
                img_id = int(r['img_id'])
                overall = float(r['overall'])
                mos[img_id] = overall
    return mos

class SOLIDViewportDataset(Dataset):
    \"\"\"Loads viewport-level data for SOLID. Expects viewports in viewports/ folder with naming like:
       BPG_img001_2_0_l0_r0_3dv_02.png and restored versions in restored_viewports/ with suffix _res.png
       The original distorted image file mapping (imgXXX...) is used to match MOS via img_id (1..156).
    \"\"\"
    def __init__(self, root, split='train', train_ratio=0.8, transform=None, viewport_size=224):
        self.root = root
        self.viewport_dir = os.path.join(root, "viewports")
        self.restored_dir = os.path.join(root, "restored_viewports")
        self.transform = transform or T.Compose([T.Resize((viewport_size,viewport_size)), T.ToTensor()])
        self.mos_map = read_mos_table(root)
        # collect viewport groups by original image name (imgXXX_..._3dv)
        files = [f for f in os.listdir(self.viewport_dir) if f.lower().endswith(".png")]
        groups = {}
        for f in files:
            # original name without the trailing _NN (port number)
            base = "_".join(f.split("_")[:6])  # BPG_img001_2_0_l0_r0_3dv  (safe for provided format)
            groups.setdefault(base, []).append(f)
        # convert groups to list
        items = []
        for base, files in groups.items():
            # extract img id like img001 -> 1
            # base starts with {TYPE}_imgXXX_...
            parts = base.split("_")
            if len(parts) < 2:
                continue
            imgpart = parts[1]
            try:
                img_id = int(imgpart.replace("img",""))
            except:
                img_id = None
            items.append((base, sorted(files), img_id))
        # sort and split
        items = sorted(items, key=lambda x: x[0])
        n_train = int(len(items)*train_ratio)
        if split=='train':
            self.items = items[:n_train]
        else:
            self.items = items[n_train:]
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        base, files, img_id = self.items[idx]
        # load paired viewports: left and right view must be present; we will assume files include both L and R viewports per port index
        # We'll build tensors by concatenating features across viewports per side (for simplicity, average features across viewports instead of per-viewport aggregation)
        left_imgs = []
        right_imgs = []
        restored_right = []
        for vf in files:
            p = os.path.join(self.viewport_dir, vf)
            img = Image.open(p).convert('RGB')
            # determine left or right by _l0 or _l1 in the filename
            if "_l0_" in vf or "_l0." in vf:
                left_imgs.append(self.transform(img))
            elif "_l1_" in vf or "_l1." in vf:
                left_imgs.append(self.transform(img))
            # look for corresponding restored viewport (right view restored is used in PDIE according to paper; but we'll attempt to find restored for every port)
            res_name = vf.replace(".png","_res.png")
            res_p = os.path.join(self.restored_dir, res_name)
            if os.path.exists(res_p):
                rimg = Image.open(res_p).convert('RGB')
                restored_right.append(self.transform(rimg))
        # fallbacks: if no restored images found, create zero tensors
        if len(left_imgs)==0:
            left_tensor = torch.zeros(3,224,224)
        else:
            left_tensor = torch.stack(left_imgs).mean(dim=0)
        if len(right_imgs)==0:
            right_tensor = torch.zeros(3,224,224)
        else:
            right_tensor = torch.stack(right_imgs).mean(dim=0)
        if len(restored_right)==0:
            restored_tensor = torch.zeros_like(right_tensor)
        else:
            restored_tensor = torch.stack(restored_right).mean(dim=0)
        # MOS mapping: if missing, default to 3.0
        mos = 3.0
        if img_id and img_id in self.mos_map:
            mos = float(self.mos_map[img_id])
        sample = {
            'left': left_tensor, 'right': right_tensor, 'restored_right': restored_tensor, 'mos': torch.tensor(mos, dtype=torch.float32), 'name': base
        }
        return sample

# metrics
def plcc(preds, gts):
    preds = np.asarray(preds); gts = np.asarray(gts)
    # linear fit before plcc as common practice
    if preds.size == 0:
        return 0.0
    a = np.vstack([preds, np.ones_like(preds)]).T
    m, c = np.linalg.lstsq(a, gts, rcond=None)[0]
    preds_lin = m*preds + c
    num = np.sum((gts - gts.mean())*(preds_lin - preds_lin.mean()))
    den = np.sqrt(np.sum((gts - gts.mean())**2)*np.sum((preds_lin - preds_lin.mean())**2)+1e-12)
    return num/den
def srcc(preds, gts):
    from scipy.stats import spearmanr
    if len(preds)==0:
        return 0.0
    return spearmanr(preds, gts).correlation
def rmse(preds, gts):
    return float(np.sqrt(np.mean((np.asarray(preds)-np.asarray(gts))**2 + 1e-12)))
