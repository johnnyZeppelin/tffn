Great! Based on your detailed description and the paper **TFFN: Three-branch Feature Fusion Network for Stereoscopic Omnidirectional Image Quality Assessment**, we will design a clean, modular, and reproducible PyTorch project that follows the architecture and data setup you've outlined.

---

## ✅ Project Structure Overview

We’ll organize the code as follows:

```
TFFN_SOIQA/
├── config/
│   └── solid_config.yaml          # dataset/model/training config
├── data/
│   └── solid_dataset.py           # custom PyTorch Dataset
├── models/
│   ├── vmamba_wrapper.py          # VMamba backbone (load pretrained, unfrozen)
│   ├── tffn.py                    # full TFFN model (TPF + PDIE + FF + regressor)
│   └── resnet50_tpf.py            # ResNet50-based TPF block (for binocular features)
├── utils/
│   ├── viewport_utils.py          # helpers for loading viewports
│   ├── metrics.py                 # PLCC, SRCC, RMSE
│   └── misc.py                    # seed, device, etc.
├── train.py                       # training loop
├── test.py                        # inference + evaluation on SOLID
├── requirements.txt
└── README.md
```

We assume:
- You have **20 viewports per image** (as per paper).
- Viewports and restored viewports are precomputed and stored as described.
- VMamba is installed and checkpoint is at `./vmamba/vmamba_s.pth` (or similar).
- We use **ResNet50** for TPF (as in Fig. 1 and Sec III-B).
- Only **right view** is used in PDIE (as stated in Sec III-C).
- We split data **8:2** (train:test) **by image ID**, not by viewport.

---

## 📁 Step-by-step File Implementation

### 1. `requirements.txt`
```txt
torch>=2.0
torchvision
numpy
pandas
opencv-python
scikit-learn
pyyaml
tqdm
mamba-ssm  # optional, but VMamba may require custom kernel
```

> Note: VMamba requires compiling its CUDA kernel. Ensure you've run:
> ```bash
> cd VMamba/kernels/selective_scan && pip install .
> ```

---

### 2. `config/solid_config.yaml`
```yaml
data:
  root: "./SOLID"
  viewport_dir: "./viewports"
  restored_dir: "./restored_viewports"
  mos_files:
    BPG: "BPGmos.xlsx"
    JPEG: "JPEGmos.xlsx"
  num_viewports: 20
  split_ratio: 0.8
  seed: 42

model:
  vmamba_ckpt: "./vmamba/vmamba_s.pth"  # or vmamba_t, adjust as needed
  resnet_pretrained: true
  embed_dim: 512  # output feature dim before fusion

train:
  batch_size: 8      # per GPU (each sample = 20 viewports → memory heavy)
  num_workers: 4
  epochs: 50
  lr: 1e-4
  weight_decay: 1e-4
  save_dir: "./checkpoints"
  log_interval: 10

test:
  ckpt_path: "./checkpoints/best.pth"
```

---

### 3. `utils/misc.py`
```python
import torch
import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

### 4. `utils/metrics.py`
```python
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

def compute_plcc(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def compute_srcc(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
```

---

### 5. `data/solid_dataset.py`
```python
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
```

> 🔔 **Note**: The above assumes you have **separate left/right viewports**. If not, you must adjust preprocessing. For now, we proceed with right-view-only and duplicate for left.

---

### 6. `models/resnet50_tpf.py`
```python
import torch
import torch.nn as nn
import torchvision.models as models
from einops import rearrange

class ResNet50TPF(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # not used (stage1)
        self.layer2 = resnet.layer2  # stage2 → used
        self.layer3 = resnet.layer3  # stage3
        self.layer4 = resnet.layer4  # stage4

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        s2 = self.layer2(x)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        return [s2, s3, s4]  # multi-scale features
```

---

### 7. `models/vmamba_wrapper.py`
```python
import torch
import torch.nn as nn
from vmamba import VSSM  # assuming you have vmamba.py in PYTHONPATH

class VMambaWrapper(nn.Module):
    def __init__(self, ckpt_path, num_classes=512):
        super().__init__()
        # Load VMamba-S (50M params) as in paper
        self.vmamba = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=num_classes,
            depths=[2, 2, 15, 2],
            dims=[96, 192, 384, 768],
            ssm_d_state=16,
            ssm_dt_rank="auto",
            ssm_ratio=2.0,
            mlp_ratio=4.0,
            downsample_version="v3",
            patchembed_version="v2",
        )
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "model" in ckpt:
                ckpt = ckpt["model"]
            self.vmamba.load_state_dict(ckpt, strict=False)
        # We'll use global avg pooling on final feature map
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, C, H, W] → VMamba expects [B, C, H, W]
        # VMamba returns [B, L, D] if return_all_tokens=True, else [B, num_classes]
        # We want feature map, so modify to return features
        # But original VMamba returns cls token only.
        # So we assume you modified VMamba to return patch tokens.
        # Alternatively, use feature extraction mode.
        # For simplicity, assume we get [B, D, H', W'] from backbone
        feats = self.vmamba.forward_features(x)  # [B, D, H', W']
        feats = feats.flatten(2)  # [B, D, N]
        feats = self.avgpool(feats).squeeze(-1)  # [B, D]
        return feats
```

> ⚠️ You may need to modify `VSSM` to expose `forward_features`. Refer to VMamba repo’s `get_feature_maps` or similar.

---

### 8. `models/tffn.py`
```python
import torch
import torch.nn as nn
from .resnet50_tpf import ResNet50TPF
from .vmamba_wrapper import VMambaWrapper
from timm.models.layers import LayerNorm2d
from einops import rearrange

class BDModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sw_msa = nn.Identity()  # Placeholder; replace with real SW-MSA if needed
        self.norm_left = LayerNorm2d(dim)
        self.norm_right = LayerNorm2d(dim)

    def forward(self, F_left, F_right):
        # Simplified: no real SW-MSA due to complexity; use identity
        F_left = F_left + torch.relu(self.norm_left(F_left))
        F_right = F_right + torch.relu(self.norm_right(F_right))
        return F_left - F_right

class BSModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, F_left, F_right):
        return F_left + F_right

class TPFBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet50TPF(pretrained=True)
        self.bd = BDModule(2048)  # stage4 dim
        self.bs = BSModule()
        self.fc = nn.Linear(2048, 512)

    def forward(self, left, right):
        # left, right: [B, 20, C, H, W] → process each viewport
        B, V, C, H, W = left.shape
        left = left.view(B*V, C, H, W)
        right = right.view(B*V, C, H, W)

        feats_left = self.resnet(left)[-1]  # [B*V, 2048, h, w]
        feats_right = self.resnet(right)[-1]

        bd_feat = self.bd(feats_left, feats_right)  # [B*V, 2048, h, w]
        bs_feat = self.bs(feats_left, feats_right)

        # Global avg pool
        bd_feat = bd_feat.mean(dim=[2,3])  # [B*V, 2048]
        bs_feat = bs_feat.mean(dim=[2,3])

        bd_feat = self.fc(bd_feat).view(B, V, -1).mean(dim=1)  # [B, 512]
        bs_feat = self.fc(bs_feat).view(B, V, -1).mean(dim=1)
        return bd_feat, bs_feat

class PDIEBlock(nn.Module):
    def __init__(self, vmamba_ckpt):
        super().__init__()
        self.vmamba = VMambaWrapper(vmamba_ckpt, num_classes=512)

    def forward(self, diff_maps):
        # diff_maps: [B, 20, C, H, W] → concat along batch
        B, V, C, H, W = diff_maps.shape
        x = diff_maps.view(B*V, C, H, W)
        feats = self.vmamba(x)  # [B*V, 512]
        return feats.view(B, V, -1).mean(dim=1)  # [B, 512]

class FFBlock(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.fc = nn.Linear(dim * 3, 1)

    def forward(self, bd, bs, mf):
        fused = torch.cat([bd, bs, mf], dim=1)  # [B, 1536]
        return self.fc(fused).squeeze(-1)

class TFFN(nn.Module):
    def __init__(self, vmamba_ckpt):
        super().__init__()
        self.tpf = TPFBlock()
        self.pdie = PDIEBlock(vmamba_ckpt)
        self.ff = FFBlock()

    def forward(self, left, right, diff):
        bd, bs = self.tpf(left, right)
        mf = self.pdie(diff)
        score = self.ff(bd, bs, mf)
        return score
```

---

### 9. `train.py`
```python
import os
import yaml
import torch
from torch.utils.data import DataLoader
from data.solid_dataset import SOLIDViewportsDataset
from models.tffn import TFFN
from utils.misc import set_seed, get_device
from utils.metrics import compute_plcc, compute_srcc
import pandas as pd
from sklearn.model_selection import train_test_split

def load_mos_data(root):
    mos_dict = {}
    for comp in ["BPG", "JPEG"]:
        df = pd.read_excel(os.path.join(root, f"{comp}mos.xlsx"))
        for _, row in df.iterrows():
            img_id = int(row["img_id"])
            mos_dict[img_id] = float(row["overall"])
    return mos_dict

def main():
    with open("config/solid_config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    device = get_device()

    mos_dict = load_mos_data(cfg["data"]["root"])
    img_ids = list(mos_dict.keys())
    train_ids, test_ids = train_test_split(
        img_ids,
        test_size=1 - cfg["data"]["split_ratio"],
        random_state=cfg["data"]["seed"]
    )

    train_dataset = SOLIDViewportsDataset(
        train_ids,
        mos_dict,
        cfg["data"]["viewport_dir"],
        cfg["data"]["restored_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"]
    )

    model = TFFN(cfg["model"]["vmamba_ckpt"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    criterion = nn.MSELoss()

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    best_loss = float('inf')
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            diff = batch["diff"].to(device)
            target = batch["mos"].to(device)

            pred = model(left, right, diff)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg["train"]["save_dir"], "best.pth"))

if __name__ == "__main__":
    main()
```

---

### 10. `test.py`
```python
import os
import yaml
import torch
from torch.utils.data import DataLoader
from data.solid_dataset import SOLIDViewportsDataset
from models.tffn import TFFN
from utils.misc import get_device
from utils.metrics import compute_plcc, compute_srcc, compute_rmse
import pandas as pd
from sklearn.model_selection import train_test_split

# Same load_mos_data and split as train.py
def load_mos_data(root):
    mos_dict = {}
    for comp in ["BPG", "JPEG"]:
        df = pd.read_excel(os.path.join(root, f"{comp}mos.xlsx"))
        for _, row in df.iterrows():
            img_id = int(row["img_id"])
            mos_dict[img_id] = float(row["overall"])
    return mos_dict

def main():
    with open("config/solid_config.yaml") as f:
        cfg = cfg = yaml.safe_load(f)

    device = get_device()
    mos_dict = load_mos_data(cfg["data"]["root"])
    img_ids = list(mos_dict.keys())
    _, test_ids = train_test_split(
        img_ids,
        test_size=0.2,
        random_state=cfg["data"]["seed"]
    )

    test_dataset = SOLIDViewportsDataset(
        test_ids,
        mos_dict,
        cfg["data"]["viewport_dir"],
        cfg["data"]["restored_dir"]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TFFN(cfg["model"]["vmamba_ckpt"]).to(device)
    model.load_state_dict(torch.load(cfg["test"]["ckpt_path"], map_location=device))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            diff = batch["diff"].to(device)
            target = batch["mos"].item()

            pred = model(left, right, diff).item()
            preds.append(pred)
            targets.append(target)

    plcc = compute_plcc(targets, preds)
    srcc = compute_srcc(targets, preds)
    rmse = compute_rmse(targets, preds)

    print(f"SOLID Test Results:")
    print(f"PLCC: {plcc:.3f}")
    print(f"SRCC: {srcc:.3f}")
    print(f"RMSE: {rmse:.3f}")

if __name__ == "__main__":
    main()
```

---

## 📦 Final Notes

- This code is **modular and extensible**.
- You must ensure:
  - VMamba is correctly installed and `VSSM` can be imported.
  - Viewport naming matches exactly.
  - Left/right viewports are properly separated (current code assumes duplication).
- For **real SW-MSA**, you may integrate from Swin Transformer.
- For **true multi-scale TPF**, aggregate features from stage2/3/4 with FC layers.

Would you like me to package all these files into a `.zip`?