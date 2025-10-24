# TFFN: Three-branch Feature Fusion Network for SOIQA

Reproduction of the TFFN model for Stereoscopic Omnidirectional Image Quality Assessment on the SOLID dataset.

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


## Setup
1. Install VMamba:  
   ```bash
   git clone https://github.com/MzeroMiko/VMamba.git
   cd VMamba && pip install -e .
   ```
2. Place pretrained VMamba checkpoint at `./vmamba/vmamba_s.pth`
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

## Data Structure
```
SOLID/
├── BPG/
├── JPEG/
├── BPGmos.xlsx
└── JPEGmos.xlsx
viewports/
restored_viewports/
```

## Train
```bash
python train.py
```

## Test
```bash
python test.py
```
```

---

