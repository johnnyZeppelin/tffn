# TFFN for Stereoscopic Omnidirectional Image Quality Assessment (SOLID Dataset)
This project is for the TFFN model for Stereoscopic Omnidirectional Image Quality Assessment (SOIQA) on the SOLID dataset.


## Project Structure
```
TFFN_SOIQA/
├── config.py               # Configuration (paths, hyperparameters)
├── data_utils.py           # Data loading & preprocessing
├── model.py                # TFFN model definition (TPF/PDIE/FF blocks)
├── train.py                # Training pipeline (train/validate)
├── test.py                 # Test pipeline (evaluate PLCC/SRCC/RMSE)
├── utils.py                # Helper functions (metrics, image loading)
├── requirements.txt        # Dependencies
├── README.md               # Usage instructions
├── saved_models/           # Trained models (auto-created)
│   └── best_tffn.pth       # Best model (saved during training)
├── logs/                   # Training logs (auto-created)
│   └── training_log.csv
├── vmamba/                 # Pretrained VMamba (user-provided)
│   └── vmamba_pretrained.pth
├── viewports/              # User-provided viewports
│   ├── BPG_img001_2_0_l0_r0_3dv_00.png
│   ├── JPEG_img002_2_0_l1_r1_3dv_01.png
│   └── ...
├── restored_viewports/     # User-provided restored viewports
│   ├── BPG_img001_2_0_l0_r0_3dv_00_res.png
│   ├── JPEG_img002_2_0_l1_r1_3dv_01_res.png
│   └── ...
└── SOLID/                  # User-provided SOLID dataset
    ├── BPG/
    ├── JPEG/
    ├── BPGmos.xlsx
    ├── JPEGmos.xlsx
    └── README.txt
```

## Prerequisites
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Organize datasets as per the project structure (SOLID, viewports, restored_viewports, vmamba).
3. Ensure pretrained VMamba is in `vmamba/vmamba_pretrained.pth`.

## Training
Run the training script:
```bash
python train.py
```
- Trained models are saved to `saved_models/best_tffn.pth`.
- Training logs are saved to `logs/training_log.csv`.

## Testing
Run the test script (uses the best trained model):
```bash
python test.py
```
- Test results (PLCC/SRCC/RMSE) are printed and saved to `logs/test_results.txt`.

## Notes
- The model uses a 8:2 train/test split (fixed seed=42 for reproducibility).
- VMamba is not frozen (parameters are updated during training).
- ResNet50 uses ImageNet pretrained weights for feature extraction.


The user will need to add their own `SOLID/`, `viewports/`, `restored_viewports/`, and `vmamba/` folders to the unzipped directory before running.