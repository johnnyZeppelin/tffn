import os
import torch

# -------------------------- Path Configuration --------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SOLID_DIR = os.path.join(ROOT_DIR, "SOLID")
VIEWPORTS_DIR = os.path.join(ROOT_DIR, "viewports")
RESTORED_VIEWPORTS_DIR = os.path.join(ROOT_DIR, "restored_viewports")
VMAMBA_DIR = os.path.join(ROOT_DIR, "vmamba")
SAVED_MODELS_DIR = os.path.join(ROOT_DIR, "saved_models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Create directories if not exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# MOS file paths
BPG_MOS_PATH = os.path.join(SOLID_DIR, "BPGmos.xlsx")
JPEG_MOS_PATH = os.path.join(SOLID_DIR, "JPEGmos.xlsx")

# Pretrained VMamba path
VMAMBA_PRETRAINED = os.path.join(VMAMBA_DIR, "vmamba_pretrained.pth")

# -------------------------- Hyperparameters --------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 42  # For reproducibility

# Data settings
TRAIN_TEST_SPLIT = 0.8  # 8:2 split
NUM_VIEWPORTS_PER_IMAGE = 20  # As per paper (20 viewports per image)
IMAGE_SIZE = (224, 224)  # Resize viewports to 224x224 (ResNet/VMamba input)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# Training settings
BATCH_SIZE = 32  # As per paper
EPOCHS = 100
INIT_LR = 1e-3  # As per paper
MOMENTUM = 0.9  # SGD momentum
WEIGHT_DECAY = 1e-4  # Weight decay for regularization
LOG_INTERVAL = 10  # Log every N batches

# Model settings
RESNET50_OUT_DIMS = [512, 1024, 2048]  # Stage2 (512), Stage3 (1024), Stage4 (2048)
SW_MSA_NUM_HEADS = 4  # SW-MSA hyperparameters
SW_MSA_WINDOW_SIZE = 7
VMAMBA_OUT_DIM = 768  # Assume VMamba outputs 768-dim features (adjust if needed)
FF_HIDDEN_DIM = 1024  # Hidden dim for FF block