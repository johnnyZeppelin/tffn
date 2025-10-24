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