import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2 as T
from sklearn.model_selection import train_test_split
import numpy as np
from .solid_dataset import SOLIDDataset

def get_data_loaders(solid_dir, viewport_dir, restored_dir, batch_size=32, test_size=0.2, random_seed=42):
    
    # Define transforms
    # We use a standard 224x224 input size for ImageNet backbones
    # The paper doesn't specify viewport size, so this is a reasonable choice.
    transforms = T.Compose([
        T.ToDtype(torch.float32),
        T.Resize((224, 224), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create the full dataset
    dataset = SOLIDDataset(
        solid_dir=solid_dir,
        viewport_dir=viewport_dir,
        restored_dir=restored_dir,
        num_viewports=20, # As per paper [cite: 198]
        transform=transforms
    )
    
    # Create train/test split
    # Following the paper's 8:2 split [cite: 334]
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed
    )
    
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"Full dataset size: {len(dataset)}")
    print(f"Train set size: {len(train_subset)}")
    print(f"Test set size: {len(test_subset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader