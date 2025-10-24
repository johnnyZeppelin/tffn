import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

from config import Config
from models.tffn import TFFN
from utils.dataset import SOLIDDataset
from utils.metrics import plcc, srcc, rmse
from utils.helpers import load_checkpoint

def test():
    config = Config()
    
    # Initialize model
    model = TFFN(config).to(config.DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
    
    # Load best model
    checkpoint_path = os.path.join(config.CHECKPOINT_PATH, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        model, optimizer, _, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print("Loaded best model for testing")
    else:
        print("No checkpoint found. Using randomly initialized model.")
    
    # Test dataset
    test_dataset = SOLIDDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Testing
    model.eval()
    test_preds = []
    test_targets = []
    
    print("Starting testing...")
    with torch.no_grad():
        for left_vp, right_vp, restored_vp, mos in tqdm(test_loader, desc='Testing'):
            left_vp = left_vp.to(config.DEVICE)
            right_vp = right_vp.to(config.DEVICE)
            restored_vp = restored_vp.to(config.DEVICE)
            
            pred_mos = model(left_vp, right_vp, restored_vp)
            
            test_preds.append(pred_mos.cpu())
            test_targets.append(mos.cpu())
    
    # Calculate metrics
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    
    test_plcc = plcc(test_preds, test_targets)
    test_srcc = srcc(test_preds, test_targets)
    test_rmse = rmse(test_preds, test_targets)
    
    print("\nTest Results:")
    print(f"PLCC: {test_plcc:.4f}")
    print(f"SRCC: {test_srcc:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    # Save results
    results_path = os.path.join(config.CHECKPOINT_PATH, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write("TFFN Test Results on SOLID Dataset\n")
        f.write("=" * 40 + "\n")
        f.write(f"PLCC: {test_plcc:.4f}\n")
        f.write(f"SRCC: {test_srcc:.4f}\n")
        f.write(f"RMSE: {test_rmse:.4f}\n")
        f.write(f"Number of test samples: {len(test_dataset)}\n")
    
    print(f"Results saved to {results_path}")

if __name__ == '__main__':
    test()