import torch
import torch.nn as nn
from config import *
from data_utils import get_solid_dataloaders
from model import TFFN
from utils import calculate_metrics

def test_best_model():
    # Set random seed
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Load data (test loader only)
    _, test_loader = get_solid_dataloaders()
    
    # Initialize model
    model = TFFN().to(DEVICE)
    
    # Load best model weights
    best_model_path = os.path.join(SAVED_MODELS_DIR, "best_tffn.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded best model (epoch {checkpoint['epoch']}, PLCC: {checkpoint['best_plcc']:.4f})")
    
    # Test model
    model.eval()
    all_preds = []
    all_mos = []
    
    with torch.no_grad():
        for batch_idx, (left_imgs, right_imgs, restored_right_imgs, mos) in enumerate(test_loader):
            # Move data to device
            left_imgs = left_imgs.to(DEVICE)
            right_imgs = right_imgs.to(DEVICE)
            restored_right_imgs = restored_right_imgs.to(DEVICE)
            mos = mos.to(DEVICE).unsqueeze(1)
            
            # Forward pass
            pred = model(left_imgs, right_imgs, restored_right_imgs)
            
            # Collect results
            all_preds.extend(pred.cpu().numpy())
            all_mos.extend(mos.cpu().numpy())
            
            # Progress
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                print(f"Test Batch [{batch_idx+1}/{len(test_loader)}]")
    
    # Calculate metrics
    plcc, srcc, rmse = calculate_metrics(all_preds, all_mos)
    
    # Print results (match Table I SOLID part)
    print("\n=== SOLID Dataset Test Results ===")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Save results
    result_path = os.path.join(LOGS_DIR, "test_results.txt")
    with open(result_path, "w") as f:
        f.write("SOLID Dataset Test Results\n")
        f.write(f"PLCC: {plcc:.4f}\n")
        f.write(f"SRCC: {srcc:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
    print(f"\nResults saved to {result_path}")

if __name__ == "__main__":
    test_best_model()