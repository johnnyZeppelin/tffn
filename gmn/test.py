import torch
from tqdm import tqdm
import numpy as np

from models.tffn import TFFN
from data.data_loader import get_data_loaders
from utils.metrics import calculate_metrics

# --- Configuration ---
SOLID_DIR = './SOLID'
VIEWPORT_DIR = './viewports'
RESTORED_DIR = './restored_viewports'
MODEL_PATH = './trained_models/tffn_solid_best.pth' # Load the best model
BATCH_SIZE = 32 # [cite: 336]
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Main Test ---
def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Load Test Data
    print("Loading SOLID test set...")
    # We only need the test loader
    _, test_loader = get_data_loaders(
        solid_dir=SOLID_DIR,
        viewport_dir=VIEWPORT_DIR,
        restored_dir=RESTORED_DIR,
        batch_size=BATCH_SIZE,
        random_state=RANDOM_SEED
    )
    
    # 2. Initialize Model
    print("Initializing TFFN model...")
    model = TFFN().to(DEVICE)
    
    # 3. Load Trained Weights
    print(f"Loading trained weights from {MODEL_PATH}...")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please run train.py first to generate the model file.")
        return
        
    # 4. Evaluation
    model.eval()
    all_preds = []
    all_gts = []
    
    print("Running evaluation on the test set...")
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for (l_vps, r_vps, res_vps), scores in progress_bar:
            l_vps, r_vps = l_vps.to(DEVICE), r_vps.to(DEVICE)
            res_vps, scores = res_vps.to(DEVICE), scores.to(DEVICE)
            
            outputs = model(l_vps, r_vps, res_vps)
            
            all_preds.extend(outputs.cpu().numpy())
            all_gts.extend(scores.cpu().numpy())
    
    # 5. Calculate Final Metrics
    plcc, srcc, rmse = calculate_metrics(np.array(all_gts), np.array(all_preds))
    
    print("\n--- Overall Test Results (SOLID Dataset) ---")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("\nThese results correspond to the 'TFFN (ours)' row for SOLID in Table I [cite: 354, 487-490].")

if __name__ == '__main__':
    main()