import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

def calculate_metrics(preds, mos):
    """
    Calculate PLCC, SRCC, and RMSE between predictions and MOS.
    Input: preds (list of np.array), mos (list of np.array)
    Output: plcc, srcc, rmse
    """
    # Flatten lists to 1D arrays
    preds_flat = np.concatenate(preds).flatten()
    mos_flat = np.concatenate(mos).flatten()
    
    # PLCC (Pearson Linear Correlation Coefficient)
    plcc, _ = pearsonr(preds_flat, mos_flat)
    plcc = max(plcc, 0.0)  # Ensure non-negative (since PLCC ranges 0-1 in paper)
    
    # SRCC (Spearman Rank-order Correlation Coefficient)
    srcc, _ = spearmanr(preds_flat, mos_flat)
    srcc = max(srcc, 0.0)  # Ensure non-negative
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(mos_flat, preds_flat))
    
    return plcc, srcc, rmse

def save_logs(logs, path):
    """Save training logs to CSV"""
    df = pd.DataFrame(logs)
    df.to_csv(path, index=False)
    print(f"Logs saved to {path}")

def load_model(model, checkpoint_path, device):
    """Load model weights from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint