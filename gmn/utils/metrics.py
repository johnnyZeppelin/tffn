import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

def calculate_metrics(gts, preds):
    """
    Calculates PLCC, SRCC, and RMSE[cite: 317].
    """
    gts = np.array(gts).squeeze()
    preds = np.array(preds).squeeze()
    
    # PLCC [cite: 319]
    plcc, _ = pearsonr(preds, gts)
    
    # SRCC [cite: 324]
    srcc, _ = spearmanr(preds, gts)
    
    # RMSE [cite: 329]
    rmse = np.sqrt(mean_squared_error(gts, preds))
    
    return plcc, srcc, rmse