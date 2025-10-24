import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

def compute_plcc(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]

def compute_srcc(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))