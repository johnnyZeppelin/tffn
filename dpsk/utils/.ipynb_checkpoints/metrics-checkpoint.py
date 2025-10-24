import torch
import numpy as np
from scipy.stats import spearmanr

def plcc(pred, target):
    """
    Pearson Linear Correlation Coefficient
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    return np.corrcoef(pred, target)[0, 1]

def srcc(pred, target):
    """
    Spearman Rank-order Correlation Coefficient
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    return spearmanr(pred, target).correlation

def rmse(pred, target):
    """
    Root Mean Squared Error
    """
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    return np.sqrt(np.mean((pred - target) ** 2))