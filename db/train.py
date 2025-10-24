import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from config import *
from data_utils import get_solid_dataloaders
from model import TFFN
from utils import calculate_metrics, save_logs

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (left_imgs, right_imgs, restored_right_imgs, mos) in enumerate(train_loader):
        # Move data to device
        left_imgs = left_imgs.to(DEVICE)
        right_imgs = right_imgs.to(DEVICE)
        restored_right_imgs = restored_right_imgs.to(DEVICE)
        mos = mos.to(DEVICE).unsqueeze(1)  # (B, 1)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(left_imgs, right_imgs, restored_right_imgs)
        
        # Compute loss
        loss = criterion(pred, mos)
        total_loss += loss.item()
        
        # Backward pass + optimize
        loss.backward()
        optimizer.step()
        
        # Log training progress
        if (batch_idx + 1) % LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {avg_loss:.4f}")
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    """Validate model on test set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_mos = []
    
    with torch.no_grad():
        for left_imgs, right_imgs, restored_right_imgs, mos in val_loader:
            # Move data to device
            left_imgs = left_imgs.to(DEVICE)
            right_imgs = right_imgs.to(DEVICE)
            restored_right_imgs = restored_right_imgs.to(DEVICE)
            mos = mos.to(DEVICE).unsqueeze(1)
            
            # Forward pass
            pred = model(left_imgs, right_imgs, restored_right_imgs)
            
            # Compute loss
            loss = criterion(pred, mos)
            total_loss += loss.item()
            
            # Collect preds and MOS for metrics
            all_preds.extend(pred.cpu().numpy())
            all_mos.extend(mos.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    plcc, srcc, rmse = calculate_metrics(all_preds, all_mos)
    
    print(f"Validation | Loss: {avg_loss:.4f} | PLCC: {plcc:.4f} | SRCC: {srcc:.4f} | RMSE: {rmse:.4f}")
    return avg_loss, plcc, srcc, rmse

def main():
    # Set random seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    
    # Load data
    train_loader, test_loader = get_solid_dataloaders()
    
    # Initialize model
    model = TFFN().to(DEVICE)
    print(f"Model initialized on {DEVICE}")
    
    # Initialize loss and optimizer
    criterion = nn.MSELoss()  # Euclidean loss = MSE
    optimizer = optim.SGD(
        model.parameters(),
        lr=INIT_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # LR decay every 30 epochs
    
    # Training logs
    logs = []
    best_plcc = 0.0  # Track best model by PLCC
    
    # Training loop
    start_time = time.time()
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch [{epoch+1}/{EPOCHS}] ===")
        
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_plcc, val_srcc, val_rmse = validate(model, test_loader, criterion)
        
        # Update scheduler
        scheduler.step()
        
        # Save log
        logs.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_plcc": val_plcc,
            "val_srcc": val_srcc,
            "val_rmse": val_rmse,
            "lr": optimizer.param_groups[0]["lr"]
        })
        
        # Save best model
        if val_plcc > best_plcc:
            best_plcc = val_plcc
            best_model_path = os.path.join(SAVED_MODELS_DIR, "best_tffn.pth")
            torch.save({
                "epoch": epoch+1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_plcc": best_plcc
            }, best_model_path)
            print(f"Best model saved to {best_model_path} (PLCC: {best_plcc:.4f})")
    
    # Save final logs
    log_path = os.path.join(LOGS_DIR, "training_log.csv")
    save_logs(logs, log_path)
    print(f"\nTraining completed in {time.time()-start_time:.2f}s")
    print(f"Logs saved to {log_path}")

if __name__ == "__main__":
    main()