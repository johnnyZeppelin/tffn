import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm

from config import Config
from models.tffn import TFFN
from utils.dataset import SOLIDDataset
from utils.metrics import plcc, srcc, rmse
from utils.helpers import save_checkpoint, plot_training_curves

def train():
    config = Config()
    
    # Initialize model
    model = TFFN(config).to(config.DEVICE)
    
    # Dataset and DataLoader
    train_dataset = SOLIDDataset(config, split='train')
    val_dataset = SOLIDDataset(config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        momentum=config.MOMENTUM, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    for epoch in range(config.NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
        for left_vp, right_vp, restored_vp, mos in pbar:
            left_vp = left_vp.to(config.DEVICE)
            right_vp = right_vp.to(config.DEVICE)
            restored_vp = restored_vp.to(config.DEVICE)
            mos = mos.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_mos = model(left_vp, right_vp, restored_vp)
            loss = criterion(pred_mos, mos)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(pred_mos.detach())
            train_targets.append(mos.detach())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_plcc = plcc(train_preds, train_targets)
        train_srcc = srcc(train_preds, train_targets)
        train_rmse = rmse(train_preds, train_targets)
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for left_vp, right_vp, restored_vp, mos in val_loader:
                left_vp = left_vp.to(config.DEVICE)
                right_vp = right_vp.to(config.DEVICE)
                restored_vp = restored_vp.to(config.DEVICE)
                mos = mos.to(config.DEVICE)
                
                pred_mos = model(left_vp, right_vp, restored_vp)
                loss = criterion(pred_mos, mos)
                
                val_loss += loss.item()
                val_preds.append(pred_mos)
                val_targets.append(mos)
        
        # Calculate validation metrics
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_plcc = plcc(val_preds, val_targets)
        val_srcc = srcc(val_preds, val_targets)
        val_rmse = rmse(val_preds, val_targets)
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train - Loss: {avg_train_loss:.4f}, PLCC: {train_plcc:.4f}, SRCC: {train_srcc:.4f}, RMSE: {train_rmse:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, PLCC: {val_plcc:.4f}, SRCC: {val_srcc:.4f}, RMSE: {val_rmse:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_PATH, 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)
            print(f"Saved best model with val_loss: {best_val_loss:.4f}")
        
        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, avg_val_loss, checkpoint_path)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, os.path.join(config.CHECKPOINT_PATH, 'training_curves.png'))
    
    print("Training completed!")

if __name__ == '__main__':
    train()