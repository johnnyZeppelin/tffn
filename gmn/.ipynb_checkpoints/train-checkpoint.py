import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from models.tffn import TFFN
from data.data_loader import get_data_loaders
from utils.metrics import calculate_metrics

# --- Configuration ---
SOLID_DIR = './SOLID'
VIEWPORT_DIR = './viewports'
RESTORED_DIR = './restored_viewports'
MODEL_SAVE_PATH = './trained_models'
NUM_EPOCHS = 50 # Paper doesn't specify, 50 is a reasonable start
BATCH_SIZE = 32 # [cite: 336]
LEARNING_RATE = 1e-3 # [cite: 336]
MOMENTUM = 0.9 # [cite: 335]
WEIGHT_DECAY = 1e-4 # [cite: 335]
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Main Training ---
def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 1. Load Data
    print("Loading SOLID dataset...")
    train_loader, test_loader = get_data_loaders(
        solid_dir=SOLID_DIR,
        viewport_dir=VIEWPORT_DIR,
        restored_dir=RESTORED_DIR,
        batch_size=BATCH_SIZE,
        random_seed=RANDOM_SEED
    )
    
    # 2. Initialize Model
    print("Initializing TFFN model...")
    model = TFFN().to(DEVICE)
    
    # 3. Setup Optimizer and Loss
    # Optimizer: SGD with momentum [cite: 335]
    optimizer = optim.SGD(
        model.parameters(), 
        lr=LEARNING_RATE, 
        momentum=MOMENTUM, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Loss: Euclidean loss (MSE) [cite: 293]
    criterion = nn.MSELoss()
    
    # Learning rate scheduler (optional, but good practice)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    # 4. Training Loop
    best_srcc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for (l_vps, r_vps, res_vps), scores in progress_bar:
            # Move data to device
            l_vps, r_vps = l_vps.to(DEVICE), r_vps.to(DEVICE)
            res_vps, scores = res_vps.to(DEVICE), scores.to(DEVICE)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(l_vps, r_vps, res_vps)
            
            # Calculate loss
            loss = criterion(outputs.squeeze(), scores)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * l_vps.size(0)
            progress_bar.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        # --- Validation (Test) Phase ---
        model.eval()
        all_preds = []
        all_gts = []
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar_val = tqdm(test_loader, desc='Validation')
            for (l_vps, r_vps, res_vps), scores in progress_bar_val:
                l_vps, r_vps = l_vps.to(DEVICE), r_vps.to(DEVICE)
                res_vps, scores = res_vps.to(DEVICE), scores.to(DEVICE)
                
                outputs = model(l_vps, r_vps, res_vps)
                
                loss = criterion(outputs.squeeze(), scores)
                val_loss += loss.item() * l_vps.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_gts.extend(scores.cpu().numpy())
        
        avg_val_loss = val_loss / len(test_loader.dataset)
        
        # Calculate metrics
        plcc, srcc, rmse = calculate_metrics(np.array(all_gts), np.array(all_preds))
        
        print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
        print(f"PLCC: {plcc:.4f} | SRCC: {srcc:.4f} | RMSE: {rmse:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Save best model
        if srcc > best_srcc:
            best_srcc = srcc
            save_file = os.path.join(MODEL_SAVE_PATH, 'tffn_solid_best.pth')
            torch.save(model.state_dict(), save_file)
            print(f"New best model saved to {save_file} (SRCC: {best_srcc:.4f})")
            
    # Save final model
    final_save_file = os.path.join(MODEL_SAVE_PATH, 'tffn_solid_final.pth')
    torch.save(model.state_dict(), final_save_file)
    print(f"Final model saved to {final_save_file}")

if __name__ == '__main__':
    main()