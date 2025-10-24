import os
import yaml
import torch
from torch.utils.data import DataLoader
from data.solid_dataset import SOLIDViewportsDataset
from models.tffn import TFFN
from utils.misc import set_seed, get_device
from utils.metrics import compute_plcc, compute_srcc
import pandas as pd
from sklearn.model_selection import train_test_split

def load_mos_data(root):
    mos_dict = {}
    for comp in ["BPG", "JPEG"]:
        df = pd.read_excel(os.path.join(root, f"{comp}mos.xlsx"))
        for _, row in df.iterrows():
            img_id = int(row["img_id"])
            mos_dict[img_id] = float(row["overall"])
    return mos_dict

def main():
    with open("config/solid_config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["data"]["seed"])
    device = get_device()

    mos_dict = load_mos_data(cfg["data"]["root"])
    img_ids = list(mos_dict.keys())
    train_ids, test_ids = train_test_split(
        img_ids,
        test_size=1 - cfg["data"]["split_ratio"],
        random_state=cfg["data"]["seed"]
    )

    train_dataset = SOLIDViewportsDataset(
        train_ids,
        mos_dict,
        cfg["data"]["viewport_dir"],
        cfg["data"]["restored_dir"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"]
    )

    model = TFFN(cfg["model"]["vmamba_ckpt"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )
    criterion = nn.MSELoss()

    os.makedirs(cfg["train"]["save_dir"], exist_ok=True)

    best_loss = float('inf')
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0
        for batch in train_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            diff = batch["diff"].to(device)
            target = batch["mos"].to(device)

            pred = model(left, right, diff)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(cfg["train"]["save_dir"], "best.pth"))

if __name__ == "__main__":
    main()