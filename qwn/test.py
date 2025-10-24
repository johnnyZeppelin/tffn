import os
import yaml
import torch
from torch.utils.data import DataLoader
from data.solid_dataset import SOLIDViewportsDataset
from models.tffn import TFFN
from utils.misc import get_device
from utils.metrics import compute_plcc, compute_srcc, compute_rmse
import pandas as pd
from sklearn.model_selection import train_test_split

# Same load_mos_data and split as train.py
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
        cfg = cfg = yaml.safe_load(f)

    device = get_device()
    mos_dict = load_mos_data(cfg["data"]["root"])
    img_ids = list(mos_dict.keys())
    _, test_ids = train_test_split(
        img_ids,
        test_size=0.2,
        random_state=cfg["data"]["seed"]
    )

    test_dataset = SOLIDViewportsDataset(
        test_ids,
        mos_dict,
        cfg["data"]["viewport_dir"],
        cfg["data"]["restored_dir"]
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = TFFN(cfg["model"]["vmamba_ckpt"]).to(device)
    model.load_state_dict(torch.load(cfg["test"]["ckpt_path"], map_location=device))
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            left = batch["left"].to(device)
            right = batch["right"].to(device)
            diff = batch["diff"].to(device)
            target = batch["mos"].item()

            pred = model(left, right, diff).item()
            preds.append(pred)
            targets.append(target)

    plcc = compute_plcc(targets, preds)
    srcc = compute_srcc(targets, preds)
    rmse = compute_rmse(targets, preds)

    print(f"SOLID Test Results:")
    print(f"PLCC: {plcc:.3f}")
    print(f"SRCC: {srcc:.3f}")
    print(f"RMSE: {rmse:.3f}")

if __name__ == "__main__":
    main()