
import torch, argparse, os
from torch.utils.data import DataLoader
from utils import SOLIDViewportDataset, plcc, srcc, rmse
from models.tffn import TFFN
import numpy as np

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SOLIDViewportDataset(args.data_root, split='test', viewport_size=args.vp_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = TFFN(root_vmamba=args.root_vmamba, pretrained_resnet=args.pretrained_resnet).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            left = batch['left'].to(device); right = batch['right'].to(device); restored = batch['restored_right'].to(device)
            mos = batch['mos'].cpu().numpy()
            out = model(left, right, restored).cpu().numpy()
            preds.extend(out.tolist()); gts.extend(mos.tolist())
    print('PLCC', plcc(preds,gts))
    print('SRCC', srcc(preds,gts))
    print('RMSE', rmse(preds,gts))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='SOLID')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--root_vmamba', type=str, default='.')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--vp_size', type=int, default=224)
    parser.add_argument('--pretrained_resnet', action='store_true')
    args = parser.parse_args()
    test(args)
