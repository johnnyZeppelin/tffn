
import torch, os, argparse, time
from torch.utils.data import DataLoader
from utils import SOLIDViewportDataset, plcc, srcc, rmse
from models.tffn import TFFN
import torch.optim as optim
import torch.nn as nn

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = SOLIDViewportDataset(args.data_root, split='train', viewport_size=args.vp_size)
    test_ds = SOLIDViewportDataset(args.data_root, split='test', viewport_size=args.vp_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = TFFN(root_vmamba=args.root_vmamba, pretrained_resnet=args.pretrained_resnet).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.MSELoss()
    start_epoch = 0
    best_srcc = -1
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            left = batch['left'].to(device)
            right = batch['right'].to(device)
            restored = batch['restored_right'].to(device)
            mos = batch['mos'].to(device)
            optimizer.zero_grad()
            preds = model(left, right, restored)
            loss = criterion(preds, mos)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f\"Epoch {epoch}: train loss {running_loss/len(train_loader):.4f}\")
        # eval
        model.eval()
        preds_all, gts_all = [], []
        with torch.no_grad():
            for batch in test_loader:
                left = batch['left'].to(device); right = batch['right'].to(device); restored = batch['restored_right'].to(device)
                mos = batch['mos'].cpu().numpy()
                preds = model(left, right, restored).cpu().numpy()
                preds_all.extend(preds.tolist()); gts_all.extend(mos.tolist())
        cur_srcc = srcc(preds_all, gts_all)
        cur_plcc = plcc(preds_all, gts_all)
        cur_rmse = rmse(preds_all, gts_all)
        print(f\"Eval: PLCC={cur_plcc:.4f}, SRCC={cur_srcc:.4f}, RMSE={cur_rmse:.4f}\")
        # checkpoint
        ckpt_dir = args.ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'opt_state': optimizer.state_dict()}, os.path.join(ckpt_dir, f\"tffn_epoch{epoch}.pth\"))
        if cur_srcc > best_srcc:
            best_srcc = cur_srcc
            torch.save(model.state_dict(), os.path.join(ckpt_dir, \"tffn_best.pth\"))
    print('Training finished. Best SRCC:', best_srcc)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='SOLID', help='path to SOLID folder')
    parser.add_argument('--root_vmamba', type=str, default='.', help='path that contains vmamba/ folder')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='where to save models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--vp_size', type=int, default=224)
    parser.add_argument('--pretrained_resnet', action='store_true')
    args = parser.parse_args()
    train(args)
