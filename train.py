
import torch, os, argparse, math
from torch import optim
from tqdm import tqdm
from model import TransformerQuantileModel
from utils import get_loaders
import numpy as np

def gaussian_nll(mu, log_var, target):
    # mu, log_var, target: (B, pred_len)
    var = torch.exp(log_var)
    nll = 0.5 * (torch.log(2*math.pi*var) + (target - mu)**2 / var)
    return nll.mean()

def evaluate_model(model, loader, device):
    model.eval()
    import torch
    total = 0
    sums = {'mse':0.0, 'mae':0.0, 'nll':0.0}
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            mu, log_var = model(x)
            mse = ((mu - y)**2).mean().item()
            mae = (torch.abs(mu - y)).mean().item()
            nll = gaussian_nll(mu, log_var, y).item()
            total += 1
            sums['mse'] += mse
            sums['mae'] += mae
            sums['nll'] += nll
    for k in sums:
        sums[k] /= total
    return sums

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/synthetic.npz')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader = get_loaders(args.data, batch_size=args.batch_size)
    device = torch.device(args.device)
    # basic hyperparams
    sample_x, sample_y = next(iter(train_loader))
    seq_len = sample_x.shape[1]
    pred_len = sample_y.shape[1]

    model = TransformerQuantileModel(seq_len=seq_len, pred_len=pred_len).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.7)

    best_val_nll = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x,y in pbar:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            mu, log_var = model(x)
            loss = gaussian_nll(mu, log_var, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        val_scores = evaluate_model(model, val_loader, device)
        print(f"[Epoch {epoch}] val_mse={val_scores['mse']:.6f}, val_mae={val_scores['mae']:.6f}, val_nll={val_scores['nll']:.6f}")
        # save best
        if val_scores['nll'] < best_val_nll:
            best_val_nll = val_scores['nll']
            ckpt = {'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
            print(f"Saved best checkpoint at epoch {epoch}")
