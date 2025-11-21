
import torch, argparse, numpy as np, math, os
from model import TransformerQuantileModel
from utils import get_loaders

def gaussian_crps(mu, sigma, y):
    # analytic CRPS for normal distribution averaged across points
    from scipy.stats import norm
    z = (y - mu) / sigma
    crps = sigma * (z * (2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/math.sqrt(math.pi))
    return np.mean(crps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data', type=str, default='./data/synthetic.npz')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    data = args.data

    train_loader, val_loader = get_loaders(data, batch_size=256, val_split=0.0)  # load all as single loader
    # Build model shape from data
    sample_x, sample_y = next(iter(train_loader))
    seq_len = sample_x.shape[1]
    pred_len = sample_y.shape[1]
    model = TransformerQuantileModel(seq_len=seq_len, pred_len=pred_len).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    import numpy as np
    from scipy.stats import norm
    mus = []
    sigmas = []
    ys = []
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            y = y.to(device)
            mu, log_var = model(x)
            sigma = torch.sqrt(torch.exp(log_var))
            mus.append(mu.cpu().numpy())
            sigmas.append(sigma.cpu().numpy())
            ys.append(y.cpu().numpy())
    mus = np.concatenate(mus, axis=0)
    sigmas = np.concatenate(sigmas, axis=0)
    ys = np.concatenate(ys, axis=0)
    rmse = np.sqrt(((mus - ys)**2).mean())
    mae = np.abs(mus - ys).mean()
    nll = 0.5 * (np.log(2*np.pi*(sigmas**2)) + ((ys - mus)**2) / (sigmas**2))
    nll = nll.mean()
    # CRPS (mean across entries), using analytic formula per-step
    crps = gaussian_crps(mus, sigmas, ys)
    # 95% coverage
    z = norm.ppf(0.975)
    lower = mus - z * sigmas
    upper = mus + z * sigmas
    coverage = ((ys >= lower) & (ys <= upper)).mean()

    print(f"RMSE: {rmse:.6f}, MAE: {mae:.6f}, NLL: {nll:.6f}, CRPS: {crps:.6f}, 95% coverage: {coverage:.4f}")
