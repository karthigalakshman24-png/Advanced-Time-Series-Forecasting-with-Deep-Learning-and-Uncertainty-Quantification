
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        # X: (N, seq_len), Y: (N, pred_len)
        self.X = X.astype('float32')
        self.Y = Y.astype('float32')
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def get_loaders(npz_path, batch_size=64, val_split=0.1, shuffle=True):
    data = np.load(npz_path)
    X, Y = data['X'], data['Y']
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    split = int(n * (1 - val_split))
    train_idx, val_idx = idx[:split], idx[split:]
    train_ds = TimeSeriesDataset(X[train_idx], Y[train_idx])
    val_ds = TimeSeriesDataset(X[val_idx], Y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
