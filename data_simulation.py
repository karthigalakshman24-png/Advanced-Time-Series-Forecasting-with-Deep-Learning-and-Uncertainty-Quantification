
import numpy as np
import argparse
import os

def generate_series(T=20000, seq_len=100, pred_len=10, seed=0):
    rng = np.random.RandomState(seed)
    X = []
    Y = []
    for i in range(T):
        t = np.arange(seq_len + pred_len)
        # base: sum of sines at different frequencies
        signal = (np.sin(0.02 * t) + 0.5 * np.sin(0.05 * t + 0.3) +
                  0.2 * np.sin(0.2 * t + 1.0))
        # regime: occasional abrupt level shifts
        if rng.rand() < 0.01:
            shift = rng.randn() * 3.0
            signal += shift * (t > seq_len // 2)
        # heteroscedastic noise: larger variance later in sequence
        noise_scale = 0.1 + 0.5 * (t / float(seq_len + pred_len))
        noise = rng.randn(len(t)) * noise_scale
        series = signal + noise
        X.append(series[:seq_len])
        Y.append(series[seq_len:seq_len+pred_len])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='./data/synthetic.npz')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    X, Y = generate_series(T=args.n, seq_len=args.seq_len, pred_len=args.pred_len, seed=args.seed)
    np.savez_compressed(args.out, X=X, Y=Y)
    print(f"Saved synthetic dataset to {args.out}. X.shape={X.shape}, Y.shape={Y.shape}")
