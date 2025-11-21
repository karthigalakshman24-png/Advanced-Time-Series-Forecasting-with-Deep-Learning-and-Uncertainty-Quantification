
# Advanced Time Series Forecasting — Full Solution

**Contents**:
- `data_simulation.py` — creates a high-fidelity synthetic dataset (heteroscedastic + regime changes).
- `model.py` — Transformer-based sequence model that outputs mean and log-variance (heteroscedastic Gaussian).
- `train.py` — training loop with logging, checkpointing, and scheduler.
- `evaluate.py` — computes RMSE, MAE, NLL, 95% coverage, and CRPS (analytic for Gaussian).
- `utils.py` — data loaders and helper functions.
- `requirements.txt` — pin of primary packages.
- `notebooks/` — lightweight Jupyter notebook starter for experimentation.
- `assets/` — includes the screenshot provided by the user for reference.
- `LICENSE` — MIT.

**How to run** (example):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python data_simulation.py --out ./data/synthetic.npz
python train.py --data ./data/synthetic.npz --epochs 5 --save_dir ./checkpoints
python evaluate.py --ckpt ./checkpoints/best.pth --data ./data/synthetic.npz
```

This repository implements probabilistic forecasting using a heteroscedastic Gaussian NLL loss (model predicts mean and log-variance).
The solution includes quantitative evaluation of calibration via interval coverage and CRPS.
