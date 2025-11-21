
import torch
import torch.nn as nn

class TransformerQuantileModel(nn.Module):
    def __init__(self, seq_len=120, d_model=64, nhead=4, num_layers=3, pred_len=24):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # head outputs mean and log-variance per future time step
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len * 2)  # predict mu and log_var for each pred step
        )
    def forward(self, x):
        # x: (B, seq_len)
        B, L = x.shape
        x = x.unsqueeze(-1)  # (B, L, 1)
        h = self.input_proj(x)  # (B, L, d_model)
        # transformer expects (L, B, d_model)
        h = h.permute(1,0,2)
        h = self.encoder(h)  # (L, B, d_model)
        h = h.permute(1,2,0)  # (B, d_model, L)
        pooled = self.pool(h).squeeze(-1)  # (B, d_model)
        out = self.head(pooled)  # (B, pred_len*2)
        out = out.view(B, -1, 2)  # (B, pred_len, 2)
        mu = out[...,0]
        log_var = out[...,1]
        return mu, log_var
