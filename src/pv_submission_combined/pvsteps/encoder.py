"""Pretrained day encoder: a small 1D-CNN, 96-slot day -> 16-number embedding.

Days are divided by the customer's own robust level, so it works on an unseen customer from its own series alone.
Inputs: float32 days x 96 (kWh per slot) of one customer. Outputs: float32 days x 16.
Fitted on: label-free masked-autoencoder training (not shipped); weights in `model/encoder_shipped.pt`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SLOTS, EMB_DIM = 96, 16
CLIP_LO, CLIP_HI = -3.0, 8.0  # the shipped weights use the 'linear' normalisation


class Encoder(nn.Module):
    def __init__(self, emb_dim: int = EMB_DIM):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(2, 32, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.GELU())
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * (SLOTS // 4), 128), nn.GELU(), nn.Linear(128, emb_dim))

    def forward(self, x, mask):
        return self.head(self.conv(torch.stack([x, mask], dim=1)))


def load_encoder(path: str | Path) -> Encoder:
    """Load the frozen weights (weights_only, CPU, eval mode)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    enc = Encoder(ck["emb_dim"])
    enc.load_state_dict(ck["state_dict"])
    return enc.eval()


def customer_scale(X: np.ndarray) -> float:
    """Robust per-customer level: median over days of the daily mean kWh per slot (floored)."""
    return max(float(np.median(X.mean(axis=1))), 0.05 * float(np.mean(np.abs(X))) + 1e-3)


@torch.no_grad()
def encode_days(enc: Encoder, X: np.ndarray, scale: float, batch: int = 4096) -> np.ndarray:
    """Embed the days of one customer: (n, 96) kWh -> (n, 16) float32. No masking at inference."""
    Xn = np.clip(X / scale, CLIP_LO, CLIP_HI).astype(np.float32)
    out = []
    for i in range(0, len(Xn), batch):
        x = torch.from_numpy(Xn[i:i + batch])
        out.append(enc(x, torch.zeros_like(x)).numpy())
    return np.concatenate(out).astype(np.float32)
