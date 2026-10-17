"""Second kWp estimate: a neural network over the daily load curves around the commissioning date.

A 1D-CNN reads each day (96 UTC slots, log1p kW); a transformer compares the 56 days before and 336 days after the date;
attention pooling and a small head give log kWp. The estimate is averaged over the date shifted by -14..+14 days and
multiplied by a fitted calibration factor. inference.py blends it 50/50 in logs with the C9v2 kWp.
Inputs: the long series frame from load.read_series, {customer: date}. Output: {customer: kWp}.
Fitted on: synthetic sites (PV-free pool consumption minus public UK PV generation per kWp, openclimatefix/uk_pv,
training only) and then fine-tuned on the 22 labelled PV customers. Training code: nw26_dropped_neurons/src/kWp/train.py.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SLOTS, PRE, STEP = 96, 56, 900
TTA_SHIFTS = (-14, -7, 0, 7, 14)
EPOCH_DAY0 = np.datetime64("1970-01-01")


def utc_days(df: pd.DataFrame) -> dict[str, tuple[int, np.ndarray]]:
    """{customer: (first UTC day number, kWh[day, slot])}; interval start = epoch - 15 min; NaN where not measured."""
    out = {}
    for cid, g in df.groupby("customer_id", sort=False):
        t = g["timestamp_epoch"].to_numpy(np.int64) - STEP
        day, slot, v = t // 86400, (t % 86400) // STEP, g["kwh"].to_numpy(np.float32)
        d0 = int(day.min())
        X = np.full((int(day.max()) - d0 + 1, SLOTS), np.nan, np.float32)
        X[day - d0, slot] = v
        out[str(cid)] = (d0, X)
    return out


def _window(d0: int, X: np.ndarray, t0: int, post: int) -> np.ndarray:
    """Days [t0-PRE, t0+post): 96 x log1p(kW), then measured share, season sin/cos, after-date flag, valid flag."""
    days = np.arange(t0 - PRE, t0 + post)
    idx = days - d0
    ok = (idx >= 0) & (idx < len(X))
    W = np.full((len(days), SLOTS), np.nan, np.float32)
    W[ok] = X[idx[ok]]
    measured = np.isfinite(W)
    load = np.log1p(np.where(measured, W * 4.0, 0.0))
    valid = measured.mean(axis=1) > 0.8
    load[~valid] = 0.0
    dt = EPOCH_DAY0 + days.astype("timedelta64[D]")
    season = (dt - dt.astype("datetime64[Y]")).astype(int) / 365.25 * 2 * math.pi
    rows = np.stack([measured.mean(axis=1), np.sin(season), np.cos(season), (days >= t0).astype(np.float32),
                     valid.astype(np.float32)], axis=1)
    return np.concatenate([load, rows], axis=1).astype(np.float32)


class KwpNet(nn.Module):
    def __init__(self, n_days: int, d: int = 128, layers: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.day = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2, stride=2), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 24, d))
        self.rows = nn.Linear(5, d)
        self.pos = nn.Parameter(torch.zeros(1, n_days, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.attn = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        valid = x[..., -1] > 0.5
        h = self.day(x[..., :SLOTS].reshape(B * T, 1, SLOTS)).reshape(B, T, -1)
        h = h + self.rows(x[..., SLOTS:]) + self.pos[:, :T]
        h = self.tf(h, src_key_padding_mask=~valid)
        w = self.attn(h).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        return self.head((w.unsqueeze(-1) * h).sum(1)).squeeze(-1) + 5.0


@torch.no_grad()
def kwp(weights: str | Path, df: pd.DataFrame, dates: dict[str, pd.Timestamp]) -> dict[str, float]:
    """kWp per customer in `dates` (customers without data are left out)."""
    torch.manual_seed(0)
    ck = torch.load(weights, map_location="cpu")
    post = ck["post"]
    net = KwpNet(PRE + post)
    net.load_state_dict(ck["model"])
    net.eval()
    days = utc_days(df[df["customer_id"].isin(list(dates))])
    cids = [c for c in dates if c in days]
    if not cids:
        return {}
    t0 = {c: int((np.datetime64(pd.Timestamp(dates[c]).date()) - EPOCH_DAY0).astype(int)) for c in cids}
    logs = []
    for s in TTA_SHIFTS:
        x = torch.from_numpy(np.stack([_window(*days[c], t0[c] + s, post) for c in cids]))
        logs.append(torch.cat([net(x[i:i + 16]) for i in range(0, len(x), 16)]).numpy())
    k = np.exp(np.mean(logs, axis=0)) * ck["scale"]
    return {c: float(v) for c, v in zip(cids, k) if np.isfinite(v) and v > 0}
