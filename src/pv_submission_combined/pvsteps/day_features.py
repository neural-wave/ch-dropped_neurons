"""Per-day feature table of one customer: embedding e0..e15, day, day-of-year sin/cos and scale-free midday cues.

Inputs: X (days x 96 kWh), days (int day index of the encoder grid), emb (days x 16), scale (customer level).
Output: DataFrame, one row per day. Fitted on: nothing. Calendar only, no weather, location or solar geometry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import BASE

EMB_COLS = [f"e{i}" for i in range(16)]
DAY_FEATS = EMB_COLS + ["doy_sin", "doy_cos", "dip_ms", "dip_mn", "min_mid", "level"]  # columns the day model reads, in order
MID, MIDW, NIGHT = slice(44, 56), slice(40, 60), slice(0, 20)  # 11-14h, 10-15h, 00-05h
SH_AM, SH_PM = slice(24, 32), slice(76, 84)  # shoulders 06-08h, 19-21h


def day_table(X: np.ndarray, days: np.ndarray, emb: np.ndarray, scale: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex(BASE + np.asarray(days).astype("timedelta64[D]"))
    doy = idx.dayofyear.values.astype(np.int16)
    mid, night = X[:, MID].mean(1), X[:, NIGHT].mean(1)
    sh = 0.5 * (X[:, SH_AM].mean(1) + X[:, SH_PM].mean(1))
    tab = pd.DataFrame(emb, columns=EMB_COLS)
    tab["day"] = days
    tab["doy_sin"], tab["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    tab["dip_ms"] = (mid - sh) / scale
    tab["dip_mn"] = (mid - night) / scale
    tab["min_mid"] = X[:, MIDW].min(1) / scale
    tab["level"] = X.mean(1) / scale
    return tab
