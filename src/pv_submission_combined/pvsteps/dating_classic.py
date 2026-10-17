"""Classical change-point dating: used by detection (gain) and extensions (t0).

Per day three indicators (log midday/night, log low solar quantile/night, log midday/shoulders), each modelled as yearly
harmonics + weekday dummies + a step weighted by a calendar envelope; the best sign-constrained summed gain wins.
Inputs: DailyMatrix. Output: `date_one` -> (step date, gain). Fitted on: nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import DailyMatrix

MARGIN, FIRST_K = 14, 28  # last candidates are dropped; a date within FIRST_K days of the start becomes the first day


def indicators(dm: DailyMatrix) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """-> (days, Y (D, 3)); NaN on unusable days."""
    x = dm.x
    with np.errstate(all="ignore"):
        night, mid = np.nanmean(x[:, :20], 1), np.nanmean(x[:, 40:60], 1)
        q = np.nanquantile(x[:, 36:64], 0.1, axis=1)
        tot = np.nansum(x, 1)
        okcov = dm.coverage.mean(1) > 0.9
        eps = 0.05 * np.nanmedian(night) + 1e-4
        sh = np.nanmean(np.c_[x[:, 28:36], x[:, 68:76]], 1)
        Y = np.c_[np.log((mid + eps) / (night + eps)), np.log((q + eps) / (night + eps)), np.log((mid + eps) / (sh + eps))]
        bad = ~okcov | (tot < 0.25 * np.nanmedian(tot[okcov])) if okcov.any() else ~okcov
    Y[bad] = np.nan
    return dm.days, Y


def envelope(days: pd.DatetimeIndex, floor: float) -> np.ndarray:
    """Calendar weight: `floor` at the summer solstice, 1 at the winter solstice."""
    c = np.cos(2 * np.pi * (days.dayofyear.to_numpy() - 172) / 365.25)
    return floor + (1 - floor) * (1 - c) / 2


def basis(days: pd.DatetimeIndex, harm: int) -> np.ndarray:
    """Intercept, `harm` yearly harmonics and six weekday dummies."""
    doy = days.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    cols = [np.ones(len(days))]
    for h in range(1, harm + 1):
        cols += [np.cos(h * doy), np.sin(h * doy)]
    w = days.dayofweek.to_numpy()
    return np.column_stack(cols + [(w == k).astype(float) for k in range(1, 7)])


def scan(days: pd.DatetimeIndex, Y: np.ndarray, harm: int = 2, env_floor: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """-> (summed gain per candidate day, NaN where not allowed; mask of usable days)."""
    ok = np.isfinite(Y).all(1)
    n = int(ok.sum())
    gain = np.full(len(days), np.nan)
    if n < 60:
        return gain, ok
    d = days[ok]
    Q, _ = np.linalg.qr(basis(d, harm))
    s = envelope(d, env_floor)
    idx = np.arange(n)
    G = (idx[:, None] >= idx[None, :]) * s[:, None]
    Gr = G - Q @ (Q.T @ G)
    gg = (Gr ** 2).sum(0) + 1e-12
    tot = np.zeros(n)
    for k in range(Y.shape[1]):
        yk = Y[ok, k]
        yr = yk - Q @ (Q.T @ yk)
        sig2 = np.var(yr) + 1e-12
        num = yr @ Gr
        b = num / gg
        tot += np.where(np.sign(b) == -1.0, num ** 2 / gg, 0.0) / sig2
    tot[idx >= n - MARGIN] = -np.inf
    gain[ok] = tot
    return gain, ok


def date_one(dm: DailyMatrix) -> tuple[pd.Timestamp, float]:
    """-> (step date, gain of the best step)."""
    days, Y = indicators(dm)
    if len(days) == 0:
        raise ValueError("empty series")
    gain, _ = scan(days, Y)
    if not np.isfinite(gain).any():
        return days[0], 0.0
    g = np.where(np.isfinite(gain), gain, -np.inf)
    j = int(np.argmax(g))
    t = days[j]
    if (t - days[0]).days <= FIRST_K:
        t = days[0]
    return t, float(g[j])
