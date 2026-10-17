"""Year-over-year kWp refinement, applied on top of the base kWp at the chosen commissioning date.

Peak deficit: for each post-install day (first 364 days) the 28 daily slots 09-16h are compared with the median of five
reference days one year earlier (364 +- 14 days, all before the date), night-load matched. The 90th percentile of the daily
best 1 h deficit, times 4, is `peak` (kW). log kWp = log(peak) + ridge(zero share at midday) + log(bayes factor). The result
is blended 50/50 in log space with the base kWp. Without enough evidence the base kWp is returned unchanged.
Inputs: DailyMatrix, date, base kWp, state (state["yoy"]: mu, sd, b, w, factor). Output: kWp float. Relies on inference.py muting RuntimeWarning (all-NaN slices).
Fitted on: the 40 labelled dev customers (ridge offset on midday zero share; `factor` is the distributional utility-optimal
multiplier computed from the leave-one-out error `sigma`, stored as a number).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import DailyMatrix

CAP, MID, NIGHT = np.arange(36, 64), np.arange(40, 56), np.r_[0:20, 84:96]
MIN_DAYS, BLEND = 30, 0.5


def peak_deficit(dm: DailyMatrix, t0: pd.Timestamp) -> dict | None:
    """-> dict(peak, zero, n_days), or None when the year-over-year comparison is not identified."""
    days = pd.DatetimeIndex(dm.days)
    x = np.asarray(dm.x, float)
    if len(days) == 0 or pd.isna(t0) or x.shape != (len(days), 96):
        return None
    t0 = pd.Timestamp(t0)
    coverage = np.asarray(dm.coverage, bool) & np.isfinite(x)
    good = (coverage[:, CAP].mean(1) >= 0.9) & (coverage[:, NIGHT].mean(1) >= 0.8)
    good &= ~np.asarray(dm.dst_day, bool)
    night = np.nanmedian(np.where(coverage[:, NIGHT], x[:, NIGHT], np.nan), axis=1)
    good &= np.isfinite(night) & (night > 1e-9)
    post = np.flatnonzero(good & (days >= t0) & (days < t0 + pd.Timedelta(days=364)))
    if len(post) < MIN_DAYS:
        return None
    stack = np.full((5, len(post), len(CAP)), np.nan)
    for j, k in enumerate(range(-2, 3)):
        ref_days = days[post] - pd.Timedelta(days=364 - 7 * k)
        idx = days.get_indexer(ref_days)
        found = idx >= 0
        safe = np.maximum(idx, 0)
        ratio = night[post] / np.maximum(night[safe], 1e-9)
        use = found & good[safe] & (ref_days < t0) & (ratio >= 1 / 3) & (ratio <= 3)
        pos = np.flatnonzero(use)
        v = x[safe[pos]][:, CAP] * ratio[pos, None]
        stack[j, pos] = np.where(coverage[safe[pos]][:, CAP], v, np.nan)
    ref = np.nanmedian(stack, axis=0)
    ref[np.isfinite(stack).sum(0) < 3] = np.nan
    diff = np.where(coverage[post][:, CAP], ref - x[post][:, CAP], np.nan)
    win = np.lib.stride_tricks.sliding_window_view(diff, 4, axis=1)
    smooth = np.nanmean(win, axis=-1)
    smooth[np.isfinite(win).sum(-1) < 3] = np.nan
    daily = np.nanmax(smooth, axis=1)
    usable = (np.isfinite(diff).sum(1) >= 20) & np.isfinite(daily)
    if usable.sum() < MIN_DAYS:
        return None
    peak = float(4 * np.quantile(daily[usable], 0.9))
    if not (np.isfinite(peak) and peak > 0):
        return None
    mid, obs = x[post[usable]][:, MID], coverage[post[usable]][:, MID]
    return dict(peak=peak, zero=float(np.mean(mid[obs] <= 1e-9)) if obs.any() else 0.0, n_days=int(usable.sum()))


def kwp_at(yoy: dict | None, dm: DailyMatrix, date: pd.Timestamp, base: float) -> float:
    """Blend of the base kWp and the year-over-year estimate at `date`; `base` if the evidence is missing."""
    if yoy is None:
        return base
    f = peak_deficit(dm, pd.Timestamp(date))
    if f is None or not (np.isfinite(base) and base > 0):
        return base
    offset = yoy["b"] + yoy["w"] * (f["zero"] - yoy["mu"]) / yoy["sd"]
    log_yoy = np.log(f["peak"]) + offset + np.log(yoy["factor"])
    merged = float(np.exp((1 - BLEND) * np.log(base) + BLEND * log_yoy))
    return merged if np.isfinite(merged) and merged > 0 else base
