"""Step 4 of the pipeline: from the CNN's per-day PV scores to a commissioning date, and the capacity cues.

Date      : the per-day P(PV active) sequence is smoothed over 7 days and a single upward mean shift
            is fitted (least squares, at least 14 days per segment). The first day of the 'after'
            segment is the commissioning date. If the 'before' mean is already above start_thr, PV is
            taken to predate the data and the first day of the series is returned.
Capacity  : scale-aware cues computed from the days after the date (peak midday reduction in kW
            against a pre-installation baseline, saturation fraction, load level). A ridge regression
            in log space maps them to kWp; it can only be fitted when training labels carry a kWp."""
import numpy as np
import pandas as pd

from config import DATE
from data import day_calendar

MIDW = slice(40, 60)     # 10:00-15:00
SH_AM = slice(24, 32)    # 06:00-08:00
SH_PM = slice(76, 84)    # 19:00-21:00
SUMMER = (4, 9)          # April..September
CAP_FEATS = ["log_peak_red", "sat_frac", "log_scale", "has_baseline"]


def smooth(s, w=None):
    w = DATE["smooth_days"] if w is None else w
    return pd.Series(s).rolling(w, center=True, min_periods=1).mean().to_numpy()


def step_fit(s, min_seg=None, increasing=True):
    min_seg = DATE["min_segment"] if min_seg is None else min_seg
    """Best single mean-shift change point: (split_index, gain, mean_before, mean_after)."""
    s = np.asarray(s, dtype=np.float64); n = len(s)
    if n < 2 * min_seg + 1:
        m = float(s.mean()) if n else 0.0
        return 0, 0.0, m, m
    cs = np.concatenate([[0.0], np.cumsum(s)]); cs2 = np.concatenate([[0.0], np.cumsum(s * s)])
    t = np.arange(min_seg, n - min_seg + 1)
    n1, n2 = t, n - t
    sum1, sum2 = cs[t], cs[n] - cs[t]
    sse = (cs2[t] - sum1 ** 2 / n1) + (cs2[n] - cs2[t] - sum2 ** 2 / n2)
    m1, m2 = sum1 / n1, sum2 / n2
    if increasing:
        sse = np.where(m2 > m1, sse, np.inf)
    if not np.isfinite(sse).any():
        return 0, 0.0, float(s.mean()), float(s.mean())
    i = int(np.argmin(sse)); sse0 = cs2[n] - cs[n] ** 2 / n
    gain = float(1 - sse[i] / sse0) if sse0 > 0 else 0.0
    return int(t[i]), gain, float(m1[i]), float(m2[i])


def estimate_date(days, day_scores, start_thr=None):
    """-> (commissioning day index, diagnostics)"""
    start_thr = DATE["start_threshold"] if start_thr is None else start_thr
    idx, gain, m1, m2 = step_fit(smooth(day_scores))
    if m1 > start_thr:
        return int(days[0]), {"gain": gain, "m1": m1, "m2": m2, "from_start": True}
    return int(days[idx]), {"gain": gain, "m1": m1, "m2": m2, "from_start": False}


def daily_peak_reduction(X, days, d0):
    """Per-day peak midday reduction in kW (NaN before d0). Baseline = same-month pre-installation
    median profile when >= 10 such days exist, otherwise the day's own shoulder level."""
    month, _ = day_calendar(days)
    post = days >= d0; pre = ~post
    peak_red = np.full(len(days), np.nan); has_base = 0.0
    for m in range(1, 13):
        pm, prm = post & (month == m), pre & (month == m)
        if not pm.any():
            continue
        if prm.sum() >= 10:
            base = np.median(X[prm], axis=0)[MIDW]
            red = base[None, :] - X[pm][:, MIDW]; has_base = 1.0
        else:
            sh = 0.5 * (X[pm][:, SH_AM].mean(1) + X[pm][:, SH_PM].mean(1))
            red = sh[:, None] - X[pm][:, MIDW]
        peak_red[pm] = red.max(1) * 4.0
    return peak_red, has_base


def capacity_features(X, days, d0, scale):
    month, _ = day_calendar(days)
    peak_red, has_base = daily_peak_reduction(X, days, d0)
    post = days >= d0
    summer = post & (month >= SUMMER[0]) & (month <= SUMMER[1])
    sel = summer if summer.sum() >= 20 else post
    f = {"log_peak_red": np.log(1e-3), "sat_frac": 0.0, "log_scale": float(np.log(scale)),
         "has_baseline": has_base, "peak_red_kw": 0.0}
    if sel.sum() == 0:
        return f
    pr = peak_red[sel]; ok = np.isfinite(pr)
    p95 = float(np.quantile(pr[ok], 0.95)) if ok.any() else 0.0
    f["log_peak_red"] = float(np.log(max(p95, 1e-3))); f["peak_red_kw"] = p95
    f["sat_frac"] = float(np.mean(X[sel][:, MIDW].min(1) < 0.02 * scale))
    return f
