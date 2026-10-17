"""Feature engineering on top of the self-supervised embeddings.

Three levels:
  day_table        : one row per day (embedding + calendar + scale-free physics cues)
  customer_features: one row per customer, built from the day-level PV score sequence
  capacity_features: absolute-scale cues (kW of midday reduction) for kWp regression
"""
import os
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from config import EMB_DIM
from data import day_calendar, day_to_date

MID = slice(44, 56)      # 11:00-14:00
MIDW = slice(40, 60)     # 10:00-15:00
NIGHT = slice(0, 20)     # 00:00-05:00
SH_AM = slice(24, 32)    # 06:00-08:00
SH_PM = slice(76, 84)    # 19:00-21:00
SUMMER = (4, 9)          # April..September (inclusive)

EMB_COLS = [f"e{i}" for i in range(EMB_DIM)]
PX_COLS = [f"px{i}" for i in range(96)]            # heatmap pixels of the day (kW / p95, gamma 0.6)
PXS_COLS = [f"pxs{i}" for i in range(96)]          # same, averaged over +-7 days
LIN_FEATS = PX_COLS + PXS_COLS + ["doy_sin", "doy_cos"]
HEAT_GAMMA = 0.6
DAY_FEATS = EMB_COLS + ["doy_sin", "doy_cos", "dip_ms", "dip_mn", "min_mid", "level"]
if os.environ.get("PV_RS_FEATS") == "1":          # experiment: relsmooth day cues for the day model
    DAY_FEATS = DAY_FEATS + ["rs_mid", "rs_min"]
SHOULDER = np.r_[24:36, 68:80]                     # 06:00-09:00 and 17:00-20:00 (relsmooth reference)
CUST_FEATS = ["s_mean", "s_q90", "s_max60", "s_step_gain", "s_step_delta",
              "dip_ms_median", "dip_ms_min_month", "midday_low_frac", "summer_ratio_t", "recent_ratio_t"]
# also computed by customer_features (available for experiments): dip_season_amp, dip_season_amp_rise,
# summer_dip_change, summer_dip_t
CAP_FEATS = ["log_peak_red", "sat_frac", "log_scale", "has_baseline"]


def relsmooth_image(X, days, size=(15, 3)):
    """log2(kW) minus the day's own shoulder level, then a (days x slots) median filter on the dense
    day grid (missing days filled with 0 before filtering). Returns rows for the given days."""
    lk = np.log2(np.maximum(X, 1e-3) * 4.0)
    Z = lk - lk[:, SHOULDER].mean(1, keepdims=True)
    d0 = int(days.min()); grid = np.zeros((int(days.max()) - d0 + 1, X.shape[1]), np.float32)
    grid[days - d0] = Z
    grid = median_filter(grid, size=size)
    return grid[days - d0].astype(np.float32)


def smooth(s, w=7):
    return pd.Series(s).rolling(w, center=True, min_periods=1).mean().to_numpy()


def step_fit(s, min_seg=14, increasing=True):
    """Best single mean-shift change point. Returns (split_index, gain, mean_before, mean_after).
    split_index = first index of the 'after' segment."""
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


def day_table(X, days, emb, scale, sun=None):
    """Per-day feature table. X raw kWh (n,96); emb (n,EMB_DIM); scale = customer scale;
    sun = per-day sun index aligned with days (None -> constant 1)."""
    month, doy = day_calendar(days)
    mid = X[:, MID].mean(1); night = X[:, NIGHT].mean(1)
    sh = 0.5 * (X[:, SH_AM].mean(1) + X[:, SH_PM].mean(1))
    tab = pd.DataFrame(emb, columns=EMB_COLS)
    tab["day"] = days; tab["month"] = month; tab["doy"] = doy
    tab["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); tab["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    tab["dip_ms"] = (mid - sh) / scale
    eps = 0.02 * max(float(np.median(X)), 1e-3)
    tab["lr_ms"] = np.log((np.maximum(mid, 0) + eps) / (np.maximum(sh, 0) + eps))   # scale-free midday/shoulder
    tab["dip_mn"] = (mid - night) / scale
    tab["min_mid"] = X[:, MIDW].min(1) / scale
    tab["level"] = X.mean(1) / scale
    tab["midday_low"] = (mid < 0.5 * sh).astype(np.float32)
    rs = relsmooth_image(X, days)
    tab["rs_mid"] = rs[:, MID].mean(1); tab["rs_min"] = rs[:, MIDW].min(1)
    tab["sun"] = np.ones(len(days), np.float32) if sun is None else sun.astype(np.float32)
    return tab


def heat_pixels(X):
    """heatmap representation of the days: per-customer scale = 95th percentile, gamma-stretched low end"""
    p95 = max(float(np.percentile(X, 95)), 1e-6)
    return (np.clip(X / p95, 0, 1) ** HEAT_GAMMA).astype(np.float32)


RS_CLIP = 3.0        # relsmooth image: log2 ratios clipped to +-3 (factor 8) and scaled to [-1, 1]
IMAGE_CHANNELS = {"heat": 1, "relsmooth": 1, "both": 2}


def relsmooth_pixels(X, days):
    """relsmooth view as an image in [-1, 1]: negative = less draw at midday than at the shoulders (PV)."""
    return (np.clip(relsmooth_image(X, days), -RS_CLIP, RS_CLIP) / RS_CLIP).astype(np.float32)


def customer_image(X, days, mode="heat"):
    """(C, L, 96) image channels of one customer. mode: 'heat' (linear kW pixels), 'relsmooth', or 'both'."""
    if mode == "heat":
        return heat_pixels(X)[None]
    if mode == "relsmooth":
        return relsmooth_pixels(X, days)[None]
    if mode == "both":
        return np.stack([heat_pixels(X), relsmooth_pixels(X, days)])
    raise ValueError(mode)


def add_pixels(tab, X):
    """append the heatmap pixels (and a +-7 day smoothed copy) as columns, for the linear mode"""
    px = heat_pixels(X)
    pxs = pd.DataFrame(px).rolling(15, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)
    extra = pd.DataFrame(np.concatenate([px, pxs], axis=1), columns=PX_COLS + PXS_COLS, index=tab.index)
    return pd.concat([tab, extra], axis=1)


def customer_features(tab):
    """Customer-level feature vector from a day table that has a 'score' column."""
    s = tab["score"].to_numpy(dtype=np.float64)
    f = {"s_mean": s.mean(), "s_q90": float(np.quantile(s, 0.9)),
         "s_max60": float(pd.Series(s).rolling(60, min_periods=min(30, len(s))).mean().max())}
    _, gain, m1, m2 = step_fit(smooth(s))
    f["s_step_gain"] = gain; f["s_step_delta"] = m2 - m1
    f["dip_ms_median"] = float(tab["dip_ms"].median())
    mm = tab.groupby("month")["dip_ms"].agg(["median", "size"])
    mm = mm[mm["size"] >= 15]
    f["dip_ms_min_month"] = float(mm["median"].min()) if len(mm) else f["dip_ms_median"]
    f["midday_low_frac"] = float(tab["midday_low"].mean())
    # Seasonality of the midday dip. PV: dip deep in summer, shallow in winter, and that contrast
    # appears at some point. A lunch break or a daytime-light load has the same dip all year.
    m = tab["month"].to_numpy(); dip = tab["dip_ms"].to_numpy(); yr = pd.DatetimeIndex(day_to_date(tab["day"].to_numpy())).year.values
    summer, winter = np.isin(m, [5, 6, 7, 8]), np.isin(m, [11, 12, 1, 2])
    amp = lambda sel_s, sel_w: float(np.median(dip[sel_w]) - np.median(dip[sel_s])) if (sel_s.sum() >= 15 and sel_w.sum() >= 15) else np.nan
    a_all = amp(summer, winter); f["dip_season_amp"] = 0.0 if np.isnan(a_all) else a_all
    amps, summers = [], []
    for y in np.unique(yr):
        sy, wy = summer & (yr == y), winter & (yr == y)
        a = amp(sy, wy)
        if not np.isnan(a):
            amps.append(a)
        if sy.sum() >= 30:
            summers.append(float(np.median(dip[sy])))
    f["dip_season_amp_rise"] = (max(amps) - amps[0]) if len(amps) >= 2 else 0.0
    f["summer_dip_change"] = (min(summers) - summers[0]) if len(summers) >= 2 else 0.0
    # significance of that change: (first summer median - lowest later summer median) in units of the
    # customer's own day-to-day spread (MAD), times sqrt(n). A big stable load makes a small dip significant.
    f["summer_dip_t"] = 0.0
    years = [y for y in np.unique(yr) if (summer & (yr == y)).sum() >= 30]
    if len(years) >= 2:
        first = dip[summer & (yr == years[0])]
        best_t = 0.0
        for y in years[1:]:
            later = dip[summer & (yr == y)]
            mad = 1.4826 * np.median(np.abs(np.concatenate([first - np.median(first), later - np.median(later)]))) + 1e-6
            t = (np.median(first) - np.median(later)) / mad * np.sqrt(min(len(first), len(later)))
            best_t = max(best_t, float(t))
        f["summer_dip_t"] = best_t
    # Same test on log(midday/shoulder): immune to a uniform change of the whole load (a business slowing
    # down), still sensitive to a midday-only reduction. Reported as log1p of the t statistic.
    lr = tab["lr_ms"].to_numpy(); best = 0.0
    if len(years) >= 2:
        first = lr[summer & (yr == years[0])]
        for y in years[1:]:
            later = lr[summer & (yr == y)]
            mad = 1.4826 * np.median(np.abs(np.concatenate([first - np.median(first), later - np.median(later)]))) + 1e-6
            best = max(best, float((np.median(first) - np.median(later)) / mad * np.sqrt(min(len(first), len(later)))))
    f["summer_ratio_t"] = float(np.log1p(best))
    # Late installs: compare the last 90 days of the series with the same calendar window in each
    # earlier year (same season, so seasonality cancels). Significance of the drop in log(midday/shoulder).
    days = tab["day"].to_numpy(); doy = tab["doy"].to_numpy().astype(int)
    last = days >= days.max() - 89; best = 0.0
    if last.sum() >= 40:
        recent = lr[last]; doys = set(doy[last].tolist())
        for y in np.unique(yr[~last]):
            prev = (yr == y) & ~last & np.isin(doy, list(doys))
            if prev.sum() >= 40:
                earlier = lr[prev]
                mad = 1.4826 * np.median(np.abs(np.concatenate([earlier - np.median(earlier), recent - np.median(recent)]))) + 1e-6
                best = max(best, float((np.median(earlier) - np.median(recent)) / mad * np.sqrt(min(len(earlier), len(recent)))))
    f["recent_ratio_t"] = float(np.log1p(best))
    return f


def estimate_date(tab, signal="score", start_thr=0.5):
    """Change-point date (int day index) from a day table. signal: 'score' or 'dip'."""
    s = tab["score"].to_numpy() if signal == "score" else -tab["dip_ms"].to_numpy()
    days = tab["day"].to_numpy()
    idx, gain, m1, m2 = step_fit(smooth(s))
    if signal == "score" and m1 > start_thr:      # PV already present when the series starts
        return int(days[0]), {"gain": gain, "m1": m1, "m2": m2, "from_start": True}
    return int(days[idx]), {"gain": gain, "m1": m1, "m2": m2, "from_start": False}


def daily_peak_reduction(X, days, d0):
    """Per-day peak midday reduction in kW relative to a baseline (NaN before d0).
    Baseline = same-month pre-installation median profile when >=10 such days exist,
    otherwise the day's own morning/evening shoulder level."""
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
        peak_red[pm] = red.max(1) * 4.0           # kWh per 15 min -> kW
    return peak_red, has_base


def capacity_features(X, days, d0, scale, sun=None):
    """Absolute-scale cues for kWp. sun: per-day sun index aligned with days (None -> no index)."""
    month, _ = day_calendar(days)
    peak_red, has_base = daily_peak_reduction(X, days, d0)
    post = days >= d0
    summer = post & (month >= SUMMER[0]) & (month <= SUMMER[1])
    sel = summer if summer.sum() >= 20 else post
    f = {"log_peak_red": np.log(1e-3), "sat_frac": 0.0, "log_scale": float(np.log(scale)),
         "has_baseline": has_base, "peak_red_kw": 0.0, "log_cap_sun": np.log(1e-3), "cap_sun_kw": 0.0}
    if sel.sum() == 0:
        return f
    pr = peak_red[sel]; ok = np.isfinite(pr)
    p95 = float(np.quantile(pr[ok], 0.95)) if ok.any() else 0.0
    f["log_peak_red"] = float(np.log(max(p95, 1e-3))); f["peak_red_kw"] = p95
    f["sat_frac"] = float(np.mean(X[sel][:, MIDW].min(1) < 0.02 * scale))
    if sun is not None:
        sunny = post & np.isfinite(peak_red) & (sun >= 0.7)
        if sunny.sum() >= 5:
            cs = float(np.median(peak_red[sunny] / sun[sunny]))
            f["log_cap_sun"] = float(np.log(max(cs, 1e-3))); f["cap_sun_kw"] = cs
        else:                                      # too few sunny days: fall back to the p95 cue
            f["log_cap_sun"] = f["log_peak_red"]; f["cap_sun_kw"] = p95
    else:
        f["log_cap_sun"] = f["log_peak_red"]; f["cap_sun_kw"] = p95
    return f
