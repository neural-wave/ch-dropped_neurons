"""Detection: rule detector plus bell gate.

PV if log(1 + classical dating gain) >= a fitted threshold, or if the largest 56-day share of midday zero readings
reaches the static threshold (catches PV that predates the data). The gate then drops a detected customer whose post/pre
change is not bell-shaped around noon (shoulder ratio minus midday ratio < BELL_THR), unless the static path fired.
Inputs: {cid: DailyMatrix}, detector state (state["live"]["det"]). Outputs: `rule_detect` -> dict(pv, static, date); `bell_gate` -> ids to drop.
Fitted on: the 40 labelled customers (F1-optimal threshold); BELL_THR set from pool anchors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dating_classic
from .load import DailyMatrix

BELL_THR = 0.05
SHOULDER, MID = np.array([8, 9, 16, 17]), np.arange(11, 14)  # hours of day


def _alive(dm: DailyMatrix) -> bool:
    """Positive night load and night rarely zero (rules out dead meters)."""
    n = dm.x[:, :20]
    ok = np.isfinite(n)
    return bool(ok.any() and np.nanmedian(n) > 0 and (n[ok] <= 0).mean() < 0.1)


def _zero_share(dm: DailyMatrix, win: int = 56, step: int = 28) -> float:
    """Largest 56-day-window share of midday (11-15h) readings at or below zero; NaN if no window has enough coverage."""
    x, best = dm.x, []
    # outage days (night max <= 0 or not finite) are not PV evidence: PV cannot zero the night
    nmax = np.nanmax(np.where(np.isfinite(x[:, :20]), x[:, :20], -np.inf), axis=1)
    x = np.where((~np.isfinite(nmax) | (nmax <= 0))[:, None], np.nan, x)
    for a in range(0, max(1, len(dm.days) - win + 1), step):
        w = x[a:a + win]
        if np.isfinite(w).mean() < 0.6:
            continue
        zb = w[:, 44:60]
        ok = np.isfinite(zb)
        best.append(float((zb[ok] <= 0).mean()) if ok.any() else np.nan)
    return float(pd.Series(best, dtype=float).max()) if best else np.nan


def rule_detect(det: dict, dms: dict[str, DailyMatrix]) -> dict:
    """-> dict(pv=set of ids, static=set of ids where the static path fired, date={cid: classical date}). Customers dating cannot handle are skipped."""
    date, score, zs = {}, {}, {}
    for cid, dm in dms.items():
        try:
            t, gain = dating_classic.date_one(dm)
        except Exception:  # empty or degenerate customer: handled as not PV
            continue
        date[cid], score[cid] = t, np.log1p(gain)
        zs[cid] = _zero_share(dm) if _alive(dm) else np.nan
    thr = det["sthr"]["static_zs"]
    static = {c for c, v in zs.items() if np.isfinite(v) and v >= thr}
    return dict(pv={c for c, s in score.items() if s >= det["thr"]} | static, static=static, date=date)


def _bell(dm: DailyMatrix, t0: pd.Timestamp) -> float:
    """Mean (post/pre) ratio of shoulder hours minus that of midday hours over months seen on both sides of t0 (Mar-Oct)."""
    d = pd.DatetimeIndex(dm.days)
    h = np.nanmean(dm.x.reshape(len(dm.x), 24, 4), axis=2)
    pre, post = d < t0 - pd.Timedelta(days=14), d >= t0 + pd.Timedelta(days=14)
    rs = []
    for m in range(3, 11):
        a, b = np.asarray(pre & (d.month == m)), np.asarray(post & (d.month == m))
        if a.sum() < 10 or b.sum() < 10:
            continue
        pa, pb = np.nanmean(h[a], axis=0), np.nanmean(h[b], axis=0)
        if np.all(pa > 0):
            rs.append(pb / pa)
    if not rs:
        return np.nan
    r = np.mean(rs, axis=0)
    return float(r[SHOULDER].mean() - r[MID].mean())


def bell_gate(dms: dict[str, DailyMatrix], found: dict) -> set[str]:
    """Ids of detected customers to switch off (no pre/post month overlap keeps the decision)."""
    off = set()
    for c in sorted(found["pv"] - found["static"]):
        if c in dms:
            b = _bell(dms[c], pd.Timestamp(found["date"][c]))
            if np.isfinite(b) and b < BELL_THR:
                off.add(c)
    return off
