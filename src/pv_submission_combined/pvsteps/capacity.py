"""Base kWp (DC) from the before/after deficit around the customer's own step date.

Matched (month, weekday/weekend) cells compare pre and post mean profiles. Three cues enter a ridge on log kWp: `lKD`
(drift-corrected deficit over unclipped solar hours / pool-learned deficit per kWp), `lKE2` (pre-install load at the edge
of the midday zero run), `zf_sh` (midday zero share in spring/autumn); one ridge per availability pattern of the cues.
Inputs: DailyMatrix, the capacity state (state["live"]["cap"]). Output: kWp float (raises if the series is unusable).
Fitted on: the 40 labelled customers at their true dates. Calendar cells only, no location or irradiance.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .load import DailyMatrix

SOLAR, MID, NIGHT = slice(28, 69), slice(44, 53), slice(0, 20)
SPRING_AUTUMN = (3, 4, 9, 10)
FEATS = ["lKD", "lKE2", "zf_sh"]
MINDAYS, NB, ZMAX, DRIFT_L = 3, 10, 0.15, 0.01


def own_date(dm: DailyMatrix) -> pd.Timestamp:
    """Single step in the weekly log(midday/night) with 3 yearly harmonics, best SSE over a weekly grid."""
    mid, ngt = np.nanmean(dm.x[:, MID], axis=1), np.nanmean(dm.x[:, NIGHT], axis=1)
    ok = np.isfinite(mid) & np.isfinite(ngt) & (ngt > 0) & (mid > 0)
    d = np.asarray(dm.days)[ok]
    if len(d) < 60:
        return pd.Timestamp(dm.days[len(dm.days) // 2])
    y = np.log(mid[ok] / ngt[ok])
    t = (d - d[0]).astype("timedelta64[D]").astype(float)
    w = 2 * np.pi * t / 365.25
    H = np.column_stack([np.ones_like(t)] + [f(k * w) for k in (1, 2, 3) for f in (np.sin, np.cos)])
    best = (np.inf, d[len(d) // 2])
    for i in range(30, len(t) - 30, 7):
        A = np.column_stack([H, (np.arange(len(t)) >= i).astype(float)])
        b, *_ = np.linalg.lstsq(A, y, rcond=None)
        if b[-1] > 0:
            continue  # PV lowers the midday/night ratio
        sse = float(((A @ b - y) ** 2).sum())
        if sse < best[0]:
            best = (sse, d[i])
    return pd.Timestamp(best[1])


def _cells(dm: DailyMatrix, t0: pd.Timestamp):
    """Yield (month, pre mean profile, post mean profile, n days, post days) for matched (month, weekend) cells."""
    days = pd.DatetimeIndex(dm.days)
    ok = (dm.coverage[:, SOLAR].mean(axis=1) >= 0.9) & ~dm.dst_day
    post, wk = days >= t0, days.dayofweek >= 5
    for m in range(1, 13):
        for w in (False, True):
            sel = ok & (days.month == m) & (wk == w)
            a, b = sel & ~post, sel & post
            if a.sum() >= MINDAYS and b.sum() >= MINDAYS:
                yield m, np.nanmean(dm.x[a], axis=0), np.nanmean(dm.x[b], axis=0), int(a.sum() + b.sum()), dm.x[b]


def cell_features(dm: DailyMatrix, t0: pd.Timestamp) -> dict:
    """Raw per-cell cues: zero shares (`zf`, weights `n`, months `m`), drift bins `b2` and zero-run edge loads `edge`."""
    zf, n, mo, b2, edge = [], [], [], [], []
    for m, pre, post, k, xb in _cells(dm, t0):
        zf.append(float((xb[:, MID] <= 1e-9).mean())); n.append(k); mo.append(m)
        bins = lambda v: v[28:68].reshape(NB, 4).sum(1)
        b2.append((m, bins(pre - post), (xb[:, 28:68] <= 1e-9).mean(0).reshape(NB, 4).mean(1), bins(pre),
                   float((pre - post)[NIGHT].sum()), float(pre[NIGHT].sum())))
        for day in xb:  # first zero slot of the midday zero run: production there ~ counterfactual load
            if day[44:53].min() <= 1e-9:
                t = 44
                while t > 28 and day[t - 1] <= 1e-9:
                    t -= 1
                if t > 28:
                    edge.append((m, (t - 28) // 4, float(bins(pre)[(t - 28) // 4])))
    return dict(zf=np.array(zf), n=np.array(n, float), mo=np.array(mo), b2=b2, edge=edge)


def khat_drift(b2: list, g: np.ndarray) -> float:
    """Per customer solve d = k*g[m,bin] + c*pre over unclipped solar bins plus night (g=0 pins c); NaN if unidentified."""
    A, y = [], []
    for m, d, z, pre, dn, pn in b2:
        for b in range(NB):
            if z[b] <= ZMAX and g[m - 1, b] > 0:
                A.append((g[m - 1, b], pre[b])); y.append(d[b])
        A.append((0.0, pn)); y.append(dn)
    if len(y) < 6:
        return np.nan
    A, y = np.array(A), np.array(y)
    sc = np.array([1.0, np.abs(A[:, 1]).mean() + 1e-9])
    M = A / sc
    if np.linalg.matrix_rank(M[:, :1]) < 1:
        return np.nan
    G = M.T @ M + np.diag([0.0, DRIFT_L]) * len(y)
    if not np.isfinite(G).all() or np.linalg.cond(G) > 1e12:
        return np.nan
    return float(np.linalg.solve(G, M.T @ y)[0] / sc[0])


def khat_edge(edge: list, g: np.ndarray, q: float = 0.8) -> float:
    """Per month upper quantile of pre-load(edge bin)/g(m,bin); median over months with >= 3 days."""
    by_month = {}
    for m, b, load in edge:
        if g[m - 1, b] > 0:
            by_month.setdefault(m, []).append(load / g[m - 1, b])
    v = [np.quantile(a, q) for a in by_month.values() if len(a) >= 3]
    return float(np.median(v)) if v else np.nan


def kwp(state: dict, dm: DailyMatrix) -> float:
    c = cell_features(dm, own_date(dm))
    f = {}
    if len(c["zf"]) and np.isin(c["mo"], SPRING_AUTUMN).any():
        k = np.isin(c["mo"], SPRING_AUTUMN)
        f["zf_sh"] = float(np.average(c["zf"][k], weights=c["n"][k]))
    ke = khat_edge(c["edge"], state["g"])
    if np.isfinite(ke):
        f["lKE2"] = np.log(max(ke, 1.0))
    kd = khat_drift(c["b2"], state["g"])
    if np.isfinite(kd) and kd > 0:
        f["lKD"] = np.log(max(kd, 1.0))
    av = [i for i, name in enumerate(FEATS) if name in f]
    hit = next((S for r in range(len(av), 0, -1) for S in itertools.combinations(av, r) if S in state["sub"]), None)
    raw = state["b"]
    if hit is not None:
        m_, s_, ws, b_ = state["sub"][hit]
        raw = b_ + (((np.array([f[FEATS[i]] for i in hit]) - m_) / s_) @ ws)
    return float(state["c"] * np.exp(state["cal"][0] + state["cal"][1] * raw))

