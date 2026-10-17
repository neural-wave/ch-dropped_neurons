"""Extensions (GatedGuard): a second same-sign step scanned after the commissioning date.

Residual model per indicator: baseline + b1*[d >= t0]*s + b2*[d >= t2]*s with t0 from the classical dating. Candidates t2 lie in
[t0 + 240 d, end - 90 d]. The TOP_K customers with the strongest second-step gain are emitted (gate off, ranking only). kWp after =
initial kWp * clip(R, 1.1, 4), R = pre-minus-after solar energy over pre-minus-between energy in matched (month, weekday) cells.
Customers switched off by the bell gate are removed AFTER the ranking, as in the reference.
Inputs: {cid: DailyMatrix}, {cid: initial kWp} of the detected customers, set of gated-off ids. Output: DataFrame COLS.
Fitted on: nothing (parameters tuned on synthetic injections).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import dating_classic as dc
from .capacity import MINDAYS, SOLAR
from .load import DailyMatrix

COLS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
MIN_GAP, MARGIN, HARM, ENV_FLOOR, TOP_K = 240, 90, 3, 1.0, 2


def second_step(dm: DailyMatrix) -> tuple[pd.Timestamp, pd.Timestamp, float] | None:
    """-> (t0, t2, gain2) of the strongest second step, or None when there is not enough data."""
    t0, _ = dc.date_one(dm)
    days, Y = dc.indicators(dm)
    ok = np.isfinite(Y).all(1)
    d = days[ok]
    n = len(d)
    if n < 120:
        return None
    s = dc.envelope(d, ENV_FLOOR)
    i0 = int(np.searchsorted(d, t0))
    idx = np.arange(n)
    B = np.column_stack([dc.basis(d, HARM), s * (idx >= i0)])
    Q, _ = np.linalg.qr(B)
    cand = np.flatnonzero((idx >= i0 + MIN_GAP) & (idx < n - MARGIN))
    if len(cand) == 0:
        return None
    G = (idx[:, None] >= cand[None, :]) * s[:, None]
    Gr = G - Q @ (Q.T @ G)
    gg = (Gr ** 2).sum(0) + 1e-12
    tot = np.zeros(len(cand))
    for k in range(Y.shape[1]):
        yk = Y[ok, k]
        yr = yk - Q @ (Q.T @ yk)
        num = yr @ Gr
        tot += np.where(np.sign(num / gg) == -1.0, num ** 2 / gg, 0.0) / (np.var(yr) + 1e-12)
    j = int(np.argmax(tot))
    return t0, d[cand[j]], float(tot[j])


def capacity_after_extension(dm: DailyMatrix, t0, t_ext, kwp0: float) -> float:
    """kwp0 * clip(R, 1.1, 4); R = 1.5 when no matched cell or the initial deficit is not positive."""
    days = pd.DatetimeIndex(dm.days)
    ok = (dm.coverage[:, SOLAR].mean(axis=1) >= 0.9) & ~dm.dst_day
    t0, t_ext = pd.Timestamp(t0), pd.Timestamp(t_ext)
    per = {"pre": days < t0, "mid": (days >= t0) & (days < t_ext), "post": days >= t_ext}
    wk = days.dayofweek >= 5
    d0 = d1 = 0.0
    for m in range(1, 13):
        for w in (False, True):
            sel = ok & (days.month == m) & (wk == w)
            sets = {k: sel & v for k, v in per.items()}
            if min(v.sum() for v in sets.values()) < MINDAYS:
                continue
            e = {k: np.nanmean(dm.x[v][:, SOLAR], axis=0).sum() for k, v in sets.items()}
            d0 += e["pre"] - e["mid"]
            d1 += e["pre"] - e["post"]
    return float(kwp0 * np.clip(d1 / d0 if d0 > 0 else 1.5, 1.1, 4.0))


def gated_extensions(dms: dict[str, DailyMatrix], kwp0: dict[str, float], off: set[str]) -> pd.DataFrame:
    """Top-K second steps among the customers of `kwp0` (iterated in dms order), then minus the gated-off ones."""
    cand = []
    for cid, dm in dms.items():
        if cid not in kwp0:
            continue
        try:
            st = second_step(dm)
        except Exception:
            continue  # degenerate customer: no extension
        if st:
            cand.append((st[2], cid, dm, st[:2]))
    cand.sort(key=lambda c: -c[0])
    rows = []
    for _, cid, dm, (t0, t2) in cand[:TOP_K]:
        try:
            rows.append((cid, t2.strftime("%Y-%m-%d"), capacity_after_extension(dm, t0, t2, kwp0[cid])))
        except Exception:
            continue
    e = pd.DataFrame(rows, columns=COLS)
    return e[~e["customer_id"].isin(off)]
