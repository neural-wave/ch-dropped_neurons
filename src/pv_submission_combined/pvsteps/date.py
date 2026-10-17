"""Commissioning date from the day scores: best single increasing mean shift of the 7-day smoothed score.

If the 'before' mean already exceeds 0.5 the PV predates the series and the first observed day is returned (a boundary
proxy, not a verified commissioning date).
Inputs: day table with a `score` column. Output: a pandas Timestamp. Fitted on: nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .load import BASE


def smooth(s, w: int = 7) -> np.ndarray:
    return pd.Series(s).rolling(w, center=True, min_periods=1).mean().to_numpy()


def step_fit(s, min_seg: int = 14) -> tuple[int, float]:
    """Best single increasing mean shift -> (index of first 'after' value, mean before)."""
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    if n < 2 * min_seg + 1:
        return 0, float(s.mean()) if n else 0.0
    cs, cs2 = np.concatenate([[0.0], np.cumsum(s)]), np.concatenate([[0.0], np.cumsum(s * s)])
    t = np.arange(min_seg, n - min_seg + 1)
    n1, n2 = t, n - t
    sum1, sum2 = cs[t], cs[n] - cs[t]
    sse = (cs2[t] - sum1 ** 2 / n1) + (cs2[n] - cs2[t] - sum2 ** 2 / n2)
    m1, m2 = sum1 / n1, sum2 / n2
    sse = np.where(m2 > m1, sse, np.inf)
    if not np.isfinite(sse).any():
        return 0, float(s.mean())
    i = int(np.argmin(sse))
    return int(t[i]), float(m1[i])


def ssl_date(tab: pd.DataFrame) -> pd.Timestamp:
    """Commissioning date of one customer from its scored day table."""
    days = tab["day"].to_numpy()
    idx, m1 = step_fit(smooth(tab["score"].to_numpy()))
    day = int(days[0]) if m1 > 0.5 else int(days[idx])
    return pd.Timestamp(BASE + np.timedelta64(day, "D"))
