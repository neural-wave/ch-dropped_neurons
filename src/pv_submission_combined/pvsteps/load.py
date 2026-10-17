"""Read the AIL consumption CSV and build the local-time 96-slot daily grids.

Input: the `;` CSV. timestamp_epoch is the END of each 15 min interval, so the interval start (Europe/Zurich wall clock)
decides day and slot. Outputs: `read_series` -> long frame; `daily_matrix` -> {cid: DailyMatrix} (days x 96, DST-safe,
NaN where unobserved); `ail_days` -> {cid: {"days", "X"}} (float32 days x 96, gaps interpolated) for the day encoder.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

TZ = "Europe/Zurich"
STEP, SLOTS = 900, 96
BASE = np.datetime64("2015-01-01")  # day index 0 of the encoder grid
MIN_VALID_SLOTS = 80  # encoder grid: drop days with fewer measured slots


class DailyMatrix(NamedTuple):
    days: pd.DatetimeIndex  # naive local dates, contiguous first..last observed day
    x: np.ndarray  # (days, 96) kWh per 15 min, NaN where not covered
    coverage: np.ndarray  # (days, 96) bool: observed (or DST-interpolated)
    dst_day: np.ndarray  # (days,) bool: 92- or 100-interval day


def customer_ids(path: str | Path) -> list[str]:
    """Distinct non-blank customer ids in file order."""
    t = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter=";"),
                       convert_options=pacsv.ConvertOptions(include_columns=["customer_id"], column_types={"customer_id": "string"}))
    return list(dict.fromkeys(c for c in t.column("customer_id").to_pylist() if c is not None and c.strip()))


def read_series(path: str | Path) -> pd.DataFrame:
    """CSV -> [customer_id, timestamp_epoch, kwh] sorted by customer then epoch. Duplicates dropped (first kept), negatives -> NaN."""
    types = {"customer_id": pa.string(), "energy_consumption_kwh": pa.float64(), "timestamp_local": pa.string(), "timestamp_epoch": pa.int64()}
    with open(path, encoding="utf-8-sig") as fh:
        header = fh.readline().strip().split(";")
    raw = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter=";"),
                         convert_options=pacsv.ConvertOptions(column_types={c: types.get(c, pa.string()) for c in header})).to_pandas()
    raw = raw.rename(columns={"energy_consumption_kwh": "kwh"})[["customer_id", "timestamp_epoch", "kwh"]]
    codes, ids = pd.factorize(raw["customer_id"], sort=True)
    ep, kwh = raw["timestamp_epoch"].to_numpy(np.int64), raw["kwh"].to_numpy(np.float64)
    o = np.lexsort((ep, codes.astype(np.int64)))
    codes, ep, kwh = codes.astype(np.int64)[o], ep[o], kwh[o]
    keep = np.ones(len(ep), bool)
    keep[1:] = (codes[1:] != codes[:-1]) | (ep[1:] != ep[:-1])
    codes, ep, kwh = codes[keep], ep[keep], kwh[keep]
    df = pd.DataFrame({"customer_id": ids.to_numpy(object)[codes].astype(str), "timestamp_epoch": ep, "kwh": np.where(kwh < 0, np.nan, kwh)})
    return df[df["customer_id"].str.strip() != ""].reset_index(drop=True)


def _wall(epoch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local wall clock of UTC epoch seconds -> (day number since 1970, minute of day)."""
    w = pd.to_datetime(np.asarray(epoch, np.int64), unit="s", utc=True).tz_convert(TZ).tz_localize(None).as_unit("s").asi8
    return w // 86400, (w % 86400) // 60


def _day_intervals(d0: int, d1: int) -> np.ndarray:
    """Number of 15 min intervals (96/92/100) of each local day d0..d1."""
    mid = pd.to_datetime(np.arange(d0, d1 + 2) * 86400, unit="s").tz_localize(TZ).tz_convert("UTC").as_unit("s").asi8
    return np.diff(mid) // STEP


def daily_matrix(df: pd.DataFrame) -> dict[str, DailyMatrix]:
    """Per customer days x 96 local slots. 92-interval day: missing hour interpolated; 100-interval day: repeated hour averaged."""
    codes, ids = pd.factorize(df["customer_id"], sort=True)
    ep, kwh = df["timestamp_epoch"].to_numpy(np.int64), df["kwh"].to_numpy(np.float64)
    day, minute = _wall(ep - STEP)
    slot = minute // 15
    out: dict[str, DailyMatrix] = {}
    o = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[o], np.arange(len(ids) + 1))
    for i, cid in enumerate(ids):
        sel = o[bounds[i]:bounds[i + 1]]
        ok = np.isfinite(kwh[sel])
        d, s, v = day[sel][ok], slot[sel][ok], kwh[sel][ok]
        if len(d) == 0:
            continue
        d0, d1 = int(day[sel].min()), int(day[sel].max())
        n = d1 - d0 + 1
        flat = (d - d0) * SLOTS + s
        tot, cnt = np.bincount(flat, weights=v, minlength=n * SLOTS), np.bincount(flat, minlength=n * SLOTS)
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.where(cnt > 0, tot / cnt, np.nan).reshape(n, SLOTS)
        length = _day_intervals(d0, d1)
        spring = length == 92
        if spring.any():
            a, b = x[spring, 7], x[spring, 12]
            x[spring, 8:12] = a[:, None] + (b - a)[:, None] * (np.arange(1, 5) / 5)
        out[str(cid)] = DailyMatrix(pd.to_datetime(np.arange(d0, d1 + 1) * 86400, unit="s"), x, np.isfinite(x), length != SLOTS)
    return out


def ail_days(df: pd.DataFrame) -> dict[str, dict]:
    """Encoder grid: {cid: {"days": int32 (n,), "X": float32 (n, 96)}}. Slots with no data are interpolated along the day."""
    ok = df["kwh"].notna()
    t = pd.to_datetime(df["timestamp_epoch"].to_numpy(np.int64)[ok] - STEP, unit="s", utc=True).tz_convert(TZ).tz_localize(None)
    g = pd.DataFrame({"cid": df["customer_id"].astype(str).to_numpy()[ok],
                      "day": ((t.values.astype("datetime64[D]") - BASE) // np.timedelta64(1, "D")).astype(np.int32),
                      "slot": (t.hour.values * 4 + t.minute.values // 15).astype(np.int16),
                      "kwh": df["kwh"].to_numpy(np.float64)[ok]}).groupby(["cid", "day", "slot"], sort=True)["kwh"].agg(["sum", "count"])
    S = g["sum"].unstack("slot").reindex(columns=range(SLOTS)).astype("float32")
    C = g["count"].unstack("slot").reindex(columns=range(SLOTS)).fillna(0).astype("int16").loc[S.index]
    Cn = C.to_numpy().astype("float32")
    M = S.to_numpy() / np.where(Cn > 0, Cn, np.nan)
    keep = (C.to_numpy() > 0).sum(axis=1) >= MIN_VALID_SLOTS
    M = pd.DataFrame(M[keep]).interpolate(axis=1, limit_direction="both").to_numpy(dtype="float32")
    cids = S.index[keep].get_level_values(0).to_numpy().astype(str)
    days = S.index[keep].get_level_values(1).to_numpy().astype(np.int32)
    return {c: {"days": days[cids == c], "X": M[cids == c]} for c in np.unique(cids)}
