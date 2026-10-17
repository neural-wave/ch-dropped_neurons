"""Loading and validation of the 15-minute consumption CSVs.

Turns a long CSV (customer_id; energy_consumption_kwh; timestamp_local; timestamp_epoch)
into one 96-slot daily profile per (customer, day), handling:
  * the organisers' notice: keep only customer_id starting with "POD", drop null ids
  * DST days: 92-interval days get the missing hour interpolated, 100-interval days
    get the duplicated hour averaged (we work in local time via timestamp_epoch)
  * partial days at the start/end of a series (dropped if < MIN_VALID_SLOTS measured)
"""
import json
import numpy as np
import pandas as pd

from config import SLOTS, TZ, BASE_DAY, MIN_VALID_SLOTS

BASE = np.datetime64(BASE_DAY)


def sniff_sep(path):
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
    return ";" if line.count(";") >= line.count(",") else ","


def day_to_date(day):
    """int day index -> numpy datetime64[D] (vectorised)."""
    return BASE + np.asarray(day).astype("timedelta64[D]")


def date_to_day(date_str):
    return int((np.datetime64(str(date_str)[:10]) - BASE) // np.timedelta64(1, "D"))


def day_calendar(days):
    """month (1-12) and day-of-year for int day indices."""
    idx = pd.DatetimeIndex(day_to_date(days))
    return idx.month.values.astype(np.int8), idx.dayofyear.values.astype(np.int16)


def _process_chunk(chunk, stats, require_pod=True):
    stats["rows_total"] += len(chunk)
    cid = chunk["customer_id"]
    is_null = cid.isna()
    keep = cid.str.startswith("POD", na=False) if require_pod else ~is_null
    stats["all_ids"].update(cid[keep].unique().tolist())
    stats["rows_null_id"] += int(is_null.sum())
    stats["rows_non_pod_id"] += int((~keep & ~is_null).sum())
    chunk = chunk.loc[keep]
    kwh = pd.to_numeric(chunk["energy_consumption_kwh"], errors="coerce")
    bad = kwh.isna() | chunk["timestamp_epoch"].isna()
    stats["rows_bad_values"] += int(bad.sum())
    chunk, kwh = chunk.loc[~bad], kwh.loc[~bad]
    if len(chunk) == 0:
        return None
    # timestamps mark the END of each 15-minute interval (first row of a day is 00:15),
    # so shift back 15 minutes to get the interval start, then convert to local time.
    epoch = chunk["timestamp_epoch"].to_numpy(dtype="int64")
    t = pd.to_datetime(epoch - 900, unit="s", utc=True).tz_convert(TZ).tz_localize(None)
    day = ((t.values.astype("datetime64[D]") - BASE) // np.timedelta64(1, "D")).astype(np.int32)
    slot = (t.hour.values * 4 + t.minute.values // 15).astype(np.int16)
    df = pd.DataFrame({
        "cid": chunk["customer_id"].astype(str).to_numpy(),
        "day": day, "slot": slot, "kwh": kwh.to_numpy(dtype="float64"),
    })
    g = df.groupby(["cid", "day", "slot"], sort=True)["kwh"].agg(["sum", "count"])
    s = g["sum"].unstack("slot").reindex(columns=range(SLOTS)).astype("float32")
    c = g["count"].unstack("slot").reindex(columns=range(SLOTS)).fillna(0).astype("int16")
    return s, c


def load_timeseries(path, chunksize=8_000_000, verbose=True, require_pod=True):
    """Read a consumption CSV and return daily profiles.

    require_pod: apply the organisers' "customer_id LIKE 'POD%'" filter (dev/unlabelled data).
    Set False for the hidden test set so that no test customer is dropped.

    Returns dict with:
      customers : np.array[str]            unique customer ids (sorted)
      cust_idx  : int32[n_days]            index into customers
      day       : int32[n_days]            day index (days since BASE_DAY)
      X         : float32[n_days, 96]      kWh per 15-min slot (local time)
      stats     : dict                     validation counters
    """
    sep = sniff_sep(path)
    stats = {"rows_total": 0, "rows_null_id": 0, "rows_non_pod_id": 0,
             "rows_bad_values": 0, "separator": sep, "all_ids": set()}
    sums, counts = [], []
    reader = pd.read_csv(
        path, sep=sep, chunksize=chunksize,
        usecols=["customer_id", "energy_consumption_kwh", "timestamp_epoch"],
        dtype={"customer_id": "string", "energy_consumption_kwh": "float64",
               "timestamp_epoch": "float64"},
    )
    for i, chunk in enumerate(reader):
        out = _process_chunk(chunk, stats, require_pod)
        if out is not None:
            sums.append(out[0]); counts.append(out[1])
        if verbose:
            print(f"  chunk {i + 1}: {stats['rows_total']:,} rows read", flush=True)
    all_ids = sorted(stats.pop("all_ids")); stats["ids_in_file"] = len(all_ids)
    S = pd.concat(sums); C = pd.concat(counts)
    # (customer, day) pairs split across chunk boundaries -> merge their partial sums
    dup = S.index.duplicated(keep=False)
    if dup.any():
        S = pd.concat([S[~dup], S[dup].groupby(level=[0, 1]).sum(min_count=1)])
        C = pd.concat([C[~dup], C[dup].groupby(level=[0, 1]).sum()])
    S = S.sort_index(); C = C.loc[S.index]

    raw_per_day = C.sum(axis=1).to_numpy()
    stats["days_total"] = int(len(S))
    stats["days_96_intervals"] = int((raw_per_day == 96).sum())
    stats["days_92_intervals_dst"] = int((raw_per_day == 92).sum())
    stats["days_100_intervals_dst"] = int((raw_per_day == 100).sum())
    valid = (C.to_numpy() > 0).sum(axis=1)
    keep = valid >= MIN_VALID_SLOTS
    stats["days_dropped_partial"] = int((~keep).sum())

    Cn = C.to_numpy().astype("float32")
    M = S.to_numpy() / np.where(Cn > 0, Cn, np.nan)       # mean kWh per slot (NaN if unmeasured)
    M = M[keep]
    M = pd.DataFrame(M).interpolate(axis=1, limit_direction="both").to_numpy(dtype="float32")
    idx = S.index[keep]
    cids = idx.get_level_values(0).to_numpy().astype(str)
    days = idx.get_level_values(1).to_numpy().astype(np.int32)
    customers = np.unique(cids)
    cust_idx = pd.Index(customers).get_indexer(cids).astype(np.int32)
    stats["customers"] = int(len(customers))
    stats["days_kept"] = int(len(days))
    stats["kwh_min"] = float(np.nanmin(M)); stats["kwh_max"] = float(np.nanmax(M))
    stats["date_min"] = str(day_to_date(days.min())); stats["date_max"] = str(day_to_date(days.max()))
    return {"customers": customers, "cust_idx": cust_idx, "day": days, "X": M, "stats": stats,
            "all_ids": np.array(all_ids, dtype=str)}


def save_cache(d, path):
    np.savez(path, customers=d["customers"], cust_idx=d["cust_idx"], day=d["day"], X=d["X"],
             stats=json.dumps(d["stats"]))


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    return {"customers": z["customers"].astype(str), "cust_idx": z["cust_idx"], "day": z["day"],
            "X": z["X"], "stats": json.loads(str(z["stats"]))}


def per_customer(d):
    """dict customer_id -> {'days': int32[n], 'X': float32[n,96]} (sorted by day)."""
    order = np.lexsort((d["day"], d["cust_idx"]))
    ci, dy, X = d["cust_idx"][order], d["day"][order], d["X"][order]
    bounds = np.flatnonzero(np.diff(ci)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(ci)]])
    return {str(d["customers"][ci[s]]): {"days": dy[s:e], "X": X[s:e]} for s, e in zip(starts, ends)}


def read_labels(path):
    df = pd.read_csv(path, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df["customer_id"] = df["customer_id"].str.strip()
    df["pv_installed"] = df["pv_installed"].str.strip().str.upper().eq("TRUE")
    df["pv_installation_date"] = df["pv_installation_date"].fillna("").str.strip()
    df["pv_capacity_kwp"] = pd.to_numeric(df["pv_capacity_kwp"], errors="coerce")
    return df
