"""Detect PV capacity extensions (date + total capacity after) from 15-minute grid-withdrawal data only.

Usage:  python inference.py consumption.csv [predictions.csv] [--out pv_extensions_predictions.csv]
See HOWTO.md. This file also defines the features and the network used by ../train.py.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SLOTS = 96                         # 15-minute slots per UTC day
NIGHT = np.r_[0:12, 88:96]         # ~22:00-03:00 UTC, unaffected by PV
MIDDAY = slice(36, 56)             # ~09:00-14:00 UTC
EPOCH_DAY0 = np.datetime64('1970-01-01')
NONE, START, EXT = 0, 1, 2         # weekly classes: nothing / PV installed / PV extended
MAX_EVENTS, MIN_GAP_WEEKS = 2, 26  # per customer
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- reading data

def read_csv(path, **kw):
    with open(path, encoding='utf-8') as f:
        header = f.readline()
    return pd.read_csv(path, sep=';' if header.count(';') >= header.count(',') else ',',
                       dtype={'customer_id': str}, **kw)


def load_consumption(path):
    """CSV -> {customer_id: (first_day, X)}, X[day, slot] = kWh, NaN where not measured. Days and slots in UTC;
    timestamp_epoch marks the end of each interval, so it is shifted back 15 minutes."""
    df = read_csv(path, usecols=['customer_id', 'energy_consumption_kwh', 'timestamp_epoch'])
    df['energy_consumption_kwh'] = pd.to_numeric(df.energy_consumption_kwh, errors='coerce')
    df['timestamp_epoch'] = pd.to_numeric(df.timestamp_epoch, errors='coerce')
    df = df.dropna().sort_values(['customer_id', 'timestamp_epoch'])
    t = df.timestamp_epoch.to_numpy(np.int64) - 900
    day, slot = t // 86400, (t % 86400) // 900
    kwh = df.energy_consumption_kwh.to_numpy(np.float32)
    ids = df.customer_id.to_numpy()
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
    out = {}
    for a, b in zip(starts[:-1], starts[1:]):
        d0 = day[a:b].min()
        X = np.full((day[a:b].max() - d0 + 1, SLOTS), np.nan, np.float32)
        X[day[a:b] - d0, slot[a:b]] = np.where(kwh[a:b] >= 0, kwh[a:b], np.nan)
        out[ids[a]] = (int(d0), X)
    return out


def eday(date):
    return int((np.datetime64(str(date)[:10]) - EPOCH_DAY0).astype(int))


def week_of(day):
    return (day + 3) // 7              # Monday-aligned week number


# ----------------------------------------------------------------------------- features

def ref_night(X):
    """Customer load level (kW): median night load over well-measured days."""
    kw = X * 4
    ok = np.isfinite(kw[:, NIGHT]).mean(1) > 0.8
    v = np.nanmedian(np.nanmean(kw[ok][:, NIGHT], 1)) if ok.any() else np.nan
    if not (np.isfinite(v) and v > 0.05):
        v = np.nanmedian(kw)
    return float(v) if np.isfinite(v) and v > 0.05 else 0.05


def weekly(d0, X):
    """-> (first week number, [weeks, 195]): mean daily curve log1p(kW / night level) (96), share of zero readings per
    slot (96), share of days measured, season sin/cos. Weeks with fewer than 3 measured days stay zero."""
    kw = X * 4 / ref_night(X)
    w = week_of(np.arange(d0, d0 + len(X)))
    w0, nw = w[0], w[-1] - w[0] + 1
    F = np.zeros((nw, 2 * SLOTS + 3), np.float32)
    measured = np.isfinite(kw).mean(1) > 0.8
    for i in range(nw):
        sel = (w - w0 == i) & measured
        if sel.sum() >= 3:
            F[i, :SLOTS] = np.log1p(np.nanmean(kw[sel], 0))
            F[i, SLOTS:2 * SLOTS] = np.nanmean(kw[sel] <= 1e-6, 0)
            F[i, 2 * SLOTS] = sel.sum() / 7
    mid = ((np.arange(nw) + w0) * 7).astype('timedelta64[D]') + EPOCH_DAY0
    doy = (mid - mid.astype('datetime64[Y]')).astype(int)
    F[:, 2 * SLOTS + 1] = np.sin(doy / 365.25 * 2 * np.pi)
    F[:, 2 * SLOTS + 2] = np.cos(doy / 365.25 * 2 * np.pi)
    return int(w0), F


def daily_pv_index(X):
    """Per-day PV strength: (night load - midday load) / customer night level. Grows with PV size."""
    kw = X * 4
    return (np.nanmean(kw[:, NIGHT], 1) - np.nanmean(kw[:, MIDDAY], 1)) / ref_night(X)


# ----------------------------------------------------------------------------- network

class WeekNet(nn.Module):
    """Per-week CNN over the mean daily curve and zero share -> BiGRU across weeks -> 3 classes per week."""

    def __init__(self, d=64):
        super().__init__()
        self.week = nn.Sequential(
            nn.Conv1d(2, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2, stride=2), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 24, d))
        self.rows = nn.Linear(3, d)
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(2 * d, 3)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.week(x[..., :2 * SLOTS].reshape(B * T, 2, SLOTS)).reshape(B, T, -1) + self.rows(x[..., 2 * SLOTS:])
        return self.out(self.gru(h)[0])


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    model = WeekNet().to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, ck['threshold']


# ----------------------------------------------------------------------------- events

def refine_date(d0, X, guess, radius=14, side=21):
    """Day of the largest rise in PV strength (median of `side` days after minus before) within +-radius of guess."""
    s = daily_pv_index(X)
    best, best_i = -np.inf, guess - d0
    for i in range(max(guess - d0 - radius, side), min(guess - d0 + radius, len(s) - side)):
        a, b = s[i - side:i], s[i:i + side]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) >= 7 and len(b) >= 7 and np.median(b) - np.median(a) > best:
            best, best_i = np.median(b) - np.median(a), i
    return d0 + best_i


@torch.no_grad()
def extension_events(model, d0, X, threshold, device='cpu'):
    """-> list of (epoch day, score) and the weekly probabilities [weeks, 3].

    score = p(extension) summed over the week and its two neighbours (0..3). Peaks above `threshold`, at most
    MAX_EVENTS, MIN_GAP_WEEKS apart; a peak where p(install) is larger is an installation, not an extension."""
    w0, F = weekly(d0, X)
    p = model(torch.from_numpy(F)[None].to(device)).softmax(-1)[0].cpu().numpy()
    pe = np.convolve(p[:, EXT], np.ones(3), 'same')
    ps = np.convolve(p[:, START], np.ones(3), 'same')
    picked = []
    for i in np.argsort(-pe):
        if pe[i] < threshold or len(picked) >= MAX_EVENTS:
            break
        if ps[i] > pe[i] or any(abs(i - j) < MIN_GAP_WEEKS for j in picked):
            continue
        picked.append(i)
    return [(refine_date(d0, X, (w0 + i) * 7), float(pe[i])) for i in sorted(picked)], p


def capacity_ratio(d0, X, day, prev_day=None, window=90):
    """Growth of PV strength at `day`: median daily PV index in the `window` days after, over the same season one year
    earlier (or the `window` days before when there is no earlier year), clipped to 1.1-4; 1.5 when not identifiable."""
    s = daily_pv_index(X)
    i = day - d0
    after = s[i:i + window]
    before = s[i - 365:i - 365 + window] if i - 365 >= 0 and (prev_day is None or day - 365 >= prev_day) \
        else s[max(i - window, 0):i]
    a, b = np.nanmedian(before) if len(before) else np.nan, np.nanmedian(after) if len(after) else np.nan
    if not (np.isfinite(a) and np.isfinite(b) and a > 0.05):
        return 1.5
    return float(np.clip(b / a, 1.1, 4.0))


def predict_extensions(model, threshold, cons, kwp0=None, start=None, allow_at_install=False, device='cpu'):
    """-> DataFrame(customer_id, extension_date, pv_capacity_after_extension_kwp).

    cons: {customer: (first_day, X)}. kwp0: {customer: initial kWp} of the PV customers (None = every customer, no
    capacity). start: {customer: installation epoch day}; extensions within 60 days after it are dropped unless
    allow_at_install."""
    start = start or {}
    rows = []
    for cid, (d0, X) in cons.items():
        if kwp0 is not None and cid not in kwp0:
            continue                                   # not a PV customer according to the main pipeline
        events, _ = extension_events(model, d0, X, threshold, device)
        cap, prev = (kwp0 or {}).get(cid, np.nan), None
        for day, _ in events:
            if cid in start and day <= start[cid] + 60 and not allow_at_install:
                continue                               # the main pipeline calls this the initial installation
            cap = cap * capacity_ratio(d0, X, day, prev) if np.isfinite(cap) else np.nan
            rows.append((cid, str(EPOCH_DAY0 + np.timedelta64(int(day), 'D')),
                         f'{cap:.2f}' if np.isfinite(cap) and cap > 0 else ''))
            prev = day
    return pd.DataFrame(rows, columns=['customer_id', 'extension_date', 'pv_capacity_after_extension_kwp'])


# ----------------------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('consumption', help='15-minute consumption CSV (customer_id, energy_consumption_kwh, timestamp_epoch)')
    ap.add_argument('predictions', nargs='?',
                    help='optional main-task predictions.csv (customer_id, pv_installed, pv_installation_date, '
                         'pv_capacity_kwp): restricts to PV customers and gives the initial kWp')
    ap.add_argument('--weights', default=os.path.join(HERE, 'ext_net.pt'))
    ap.add_argument('--out', default='pv_extensions_predictions.csv')
    ap.add_argument('--allow-at-install', action='store_true',
                    help='keep extensions within 60 days of the initial installation date given in predictions.csv')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, threshold = load_model(a.weights, device)
    cons = load_consumption(a.consumption)
    kwp0, start = None, None
    if a.predictions:
        pr = read_csv(a.predictions)
        pr = pr[pr.pv_installed.astype(str).str.strip().str.upper().isin(['TRUE', '1'])]
        kwp0 = dict(zip(pr.customer_id, pd.to_numeric(pr.pv_capacity_kwp, errors='coerce')))
        start = {c: eday(d) for c, d in zip(pr.customer_id, pr.pv_installation_date) if isinstance(d, str) and d}
    out = predict_extensions(model, threshold, cons, kwp0, start, a.allow_at_install, device)
    out.to_csv(a.out, sep=';', index=False, encoding='utf-8')
    print(f'wrote {a.out}: {len(out)} extension(s) for {out.customer_id.nunique()} of {len(cons)} customers')


if __name__ == '__main__':
    main()
