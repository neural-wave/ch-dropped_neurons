"""Estimate the nominal PV capacity (kWp) of customers from 15-minute grid-withdrawal data only.

Usage:  python inference.py consumption.csv detections.csv [--out predictions.csv]
See HOWTO.md. This file also defines the features and the network used by ../train.py.
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SLOTS = 96                       # 15-minute slots per UTC day
PRE = 56                         # days of context before the commissioning date
TTA_SHIFTS = (-14, -7, 0, 7, 14)  # commissioning-date shifts averaged at prediction time
EPOCH_DAY0 = np.datetime64('1970-01-01')
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------------- reading data

def read_csv(path, **kw):
    with open(path, encoding='utf-8') as f:
        header = f.readline()
    return pd.read_csv(path, sep=';' if header.count(';') >= header.count(',') else ',',
                       dtype={'customer_id': str}, **kw)


def load_consumption(path):
    """CSV -> {customer_id: (first_day, X)} with X[day, slot] = kWh, NaN where not measured.

    Days and slots are in UTC, which sidesteps daylight-saving days. Timestamps mark the end of
    each 15-minute interval, so they are shifted back 15 minutes to the interval start.
    """
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


def date_to_epoch_day(date):
    return int((np.datetime64(str(date)[:10]) - EPOCH_DAY0).astype(int))


def day_of_year(epoch_days):
    dt = EPOCH_DAY0 + np.asarray(epoch_days).astype('timedelta64[D]')
    return (dt - dt.astype('datetime64[Y]')).astype(int)


# ----------------------------------------------------------------------------- features

def make_window(d0, X, t0, post):
    """Days [t0 - PRE, t0 + post) of one customer as kWh[day, slot] (NaN outside the data)."""
    days = np.arange(t0 - PRE, t0 + post)
    idx = days - d0
    ok = (idx >= 0) & (idx < len(X))
    W = np.full((len(days), SLOTS), np.nan, np.float32)
    W[ok] = X[idx[ok]]
    return W, days


def encode(W, days, t0):
    """One row per day: 96 x log1p(kW), then fraction measured, season sin/cos, after-install flag, valid flag.

    Days with less than 80% of slots measured are zeroed and flagged invalid (the network ignores them).
    """
    measured = np.isfinite(W)
    load = np.log1p(np.where(measured, W * 4.0, 0.0))
    valid = measured.mean(axis=1) > 0.8
    load[~valid] = 0.0
    season = day_of_year(days) / 365.25 * 2 * math.pi
    rows = np.stack([measured.mean(axis=1), np.sin(season), np.cos(season),
                     (days >= t0).astype(np.float32), valid.astype(np.float32)], axis=1)
    return np.concatenate([load, rows], axis=1).astype(np.float32)


def site_input(d0, X, t0, post):
    return encode(*make_window(d0, X, t0, post), t0)


# ----------------------------------------------------------------------------- network

class KwpNet(nn.Module):
    """Per-day CNN over the 96 slots -> transformer across days -> attention pooling -> log(kWp)."""

    def __init__(self, n_days, d=128, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.day = nn.Sequential(                       # shape of one day's load curve
            nn.Conv1d(1, 32, 5, padding=2), nn.GELU(),
            nn.Conv1d(32, 64, 5, padding=2, stride=2), nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2, stride=2), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 24, d))
        self.rows = nn.Linear(5, d)                     # season, before/after install, data quality
        self.pos = nn.Parameter(torch.zeros(1, n_days, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.attn = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):
        B, T, _ = x.shape
        valid = x[..., -1] > 0.5
        h = self.day(x[..., :SLOTS].reshape(B * T, 1, SLOTS)).reshape(B, T, -1)
        h = h + self.rows(x[..., SLOTS:]) + self.pos[:, :T]
        h = self.tf(h, src_key_padding_mask=~valid)
        w = self.attn(h).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        return self.head((w.unsqueeze(-1) * h).sum(1)).squeeze(-1) + 5.0   # log(kWp); e^5 ~ 150 kWp


@torch.no_grad()
def predict_log_kwp(model, sites, post, device, batch=32):
    """Mean log(kWp) over commissioning-date shifts. sites: list of (first_day, X, t0)."""
    model.eval()
    preds = []
    for s in TTA_SHIFTS:
        x = np.stack([site_input(d0, X, t0 + s, post) for d0, X, t0 in sites])
        preds.append(torch.cat([model(torch.from_numpy(x[i:i + batch]).to(device)).float().cpu()
                                for i in range(0, len(x), batch)]).numpy())
    return np.mean(preds, axis=0)


def load_model(path, device):
    ck = torch.load(path, map_location=device)
    model = KwpNet(PRE + ck['post']).to(device)
    model.load_state_dict(ck['model'])
    return model, ck['scale'], ck['post']


# ----------------------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('consumption', help='15-minute consumption CSV (customer_id, energy_consumption_kwh, timestamp_epoch)')
    ap.add_argument('detections', help='CSV with customer_id, pv_installed, pv_installation_date')
    ap.add_argument('--weights', default=os.path.join(HERE, 'kwp.pt'))
    ap.add_argument('--out', default='predictions.csv')
    a = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, scale, post = load_model(a.weights, device)
    cons = load_consumption(a.consumption)
    det = read_csv(a.detections)
    det['pv_installed'] = det.pv_installed.astype(str).str.strip().str.upper().isin(['TRUE', '1'])

    todo = det.pv_installed & det.pv_installation_date.notna() & det.customer_id.isin(cons.keys())
    sites = [(*cons[r.customer_id], date_to_epoch_day(r.pv_installation_date)) for r in det[todo].itertuples()]
    det['pv_capacity_kwp'] = np.nan
    if sites:
        det.loc[todo, 'pv_capacity_kwp'] = np.round(np.exp(predict_log_kwp(model, sites, post, device)) * scale, 1)

    det['pv_installed'] = np.where(det.pv_installed, 'TRUE', 'FALSE')
    det.loc[det.pv_installed == 'FALSE', 'pv_installation_date'] = ''
    out = det[['customer_id', 'pv_installed', 'pv_installation_date', 'pv_capacity_kwp']]
    out.to_csv(a.out, sep=';', index=False, encoding='utf-8')
    skipped = int((det.pv_installed == 'TRUE').sum() - todo.sum())
    print(f'wrote {a.out}: {len(out)} customers, {int(todo.sum())} with kWp'
          + (f', {skipped} PV customers skipped (no date or no consumption data)' if skipped else ''))


if __name__ == '__main__':
    main()
