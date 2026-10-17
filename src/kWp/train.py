"""Train the kWp model (consumption-only capacity estimation).

There are only 22 labelled PV sites, so the network learns on synthetic sites first:
PV-free AIL consumption minus real UK PV generation (uk_pv, see build_pv_library.py) for a random
capacity, clipped at zero because the meter only records withdrawals. It is then fine-tuned on
the labelled sites (mixed with synthetic ones) and given a global MAPE-optimal scale factor.

Steps:
  python build_pv_library.py                                       # once: UK PV per kWp, 15 min
  python train.py prepare --ts energy_consumption_dev.csv --ts energy_consumption_unlabelled_dev.csv
  python train.py pretrain --out pretrained.pt                      # synthetic sites only
  python train.py cv --init pretrained.pt                           # 5-fold check on the labelled sites
  python train.py final --init pretrained.pt                        # fine-tune on all -> inference/kwp.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference'))
from inference import (PRE, SLOTS, KwpNet, date_to_epoch_day, day_of_year, encode,  # noqa: E402
                       load_consumption, make_window, predict_log_kwp, read_csv)

DATA = '/workspace/data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Synthetic sites
POST = 336                     # days after commissioning the network sees
KWP_PER_KW = (0.3, 25)         # capacity / mean load (kW), sampled log-uniformly
UK_TO_CH = (0.6, 1.4)          # yield multiplier on UK generation (absorbs climate, orientation, sizing)
DATE_JITTER = 10               # days between the date the model is told and the real PV start

# Fine-tuning on labelled sites
FT_STEPS, FT_LR, FT_SYN, FT_REAL = 400, 5e-5, 48, 16


# ----------------------------------------------------------------------------- data

def save_cons(cons, path):
    np.savez_compressed(path, ids=np.array(list(cons)), day0=np.array([v[0] for v in cons.values()]),
                        lens=np.array([len(v[1]) for v in cons.values()]),
                        X=np.concatenate([v[1] for v in cons.values()]))


def load_cons(path):
    z = np.load(path)
    X, off = z['X'], np.r_[0, np.cumsum(z['lens'])]      # read the big array once
    return {c: (int(d0), X[off[i]:off[i + 1]]) for i, (c, d0) in enumerate(zip(z['ids'], z['day0']))}


def read_labels(path):
    lab = read_csv(path)
    lab['pv_installed'] = lab.pv_installed.astype(str).str.upper().isin(['TRUE', '1'])
    if 'pv_capacity_kwp' in lab:
        lab['pv_capacity_kwp'] = pd.to_numeric(lab.pv_capacity_kwp, errors='coerce')
    return lab


def labelled_sites(cons, labels):
    """Labelled PV customers as (first_day, X, t0, kWp)."""
    pv = labels[labels.pv_installed & labels.pv_capacity_kwp.notna()]
    return [(*cons[r.customer_id], date_to_epoch_day(r.pv_installation_date), r.pv_capacity_kwp)
            for r in pv.itertuples() if r.customer_id in cons]


def pv_free_segments(cons, labels, cnn):
    """(customer, first_day, end_day) periods without PV, from customers the CNN detector labelled
    confidently: whole series of non-PV customers, pre-installation part of PV customers.
    Labelled customers are never used here."""
    cnn = cnn.set_index('customer_id')
    out = []
    for cid, (d0, X) in cons.items():
        if cid in set(labels.customer_id) or cid not in cnn.index:
            continue
        r = cnn.loc[cid]
        if str(r.confident).upper() != 'TRUE':
            continue
        end = d0 + len(X)
        if r.pv_installed:
            if pd.isna(r.pv_installation_date):
                continue
            end = min(end, date_to_epoch_day(r.pv_installation_date) - 14)
        if end - d0 >= 120:
            out.append((cid, d0, end))
    return out


# ----------------------------------------------------------------------------- synthetic sites

def daylength(doy, lat):
    decl = -23.44 * np.cos(2 * np.pi * (doy + 10) / 365.25)
    return 24 / np.pi * np.arccos(np.clip(-np.tan(np.radians(lat)) * np.tan(np.radians(decl)), -1, 1))


class PVLibrary:
    """UK generation per kWp [system, day, slot] (from build_pv_library.py), mapped to Swiss days."""

    def __init__(self, path):
        z = np.load(path)
        self.gen = z['gen']
        self.day0 = date_to_epoch_day(z['days'][0])
        self.years = sorted({int(str(d)[:4]) for d in z['days']})
        self.good = np.isfinite(self.gen).all(axis=2)
        self.median = np.nan_to_num(np.nanmedian(self.gen.astype(np.float32), axis=0))

    def curve(self, rng, days, system, year, shift):
        """One system's generation on the same calendar days of `year`, shifted `shift` slots earlier
        (UK -> Swiss solar noon) and stretched to Swiss day length (46.5 N instead of 52.5 N)."""
        doy = day_of_year(days)
        li = np.clip(date_to_epoch_day(f'{year}-01-01') - self.day0 + doy, 0, self.gen.shape[1] - 1)
        G = self.gen[system, li].astype(np.float32)
        bad = ~self.good[system, li]
        G[bad] = self.median[li[bad]]
        stretch = daylength(doy, 46.5) / daylength(doy, 52.5)
        noon = 45.6                                    # ~11:24 UTC, in slots
        slots = np.arange(SLOTS) + 0.5
        return np.stack([np.interp(noon + shift + (slots - noon) / w - 0.5, np.arange(SLOTS), g, left=0, right=0)
                         for g, w in zip(G, stretch)])

    def sample(self, rng, days):
        system, year = rng.integers(len(self.gen)), self.years[rng.integers(len(self.years))]
        shift = rng.uniform(2.0, 3.4)
        G = self.curve(rng, days, system, year, shift)
        if rng.random() < 0.35:                         # east/west split array: broader, flatter curve
            d = rng.uniform(2.0, 6.0)
            G = 0.5 * (self.curve(rng, days, system, year, shift + d) +
                       self.curve(rng, days, system, year, shift - d)) * rng.uniform(0.8, 0.95)
        if rng.random() < 0.6:                          # inverter clipping, DC/AC ratio 1.0-1.5
            G = np.minimum(G, 0.25 / rng.uniform(1.0, 1.5))
        return G


class SyntheticSites(torch.utils.data.IterableDataset):
    """Endless stream of (input, log kWp): a PV-free segment with synthetic PV added at a random date."""

    def __init__(self, cons, segments, library_path, seed=0):
        self.cons, self.segments, self.library_path, self.seed = cons, segments, library_path, seed
        self.mean_kw = [4 * np.nanmean(cons[c][1][a - cons[c][0]:b - cons[c][0]]) for c, a, b in segments]

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, worker.id if worker else 0, time.time_ns()])
        lib = PVLibrary(self.library_path)
        while True:
            k = rng.integers(len(self.segments))
            cid, a, b = self.segments[k]
            d0, X = self.cons[cid]
            t0 = int(rng.integers(a - PRE // 2, max(a - PRE // 2 + 1, b - 45)))
            W, days = make_window(d0, X, t0, POST)
            W[(days < a) | (days >= b)] = np.nan
            after = days >= t0

            kwp = float(np.clip(self.mean_kw[k] * np.exp(rng.uniform(*np.log(KWP_PER_KW))), 2, 5000))
            pv = lib.sample(rng, days) * kwp * rng.uniform(*UK_TO_CH)
            if rng.random() < 0.5:                      # other load changes often coincide with PV
                W[after] *= np.exp(rng.normal(0, 0.15))
            on = days >= t0 + int(rng.integers(-DATE_JITTER, DATE_JITTER + 1))
            W[on] = np.maximum(W[on] - pv[on], 0.0)     # the meter only sees withdrawals

            if rng.random() < 0.25:                     # data starting after installation
                W[~after] = np.nan
            if rng.random() < 0.4:                      # data ending soon after installation
                W[days >= t0 + rng.integers(30, POST)] = np.nan
            if (np.isfinite(W[after]).mean(axis=1) > 0.8).sum() < 20:   # too little usable data
                continue
            yield torch.from_numpy(encode(W, days, t0)), torch.tensor(np.log(kwp), dtype=torch.float32)


def synthetic_loader(cons, labels, cnn_path, library_path, batch, workers, seed=0):
    segments = pv_free_segments(cons, labels, read_labels(cnn_path))
    ds = SyntheticSites(cons, segments, library_path, seed)
    return iter(torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, prefetch_factor=4))


# ----------------------------------------------------------------------------- training helpers

def train_step(model, opt, x, y, weights=None):
    model.train()
    with torch.autocast(DEVICE, dtype=torch.bfloat16):
        pred = model(x.to(DEVICE)).float()
    loss = nn.functional.smooth_l1_loss(pred, y.to(DEVICE), beta=0.1, reduction='none')
    loss = loss.mean() if weights is None else (loss * weights.to(DEVICE)).sum() / weights.sum()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def augmented_real(rng, site):
    """A labelled site with a jittered date and, sometimes, missing context (as in synthesis)."""
    d0, X, t0, _ = site
    t = t0 + int(rng.integers(-DATE_JITTER, DATE_JITTER + 1))
    W, days = make_window(d0, X, t, POST)
    if rng.random() < 0.25:
        W[days < t] = np.nan
    if rng.random() < 0.4:
        W[days >= t + rng.integers(60, POST)] = np.nan
    return encode(W, days, t)


def mape_scale(pred, true):
    """Factor c minimising mean |c * pred - true| / true (a weighted median of true / pred)."""
    r, w = true / pred, pred / true
    o = np.argsort(r)
    return float(r[o][np.searchsorted(np.cumsum(w[o]) / w.sum(), 0.5)])


def scores(pred, true):
    ape = np.abs(pred - true) / true
    return dict(MAPE=ape.mean(), medAPE=np.median(ape), P=np.maximum(0, 1 - ape / 0.5).mean())


def load_pretrained(path):
    ck = torch.load(path, map_location=DEVICE)
    n_days = ck['model']['pos'].shape[1]
    global POST
    POST = n_days - PRE
    model = KwpNet(n_days).to(DEVICE)
    model.load_state_dict(ck['model'])
    return model, ck


def fine_tune(pretrained, sites, syn, seed):
    """Copy of the pretrained model trained on `sites` (augmented) mixed 16:48 with synthetic sites."""
    model = KwpNet(PRE + POST).to(DEVICE)
    model.load_state_dict(pretrained.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    for _ in range(FT_STEPS):
        xs, ys = next(syn)
        pick = rng.integers(len(sites), size=FT_REAL)
        xr = torch.from_numpy(np.stack([augmented_real(rng, sites[i]) for i in pick]))
        yr = torch.tensor([np.log(sites[i][3]) for i in pick], dtype=torch.float32)
        train_step(model, opt, torch.cat([xs, xr]), torch.cat([ys, yr]))
    return model


# ----------------------------------------------------------------------------- commands

def cmd_prepare(a):
    cons = {}
    for path in a.ts:
        cons.update(load_consumption(path))
        print(f'{path}: {len(cons)} customers so far', flush=True)
    save_cons(cons, a.cons)


def cmd_pretrain(a):
    cons, labels = load_cons(a.cons), read_labels(a.labels)
    syn = synthetic_loader(cons, labels, a.cnn_labels, a.library, 64, a.workers)
    real = labelled_sites(cons, labels)
    true = np.array([s[3] for s in real])
    model = KwpNet(PRE + POST).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.05)
    running = None
    for step in range(1, a.steps + 1):
        loss = train_step(model, opt, *next(syn))
        sched.step()
        running = loss if running is None else 0.98 * running + 0.02 * loss
        if step % 500 == 0 or step == a.steps:
            pred = np.exp(predict_log_kwp(model, [s[:3] for s in real], POST, DEVICE))
            s = scores(pred, true)
            print(f'step {step}: synthetic loss {running:.3f} | labelled sites (diagnostic, raw) '
                  f'MAPE {s["MAPE"]:.1%} P {s["P"]:.3f}', flush=True)
            torch.save({'model': model.state_dict(), 'post': POST}, a.out)


def cmd_cv(a):
    """K-fold estimate: fine-tune on k-1 folds, fit the scale there, predict the held-out fold."""
    pretrained, _ = load_pretrained(a.init)
    cons, labels = load_cons(a.cons), read_labels(a.labels)
    syn = synthetic_loader(cons, labels, a.cnn_labels, a.library, FT_SYN, a.workers, seed=1)
    real = labelled_sites(cons, labels)
    true = np.array([s[3] for s in real])
    eval_sites = [s[:3] for s in real]
    if a.eval_dates:                                   # score held-out sites with estimated dates
        est = read_labels(a.eval_dates).set_index('customer_id').pv_installation_date.dropna()
        ids = labels[labels.pv_installed & labels.pv_capacity_kwp.notna()].customer_id
        ids = [c for c in ids if c in cons]
        eval_sites = [(d0, X, date_to_epoch_day(est[c]) if c in est.index else t0)
                      for c, (d0, X, t0, _) in zip(ids, real)]
    oof = np.zeros(len(real))
    for rep in range(a.repeats):
        folds = np.random.default_rng(rep).permutation(len(real)) % a.folds
        for f in range(a.folds):
            train, test = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
            model = fine_tune(pretrained, [real[i] for i in train], syn, seed=100 * rep + f)
            scale = mape_scale(np.exp(predict_log_kwp(model, [real[i][:3] for i in train], POST, DEVICE)), true[train])
            pred = np.exp(predict_log_kwp(model, [eval_sites[i] for i in test], POST, DEVICE)) * scale
            oof[test] += pred / a.repeats
            print(f'repeat {rep} fold {f}: held-out MAPE {scores(pred, true[test])["MAPE"]:.1%}', flush=True)
    s = scores(oof, true)
    print(f'out-of-fold: MAPE {s["MAPE"]:.1%}  median APE {s["medAPE"]:.1%}  P {s["P"]:.3f}')


def cmd_final(a):
    """Fine-tune on all labelled sites and write the inference weights."""
    pretrained, _ = load_pretrained(a.init)
    cons, labels = load_cons(a.cons), read_labels(a.labels)
    syn = synthetic_loader(cons, labels, a.cnn_labels, a.library, FT_SYN, a.workers, seed=2)
    real = labelled_sites(cons, labels)
    true = np.array([s[3] for s in real])
    model = fine_tune(pretrained, real, syn, seed=0)
    scale = mape_scale(np.exp(predict_log_kwp(model, [s[:3] for s in real], POST, DEVICE)), true)
    torch.save({'model': model.state_dict(), 'scale': scale, 'post': POST}, a.out)
    print(f'saved {a.out} (scale {scale:.3f}, window {PRE}+{POST} days)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    cmds = {'prepare': cmd_prepare, 'pretrain': cmd_pretrain, 'cv': cmd_cv, 'final': cmd_final}
    for name in cmds:
        p = sub.add_parser(name)
        p.add_argument('--cons', default=f'{DATA}/cons.npz')
        p.add_argument('--labels', default=f'{DATA}/ail/pv_labels_dev.csv')
        p.add_argument('--cnn_labels', default=f'{DATA}/cnn_inferenced_label.csv',
                       help='presence/date labels for the unlabelled customers (picks PV-free periods)')
        p.add_argument('--library', default=f'{DATA}/pv_library_15min.npz')
        p.add_argument('--workers', type=int, default=32)
    sub.choices['prepare'].add_argument('--ts', action='append', required=True)
    sub.choices['pretrain'].add_argument('--steps', type=int, default=12000)
    sub.choices['pretrain'].add_argument('--out', default='pretrained.pt')
    for name in ('cv', 'final'):
        sub.choices[name].add_argument('--init', required=True, help='checkpoint from pretrain')
    sub.choices['cv'].add_argument('--folds', type=int, default=5)
    sub.choices['cv'].add_argument('--repeats', type=int, default=2)
    sub.choices['cv'].add_argument('--eval_dates', help='CSV of estimated install dates for held-out sites')
    sub.choices['final'].add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                    'inference', 'kwp.pt'))
    a = ap.parse_args()
    cmds[a.cmd](a)


if __name__ == '__main__':
    main()
