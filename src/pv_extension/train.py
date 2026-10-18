"""Train the PV-extension model (weekly 3-class sequence model), AIL data only.

Each customer is a sequence of weeks (mean daily curve + zero share, see inference/inference.py). A CNN + BiGRU gives,
for every week, p(nothing), p(PV installed), p(PV extended). The explicit "installed" class is what keeps a change
from no PV to PV from being reported as an extension.

Training sequences:
  * real extensions: manual labels (customer, extension_date, start_date) + the official dev extension
  * real negatives: dev customers (PV installs at their official dates, customers without PV)
  * synthetic extensions: a PV customer's own estimated PV output (matched pre-install profile minus withdrawal),
    scaled by alpha and subtracted again after a random date, clipped at zero ("same shape, stronger")
  * synthetic installs: a donor's estimated PV output, scaled to the recipient, subtracted from a customer without PV

Commands (run on the machine holding the data):
  python train.py loo --fold K      # leave-one-out fold K over the manual extension customers (run K = 0..10)
  python train.py summary           # aggregate the fold results, pick the threshold
  python train.py final             # train on everything -> inference/ext_net.pt
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference'))
from inference import EXT, NIGHT, NONE, START, WeekNet, eday, extension_events, week_of, weekly  # noqa: E402

DATA = '/workspace/data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DAY_SLOTS = slice(16, 80)          # 04:00-20:00 UTC: where PV can act
STEPS, BATCH, LR = 1500, 64, 1e-3
CLASS_WEIGHT = (1.0, 20.0, 40.0)   # events are ~1 week in 200


# ----------------------------------------------------------------------------- data

def load_cons(path):
    """Prepared consumption cache (see src/kWp/train.py prepare): {customer: (first UTC day, kWh[day, slot])}."""
    z = np.load(path)
    X, off = z['X'], np.r_[0, np.cumsum(z['lens'])]
    return {c: (int(d0), X[off[i]:off[i + 1]]) for i, (c, d0) in enumerate(zip(z['ids'], z['day0']))}


class Customer:
    def __init__(self, cid, d0, X, starts=(), exts=()):
        self.cid, self.d0, self.X, self.starts, self.exts = cid, d0, X, list(starts), list(exts)


def load_customers(a):
    """-> (extension customers, official extension customers, negative dev customers)."""
    cons = load_cons(a.cons)
    lab = read(a.labels)
    lab['pv'] = lab.pv_installed.str.upper() == 'TRUE'
    man = read(a.manual)
    # the official file has a ','-separated header but ';'-separated rows
    off = pd.read_csv(a.official_ext, sep=';', dtype=str, skiprows=1,
                      names=['customer_id', 'extension_date', 'pv_capacity_after_extension_kwp'])
    pos = []
    for r in man.itertuples():
        d0, X = cons[r.customer_id]
        pos.append(Customer(r.customer_id, d0, X, [eday(r.start_date)] if eday(r.start_date) > d0 else [],
                            [eday(r.extension_date)]))
    official = []
    for cid, g in off.groupby('customer_id'):
        official.append(Customer(cid, *cons[cid], [eday(lab.set_index('customer_id').loc[cid].pv_installation_date)],
                                 [eday(d) for d in g.extension_date]))
    skip = set(man.customer_id) | set(off.customer_id)
    neg = []
    for r in lab[~lab.customer_id.isin(skip)].itertuples():
        d0, X = cons[r.customer_id]
        neg.append(Customer(r.customer_id, d0, X, [eday(r.pv_installation_date)] if r.pv and eday(r.pv_installation_date) > d0 else []))
    return pos, official, neg


def read(path):
    with open(path) as f:
        sep = ';' if f.readline().count(';') else ','
    return pd.read_csv(path, sep=sep, dtype=str)


# ----------------------------------------------------------------------------- synthetic data

def pv_estimate(d0, X, t1):
    """Estimated PV output (kWh/slot) on days >= t1: pre-install profile of the same month (+-1) and weekday type, scaled
    by the day's night level, minus the withdrawal. Slots without a solar shape are zeroed. None if too little history."""
    days = np.arange(d0, d0 + len(X))
    month = (np.datetime64('1970-01-01') + days.astype('timedelta64[D]')).astype('datetime64[M]').astype(int) % 12
    wkend = ((days + 3) % 7) >= 5
    valid = np.isfinite(X).mean(1) > 0.9
    pre = valid & (days < t1)
    if pre.sum() < 60:
        return None
    night = np.nanmean(X[:, NIGHT], 1)
    ref = {}
    for m in range(12):
        for we in (False, True):
            sel = pre & (wkend == we) & (np.abs((month - m + 6) % 12 - 6) <= 1)
            if sel.sum() >= 3:
                ref[(m, we)] = (np.nanmean(X[sel], 0), np.nanmean(night[sel]))
    P = np.zeros_like(X)
    for i in np.flatnonzero(valid & (days >= t1)):
        r = ref.get((month[i], wkend[i]))
        if r is not None:
            scale = np.clip(night[i] / max(r[1], 1e-6), 0.5, 2.0)
            P[i, DAY_SLOTS] = np.maximum(r[0][DAY_SLOTS] * scale - X[i, DAY_SLOTS], 0)
    P = np.nan_to_num(P)
    shape = P[days >= t1].mean(0)
    P[:, shape < 0.1 * shape.max()] = 0
    return P if shape.max() > 0 else None


def inject_extension(rng, d0, X, t1, P, n_ext):
    """1-2 synthetic extensions after t1: subtract alpha * P from a random date on (cumulative), clip at zero."""
    end, Y, exts, lo = d0 + len(X), X.copy(), [], t1 + 120
    for _ in range(n_ext):
        if end - 120 <= lo:
            break
        t2 = int(rng.integers(lo, end - 120))
        Y[t2 - d0:] = np.maximum(Y[t2 - d0:] - rng.uniform(0.25, 1.5) * P[t2 - d0:], 0)
        exts.append(t2)
        lo = t2 + 180
    return Y, exts


def inject_start(rng, d0, X, donor, ratio):
    """Synthetic install on a customer without PV: donor PV output (matched by day of year), scaled, from a random date."""
    dd0, P = donor
    active = P.sum(1) > 0
    if active.sum() < 120 or len(X) < 240:
        return X, []
    lut = {}
    for i in np.flatnonzero(active):
        lut.setdefault((dd0 + i) % 365, i)
    t1 = int(rng.integers(d0 + 90, d0 + len(X) - 120))
    Y = X.copy()
    scale = ratio * np.nanmean(X[:, NIGHT]) / max(P[active].max(1).mean(), 1e-6)
    for i in range(t1 - d0, len(X)):
        j = lut.get((d0 + i) % 365)
        if j is not None:
            Y[i] = np.maximum(Y[i] - scale * P[j], 0)
    return Y, [t1]


def make_target(w0, nw, starts, exts, sigma=1.0):
    """Per-week soft targets: Gaussian bumps (sigma in weeks) at install / extension weeks."""
    T = np.zeros((nw, 3), np.float32)
    idx = np.arange(nw)
    for cls, ev in ((START, starts), (EXT, exts)):
        for d in ev:
            T[:, cls] = np.maximum(T[:, cls], np.exp(-0.5 * ((idx - (week_of(d) - w0)) / sigma) ** 2))
    T[:, NONE] = np.clip(1 - T[:, START] - T[:, EXT], 0, 1)
    return T / T.sum(1, keepdims=True)


def sample(rng, bases, nonpv, donors, real_pos, real_neg):
    """One training sequence (weekly features, targets) from a random source, randomly cropped to 1-4 years."""
    u = rng.random()
    if u < 0.25 and real_pos:                       # real extension customer
        e = real_pos[rng.integers(len(real_pos))]
        X, d0, starts, exts = e.X, e.d0, e.starts, e.exts
    elif u < 0.65:                                  # synthetic extension(s) on a PV customer with a known install
        e, P = bases[rng.integers(len(bases))]
        X, exts = inject_extension(rng, e.d0, e.X, e.starts[0], P, 1 if rng.random() < 0.7 else 2)
        d0, starts = e.d0, e.starts
        if rng.random() < 0.3:                      # PV already present when the data starts
            cut = e.starts[0] - e.d0 + int(rng.integers(0, 60))
            if cut < len(X) - 200:
                X, d0, starts = X[cut:], e.d0 + cut, []
                exts = [t for t in exts if t >= d0]
    elif u < 0.8:                                   # synthetic install on a customer without PV (never EXT)
        e = nonpv[rng.integers(len(nonpv))]
        X, starts = inject_start(rng, e.d0, e.X, donors[rng.integers(len(donors))], rng.uniform(0.5, 6))
        d0, exts = e.d0, []
    else:                                           # real negatives: installs, no PV
        e = real_neg[rng.integers(len(real_neg))]
        X, d0, starts, exts = e.X, e.d0, e.starts, e.exts
    n = len(X)
    L = int(rng.integers(min(n, 365), n + 1))
    a = int(rng.integers(0, n - L + 1))
    X, d0 = X[a:a + L], d0 + a
    w0, F = weekly(d0, X)
    return F, make_target(w0, len(F), [t for t in starts if d0 <= t < d0 + L], [t for t in exts if d0 <= t < d0 + L])


def batchify(items):
    T = max(len(f) for f, _ in items)
    x = np.zeros((len(items), T, items[0][0].shape[1]), np.float32)
    y = np.zeros((len(items), T, 3), np.float32)
    m = np.zeros((len(items), T), np.float32)
    for i, (f, t) in enumerate(items):
        x[i, :len(f)], y[i, :len(f)], m[i, :len(f)] = f, t, 1
    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)


class Sequences(torch.utils.data.IterableDataset):
    """Endless training sequences, generated in parallel worker processes."""

    def __init__(self, seed, *pools):
        self.seed, self.pools = seed, pools

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, w.id if w else 0])
        while True:
            yield sample(rng, *self.pools)


# ----------------------------------------------------------------------------- training

def train(seed, pos, neg, workers, tag):
    """Model trained on real extensions `pos` and negatives `neg` (+ synthetic data built from `neg`)."""
    bases = [(e, P) for e in neg if e.starts for P in [pv_estimate(e.d0, e.X, e.starts[0])] if P is not None]
    nonpv = [e for e in neg if not e.starts]
    donors = [(e.d0, P) for e, P in bases]
    loader = iter(torch.utils.data.DataLoader(Sequences(seed, bases, nonpv, donors, pos, neg), batch_size=BATCH,
                                              num_workers=workers, collate_fn=batchify, prefetch_factor=4))
    torch.manual_seed(seed)
    model = WeekNet().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.1)
    cw = torch.tensor(CLASS_WEIGHT, device=DEVICE)
    t0, running = time.time(), None
    for step in range(STEPS):
        model.train()
        x, y, m = (t.to(DEVICE) for t in next(loader))
        loss = (-((y * model(x).log_softmax(-1)) * cw).sum(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running = loss.item() if running is None else 0.98 * running + 0.02 * loss.item()
        if (step + 1) % 250 == 0:
            print(f'[{tag}] step {step + 1}/{STEPS} loss {running:.3f} | {time.time() - t0:.0f}s', flush=True)
    return model.eval()


def top_event(model, c):
    """Strongest extension candidate of a customer (no threshold): (day, score) or None."""
    ev, _ = extension_events(model, c.d0, c.X, threshold=0.0, device=DEVICE)
    return max(ev, key=lambda e: e[1]) if ev else None


# ----------------------------------------------------------------------------- commands

def cmd_loo(a, pos, official, neg):
    """Hold out extension customer `fold` and ~1/11 of the negatives; score both."""
    folds = np.random.default_rng(0).permutation(np.arange(len(neg)) % len(pos))
    held, test_neg = pos[a.fold], [e for e, f in zip(neg, folds) if f == a.fold]
    model = train(a.fold, [p for p in pos if p is not held] + official,
                  [e for e, f in zip(neg, folds) if f != a.fold], a.workers, f'fold {a.fold}')
    ev = top_event(model, held)
    res = dict(fold=a.fold, err_days=abs(ev[0] - held.exts[0]) if ev else None, score=ev[1] if ev else 0.0,
               neg=[dict(install=bool(e.starts), score=(top_event(model, e) or (0, 0.0))[1]) for e in test_neg])
    json.dump(res, open(f'{a.out_dir}/loo_fold{a.fold}.json', 'w'))
    print(f'fold {a.fold}: held-out extension error {res["err_days"]} d, score {res["score"]:.2f} | '
          f'negatives max score {max([n["score"] for n in res["neg"]] or [0]):.2f}')


def cmd_summary(a, *_):
    res = [json.load(open(f)) for f in sorted(glob.glob(f'{a.out_dir}/loo_fold*.json'))]
    hit = np.array([r['err_days'] is not None and r['err_days'] <= 30 for r in res])
    pos_s = np.array([r['score'] for r in res])
    neg_s = np.array([n['score'] for r in res for n in r['neg']])
    errs = [r['err_days'] for r, h in zip(res, hit) if h]
    print(f'{len(res)} folds: top candidate within 30 d for {hit.sum()}, median date error {np.median(errs):.0f} d')
    for thr in (1.0, 1.5, 2.0, 2.5):
        tp = int(((pos_s >= thr) & hit).sum())
        fp = int((neg_s >= thr).sum() + ((pos_s >= thr) & ~hit).sum())
        print(f'  threshold {thr}: TP {tp}  FP {fp} (on {len(neg_s)} negatives)  FN {len(res) - tp}  '
              f'event F1 {2 * tp / max(2 * tp + fp + len(res) - tp, 1):.2f}')


def cmd_final(a, pos, official, neg):
    model = train(1234, pos + official, neg, a.workers, 'final')
    torch.save({'model': model.state_dict(), 'threshold': a.threshold}, a.out)
    print(f'saved {a.out} (threshold {a.threshold})')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['loo', 'summary', 'final'])
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--cons', default=f'{DATA}/cons.npz')
    ap.add_argument('--labels', default=f'{DATA}/ail/pv_labels_dev.csv')
    ap.add_argument('--official_ext', default=f'{DATA}/ail/pv_extensions_labels_dev.csv')
    ap.add_argument('--manual', default=os.path.join(here, '../../data/manual_labels/extension_with_startdates.csv'))
    ap.add_argument('--out_dir', default='/workspace/ext')
    ap.add_argument('--out', default=os.path.join(here, 'inference', 'ext_net.pt'))
    ap.add_argument('--threshold', type=float, default=2.5, help='score threshold stored in the weights (LOO-best)')
    ap.add_argument('--workers', type=int, default=40)
    a = ap.parse_args()
    data = load_customers(a) if a.cmd != 'summary' else ()
    {'loo': cmd_loo, 'summary': cmd_summary, 'final': cmd_final}[a.cmd](a, *data)


if __name__ == '__main__':
    main()
