"""Bidirectional LSTM + attention over a customer's day sequence.

Input per day : SSL embedding + calendar + physics cues + sun index (DAY_FEATS)
Outputs       : per-day logit "PV active on this day" (used for the change-point date)
                customer logit from attention pooling (PV presence)
Training      : random window crops per epoch; a crop that ends before the commissioning
                date of a PV customer is a negative example (label 0) -> more variety.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    torch.set_num_threads(8)
except RuntimeError:          # already set by an earlier import; harmless
    pass

from config import SEED
from data import date_to_day
from features import DAY_FEATS
from ssl_model import get_device


class LSTMAttn(nn.Module):
    def __init__(self, n_in, hidden=32, dropout=0.25):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.day_head = nn.Linear(2 * hidden, 1)
        self.att = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.cust_head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        """x (B,T,F), all sequences of equal length -> day logits (B,T), customer logits (B), attention (B,T)"""
        h, _ = self.lstm(self.inp(x))
        h = self.drop(h)
        day_logit = self.day_head(h).squeeze(-1)
        w = torch.softmax(self.att(h).squeeze(-1), dim=1)
        pooled = (w.unsqueeze(-1) * h).sum(1)
        return day_logit, self.cust_head(pooled).squeeze(-1), w


class SeqPVModel:
    """sklearn-like wrapper: fit(prep, labels) / predict(prep) -> (cust_prob dict, day_prob dict)."""

    def __init__(self, hidden=32, epochs=150, lr=2e-3, weight_decay=1e-3, min_crop=120,
                 day_loss_weight=1.0, seed=SEED, verbose=False):
        self.hidden, self.epochs, self.lr, self.wd = hidden, epochs, lr, weight_decay
        self.min_crop, self.day_w, self.seed, self.verbose = min_crop, day_loss_weight, seed, verbose
        self.mu = self.sd = None; self.net = None

    def _feats(self, tab):
        return tab[DAY_FEATS].to_numpy(dtype=np.float32)

    def fit(self, prep, labels):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        cids = [c for c in labels["customer_id"] if c in prep]
        lab = labels.set_index("customer_id")
        seqs, ydays, ycust = [], [], []
        for c in cids:
            f = self._feats(prep[c]["tab"]); days = prep[c]["tab"]["day"].to_numpy()
            r = lab.loc[c]
            if r["pv_installed"] and r["pv_installation_date"]:
                yd = (days >= date_to_day(r["pv_installation_date"])).astype(np.float32)
            else:
                yd = np.full(len(days), float(bool(r["pv_installed"])), np.float32)
            seqs.append(f); ydays.append(yd); ycust.append(float(bool(r["pv_installed"])))
        allf = np.concatenate(seqs); self.mu = allf.mean(0); self.sd = allf.std(0) + 1e-6
        seqs = [(s - self.mu) / self.sd for s in seqs]
        self.net = LSTMAttn(len(DAY_FEATS), self.hidden)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Tmin = min(len(q) for q in seqs)
        for ep in range(self.epochs):
            self.net.train()
            # one crop length per epoch (dense batch, no padding), random start per customer
            L = int(rng.integers(min(self.min_crop, Tmin), Tmin + 1))
            xs, yds, ycs = [], [], []
            for q, yd, yc in zip(seqs, ydays, ycust):
                a = int(rng.integers(0, len(q) - L + 1))
                xs.append(q[a:a + L]); yds.append(yd[a:a + L]); ycs.append(float(yd[a:a + L].max() > 0) if yc > 0 else 0.0)
            x = torch.from_numpy(np.stack(xs)); yd_t = torch.from_numpy(np.stack(yds))
            day_logit, cust_logit, _ = self.net(x)
            l_day = F.binary_cross_entropy_with_logits(day_logit, yd_t)
            l_cust = F.binary_cross_entropy_with_logits(cust_logit, torch.tensor(ycs))
            loss = l_cust + self.day_w * l_day
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), 1.0); opt.step(); sched.step()
            if self.verbose and (ep + 1) % 25 == 0:
                print(f"    epoch {ep + 1}: cust {l_cust.item():.3f} day {l_day.item():.3f}")
        return self

    @torch.no_grad()
    def predict(self, prep, cids=None):
        self.net.eval(); cids = list(prep) if cids is None else cids
        cust, day = {}, {}
        for c in cids:                                   # one at a time: full sequences
            q = (self._feats(prep[c]["tab"]) - self.mu) / self.sd
            dl, cl, _ = self.net(torch.from_numpy(q[None]))
            cust[c] = float(torch.sigmoid(cl)[0]); day[c] = torch.sigmoid(dl)[0].numpy().astype(np.float32)
        return cust, day


# The 2-D CNN lives in cnn_model.py; re-exported here so that models pickled before the move still load.
from cnn_model import HeatCNN, CNNPVModel   # noqa: E402,F401
