"""Step 3 of the pipeline: the 2-D convolutional network and its training recipe.

Input (from images.build_image): float32[C + 3, L, 96], L calendar days x 96 fifteen-minute slots;
C image channel(s), day-of-year sine, day-of-year cosine, mask channel (zeros). With calendar=False the two
day-of-year channels are dropped (C + 1 channels): the network then sees only the consumption image.

Network (HeatCNN), 83,859 parameters for C = 1:
  conv stack (also used by pretrain.py):
    Conv2d(in_ch -> 16, kernel 5 days x 7 slots)  GELU  MaxPool(1 x 2)   slots 96 -> 48
    Conv2d(16 -> 32,    kernel 5 x 5)             GELU  MaxPool(1 x 2)   slots 48 -> 24
    Conv2d(32 -> 32,    kernel 5 x 5)             GELU  MaxPool(1 x 4)   slots 24 -> 6
    -> feature maps (32, L, 6): the day axis is never pooled, every day keeps its own features
  temporal head:
    Conv1d(32 * 6 = 192 -> 32, kernel 7 days)  GELU  Dropout(0.25)       -> (32, L)
    day head       Conv1d(32 -> 1, 1)                                    -> one logit per day "PV active"
    attention      Conv1d(32 -> 1, 1), softmax over days                 -> which days matter
    customer head  Linear(32 -> 1) on the attention-weighted mean        -> one logit "customer has PV"

Training (CNNPVModel.fit):
  labels    customer y = PV yes/no; day y_t = 1 on/after the commissioning date of a PV customer
  epochs    each epoch is ONE batch holding every training customer: a window length L is drawn in
            [min_crop, min(max_crop, shortest series)] and every customer contributes a random window
            of L consecutive days; the window's customer label is 1 only if it contains a PV day
  loss      BCE(customer logit) + day_loss_weight * BCE(day logits)
  optimiser AdamW(lr, weight_decay), cosine schedule over the epochs, gradient-norm clip 1.0;
            a pretrained conv stack trains at lr * conv_lr_mult
  inference full series, no crop -> sigmoid(customer logit), sigmoid(day logits), attention weights
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEED, IMAGE, MODEL, TRAIN
from data import date_to_day
from images import build_image, n_channels


def get_device():
    """CUDA > Apple MPS > CPU; override with PV_DEVICE=cuda|mps|cpu"""
    import os
    forced = os.environ.get("PV_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class HeatCNN(nn.Module):
    def __init__(self, hidden=32, dropout=0.25, in_ch=4):
        super().__init__()
        self.in_ch = in_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, (5, 7), padding=(2, 3)), nn.GELU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, (5, 5), padding=(2, 2)), nn.GELU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(32, 32, (5, 5), padding=(2, 2)), nn.GELU(), nn.MaxPool2d((1, 4)),
        )
        self.time = nn.Sequential(nn.Conv1d(32 * 6, hidden, 7, padding=3), nn.GELU(), nn.Dropout(dropout))
        self.day_head = nn.Conv1d(hidden, 1, 1)
        self.att = nn.Conv1d(hidden, 1, 1)
        self.cust_head = nn.Linear(hidden, 1)

    def features(self, x):
        return self.conv(x)                                               # (B, 32, L, 6)

    def heads(self, feats):
        B, C, L, S = feats.shape
        h = self.time(feats.permute(0, 1, 3, 2).reshape(B, C * S, L))     # (B, hidden, L)
        day_logit = self.day_head(h).squeeze(1)                           # (B, L)
        w = torch.softmax(self.att(h).squeeze(1), dim=1)                  # (B, L)
        pooled = (h * w.unsqueeze(1)).sum(2)                              # (B, hidden)
        return day_logit, self.cust_head(pooled).squeeze(-1), w

    def forward(self, x):
        """x (B, in_ch, L, 96) -> day logits (B, L), customer logits (B), attention (B, L)"""
        return self.heads(self.features(x))


class CNNPVModel:
    """fit(data, labels) / predict(data, cids) on data = {cid: {'days': int32[L], 'X': float32[L, 96]}}"""

    def __init__(self, image=None, calendar=None, pretrained=None, conv_lr_mult=None, hidden=None, dropout=None, epochs=None,
                 lr=None, weight_decay=None, min_crop=None, max_crop=None, day_loss_weight=None, seed=SEED, verbose=False):
        """every None falls back to config.yaml (image / model / train sections)"""
        self.image = image or IMAGE["mode"]; self.calendar = IMAGE["calendar"] if calendar is None else calendar
        self.in_ch = n_channels(self.image, self.calendar)
        self.pretrained = pretrained
        self.conv_lr_mult = TRAIN["conv_lr_mult"] if conv_lr_mult is None else conv_lr_mult
        self.hidden = MODEL["hidden"] if hidden is None else hidden
        self.dropout = MODEL["dropout"] if dropout is None else dropout
        self.epochs = TRAIN["epochs"] if epochs is None else epochs
        self.lr = TRAIN["lr"] if lr is None else lr
        self.wd = TRAIN["weight_decay"] if weight_decay is None else weight_decay
        self.min_crop = TRAIN["min_crop"] if min_crop is None else min_crop
        self.max_crop = TRAIN["max_crop"] if max_crop is None else max_crop
        self.day_w = TRAIN["day_loss_weight"] if day_loss_weight is None else day_loss_weight
        self.seed, self.verbose = seed, verbose
        self.net = None
        self.device = get_device()

    def __getstate__(self):                      # pickled on the CPU ...
        d = self.__dict__.copy()
        if d.get("net") is not None:
            d["net"] = copy.deepcopy(self.net).cpu()
        d["device"] = None
        return d

    def __setstate__(self, d):                   # ... restored on this machine's device
        self.__dict__.update(d); self.device = get_device()
        if self.net is not None:
            self.net = self.net.to(self.device).eval()

    def image_of(self, d):
        return build_image(d["X"], d["days"], self.image, self.calendar)

    @staticmethod
    def day_labels(days, row):
        if row["pv_installed"] and row["pv_installation_date"]:
            return (days >= date_to_day(row["pv_installation_date"])).astype(np.float32)
        return np.full(len(days), float(bool(row["pv_installed"])), np.float32)

    def _build_net(self):
        net = HeatCNN(self.hidden, self.dropout, in_ch=self.in_ch).to(self.device)
        if not self.pretrained:
            return net, torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        ck = torch.load(self.pretrained, map_location="cpu")
        if ck.get("image") != self.image or ck.get("calendar", True) != self.calendar:
            raise ValueError(f"pretrained conv stack is for image='{ck.get('image')}' calendar={ck.get('calendar', True)}, "
                             f"model uses image='{self.image}' calendar={self.calendar}")
        net.conv.load_state_dict(ck["conv"])
        conv_ids = {id(p) for p in net.conv.parameters()}
        opt = torch.optim.AdamW([{"params": list(net.conv.parameters()), "lr": self.lr * self.conv_lr_mult},
                                 {"params": [p for p in net.parameters() if id(p) not in conv_ids], "lr": self.lr}],
                                weight_decay=self.wd)
        return net, opt

    def fit(self, data, labels):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        lab = labels.set_index("customer_id")
        cids = [c for c in lab.index if c in data]
        imgs = [self.image_of(data[c]) for c in cids]
        ydays = [self.day_labels(data[c]["days"], lab.loc[c]) for c in cids]
        ycust = [float(bool(lab.at[c, "pv_installed"])) for c in cids]

        self.net, opt = self._build_net()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Lmax = min(min(im.shape[1] for im in imgs), self.max_crop)
        for ep in range(self.epochs):
            self.net.train()
            L = int(rng.integers(min(self.min_crop, Lmax), Lmax + 1))
            xs, yds, ycs = [], [], []
            for im, yd, yc in zip(imgs, ydays, ycust):
                a = int(rng.integers(0, im.shape[1] - L + 1))
                xs.append(im[:, a:a + L]); yds.append(yd[a:a + L])
                ycs.append(float(yd[a:a + L].max() > 0) if yc > 0 else 0.0)
            x = torch.from_numpy(np.stack(xs)).to(self.device)
            yd_t = torch.from_numpy(np.stack(yds)).to(self.device)
            yc_t = torch.tensor(ycs, device=self.device)
            day_logit, cust_logit, _ = self.net(x)
            loss = F.binary_cross_entropy_with_logits(cust_logit, yc_t) \
                + self.day_w * F.binary_cross_entropy_with_logits(day_logit, yd_t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            opt.step(); sched.step()
            if self.verbose and (ep + 1) % 25 == 0:
                print(f"    epoch {ep + 1}/{self.epochs}: loss {loss.item():.3f}", flush=True)
        return self

    @torch.no_grad()
    def predict(self, data, cids=None):
        """-> ({cid: P(PV)}, {cid: float32[L] per-day P(PV active)}, {cid: float32[L] attention})"""
        self.net.eval(); cids = list(data) if cids is None else cids
        cust, day, att = {}, {}, {}
        for c in cids:
            x = torch.from_numpy(self.image_of(data[c])[None]).to(self.device)
            dl, cl, w = self.net(x)
            cust[c] = float(torch.sigmoid(cl)[0])
            day[c] = torch.sigmoid(dl)[0].cpu().numpy().astype(np.float32)
            att[c] = w[0].cpu().numpy().astype(np.float32)
        return cust, day, att
