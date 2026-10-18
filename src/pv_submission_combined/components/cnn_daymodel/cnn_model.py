"""2-D convolutional PV detector on a customer's (days x hours) image.

Input image
-----------
One customer = one image of shape (channels, L, 96): L calendar days (rows) x 96 fifteen-minute slots
(columns). Channels, see features.customer_image():
  image channel(s)  'heat'      : kW / the customer's 95th percentile, gamma 0.6, in [0, 1]
                    'relsmooth' : log2(kW) minus the day's own 06-09h / 17-20h shoulder level, then a
                                  15-day x 3-slot median filter, clipped to +-3 and scaled to [-1, 1]
                                  (negative = less grid draw at midday than at the shoulders = PV)
                    'both'      : the two above
  doy_sin, doy_cos  : day of year as sine / cosine, repeated over the 96 slots (season context)
  mask              : all zeros at training / inference; 1 where a patch is hidden in ssl2d_pretrain.py

Network (HeatCNN)
-----------------
  conv stack (shared with the self-supervised pretraining):
    Conv2d(in_ch -> 16, kernel 5 days x 7 slots)  GELU  MaxPool(1 x 2)   slots 96 -> 48
    Conv2d(16 -> 32,    kernel 5 x 5)             GELU  MaxPool(1 x 2)   slots 48 -> 24
    Conv2d(32 -> 32,    kernel 5 x 5)             GELU  MaxPool(1 x 4)   slots 24 -> 6
    -> feature maps (32, L, 6): the day axis is never pooled, so every day keeps its own features
  temporal head:
    Conv1d(32 * 6 = 192 -> 32, kernel 7 days)  GELU  Dropout(0.25)      -> (32, L)
    day head        Conv1d(32 -> 1, 1)                                 -> one logit per day "PV active"
    attention       Conv1d(32 -> 1, 1), softmax over days               -> which days matter
    customer head   Linear(32 -> 1) on the attention-weighted mean      -> one logit "customer has PV"

Training (CNNPVModel.fit)
-------------------------
  labels    : customer y = PV yes/no; day y_t = 1 for days on/after the commissioning date of a PV customer
  epochs    : each epoch is ONE batch holding every training customer. A crop length L is drawn in
              [min_crop, min(max_crop, shortest series)] and every customer contributes a random window
              of L consecutive days. The customer label of a window is 1 only if the window contains at
              least one PV day, so a window before the commissioning date is a negative example.
  loss      : BCE(customer logit) + day_loss_weight * BCE(day logits)
  optimiser : AdamW(lr, weight_decay), cosine schedule over the epochs, gradient-norm clipping at 1.0;
              with a pretrained conv stack its learning rate is lr * conv_lr_mult
  inference : the full series (no crop) -> sigmoid(customer logit) and sigmoid(day logits)

Self-supervised pretraining of the conv stack: ssl2d_pretrain.py (masked-patch reconstruction).
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SEED
from data import date_to_day
from features import customer_image, IMAGE_CHANNELS
from ssl_model import get_device


class HeatCNN(nn.Module):
    def __init__(self, hidden=32, dropout=0.25, in_ch=4):
        """in_ch = image channels + 2 calendar channels (doy sin/cos) + 1 mask channel"""
        super().__init__()
        self.in_ch = in_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, (5, 7), padding=(2, 3)), nn.GELU(), nn.MaxPool2d((1, 2)),   # slots 96 -> 48
            nn.Conv2d(16, 32, (5, 5), padding=(2, 2)), nn.GELU(), nn.MaxPool2d((1, 2)),      # 48 -> 24
            nn.Conv2d(32, 32, (5, 5), padding=(2, 2)), nn.GELU(), nn.MaxPool2d((1, 4)),      # 24 -> 6
        )
        self.time = nn.Sequential(nn.Conv1d(32 * 6, hidden, 7, padding=3), nn.GELU(), nn.Dropout(dropout))
        self.day_head = nn.Conv1d(hidden, 1, 1)
        self.att = nn.Conv1d(hidden, 1, 1)
        self.cust_head = nn.Linear(hidden, 1)

    def forward(self, x):
        """x (B, in_ch, L, 96) -> day logits (B, L), customer logits (B), attention weights (B, L)"""
        h = self.conv(x)                                                  # (B, 32, L, 6)
        B, C, L, S = h.shape
        h = self.time(h.permute(0, 1, 3, 2).reshape(B, C * S, L))         # (B, hidden, L)
        day_logit = self.day_head(h).squeeze(1)                           # (B, L)
        w = torch.softmax(self.att(h).squeeze(1), dim=1)                  # (B, L)
        pooled = (h * w.unsqueeze(1)).sum(2)                              # (B, hidden)
        return day_logit, self.cust_head(pooled).squeeze(-1), w


class CNNPVModel:
    """sklearn-like wrapper used by pipeline.PVPipeline(clf="cnn"):
    fit(prep, labels) / predict(prep, cids) -> (customer prob dict, day prob dict)."""

    def __init__(self, image="heat", pretrained=None, conv_lr_mult=0.2, hidden=32, epochs=150, lr=2e-3,
                 weight_decay=1e-3, min_crop=120, max_crop=365, day_loss_weight=1.0, seed=SEED, verbose=False):
        self.image, self.in_ch = image, IMAGE_CHANNELS[image] + 3
        self.pretrained, self.conv_lr_mult = pretrained, conv_lr_mult      # path to ssl2d_pretrain.py weights
        self.hidden, self.epochs, self.lr, self.wd = hidden, epochs, lr, weight_decay
        self.min_crop, self.max_crop, self.day_w, self.seed, self.verbose = min_crop, max_crop, day_loss_weight, seed, verbose
        self.net = None
        self.device = get_device()

    # ---- persistence: the network is pickled on the CPU and restored on this machine's device ----
    def __getstate__(self):
        d = self.__dict__.copy()
        if d.get("net") is not None:
            d["net"] = copy.deepcopy(self.net).cpu()
        d["device"] = None
        return d

    def __setstate__(self, d):
        self.__dict__.update(d); self.device = get_device()
        if self.net is not None:
            self.net = self.net.to(self.device).eval()

    # ---- data ----
    def _image(self, cid, prep):
        """(in_ch, L, 96) float32 image of one prepared customer"""
        px = customer_image(prep[cid]["X"], prep[cid]["days"], self.image)          # (C, L, 96)
        tab = prep[cid]["tab"]
        ds = np.repeat(tab["doy_sin"].to_numpy(np.float32)[:, None], 96, 1)
        dc = np.repeat(tab["doy_cos"].to_numpy(np.float32)[:, None], 96, 1)
        return np.concatenate([px, ds[None], dc[None], np.zeros_like(ds)[None]])   # mask channel = 0

    @staticmethod
    def _day_labels(days, row):
        if row["pv_installed"] and row["pv_installation_date"]:
            return (days >= date_to_day(row["pv_installation_date"])).astype(np.float32)
        return np.full(len(days), float(bool(row["pv_installed"])), np.float32)

    # ---- training ----
    def _build_net(self):
        net = HeatCNN(self.hidden, in_ch=self.in_ch).to(self.device)
        if not self.pretrained:
            return net, torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.wd)
        ck = torch.load(self.pretrained, map_location="cpu")
        if ck.get("image", "heat") != self.image:
            raise ValueError(f"pretrained conv stack was trained on image='{ck.get('image', 'heat')}', model uses '{self.image}'")
        net.conv.load_state_dict(ck["conv"])
        conv_ids = {id(p) for p in net.conv.parameters()}
        opt = torch.optim.AdamW([{"params": list(net.conv.parameters()), "lr": self.lr * self.conv_lr_mult},
                                 {"params": [p for p in net.parameters() if id(p) not in conv_ids], "lr": self.lr}],
                                weight_decay=self.wd)
        return net, opt

    def fit(self, prep, labels):
        torch.manual_seed(self.seed); rng = np.random.default_rng(self.seed)
        lab = labels.set_index("customer_id")
        cids = [c for c in lab.index if c in prep]
        imgs = [self._image(c, prep) for c in cids]
        ydays = [self._day_labels(prep[c]["tab"]["day"].to_numpy(), lab.loc[c]) for c in cids]
        ycust = [float(bool(lab.at[c, "pv_installed"])) for c in cids]

        self.net, opt = self._build_net()
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        Lmax = min(min(im.shape[1] for im in imgs), self.max_crop)
        for ep in range(self.epochs):
            self.net.train()
            L = int(rng.integers(min(self.min_crop, Lmax), Lmax + 1))          # one crop length per epoch
            xs, yds, ycs = [], [], []
            for im, yd, yc in zip(imgs, ydays, ycust):
                a = int(rng.integers(0, im.shape[1] - L + 1))                  # random window per customer
                xs.append(im[:, a:a + L]); yds.append(yd[a:a + L])
                ycs.append(float(yd[a:a + L].max() > 0) if yc > 0 else 0.0)    # window label: any PV day inside?
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
                print(f"    epoch {ep + 1}: loss {loss.item():.3f}")
        return self

    # ---- inference ----
    @torch.no_grad()
    def predict(self, prep, cids=None):
        """full series per customer -> ({cid: P(PV)}, {cid: per-day P(PV active) float32[L]})"""
        self.net.eval(); cids = list(prep) if cids is None else cids
        cust, day = {}, {}
        for c in cids:
            x = torch.from_numpy(self._image(c, prep)[None]).to(self.device)
            dl, cl, _ = self.net(x)
            cust[c] = float(torch.sigmoid(cl)[0]); day[c] = torch.sigmoid(dl)[0].cpu().numpy().astype(np.float32)
        return cust, day
