"""Step 2: self-supervised pretraining of the daily-profile encoder on ALL customers
(labelled + unlabelled, no labels used). Masked-reconstruction objective."""
import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from config import CACHE_DIR, ARTIFACT_DIR, SEED, EMB_DIM
from data import load_cache, per_customer
from ssl_model import (Encoder, Decoder, customer_scale, normalize_days, random_block_mask,
                       get_device, save_encoder)


def build_training_matrix(names, norm="linear"):
    parts = []
    for name in names:
        p = CACHE_DIR / f"{name}.npz"
        if not p.exists():
            print(f"  (skipping {name}: {p} not found)"); continue
        pc = per_customer(load_cache(p))
        for cid, d in pc.items():
            parts.append(normalize_days(d["X"], customer_scale(d["X"]), norm, d["days"]))
        print(f"  {name}: {len(pc)} customers, {sum(len(x) for x in parts):,} days so far")
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-days", type=int, default=0, help="subsample days (0 = use all)")
    ap.add_argument("--sets", nargs="+", default=["labelled", "unlabelled"])
    ap.add_argument("--out", default=str(ARTIFACT_DIR / "encoder.pt"))
    ap.add_argument("--norm", default="linear", choices=["linear", "log", "minmax", "relsmooth"], help="input normalisation of the days")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    X = build_training_matrix(args.sets, args.norm)
    rng = np.random.default_rng(SEED)
    rng.shuffle(X)
    if args.max_days and len(X) > args.max_days:
        X = X[:args.max_days]
    n_val = max(1000, int(0.02 * len(X)))
    Xval, Xtr = X[:n_val], X[n_val:]
    print(f"train days {len(Xtr):,}  val days {len(Xval):,}")

    device = get_device(); print("device:", device)
    enc, dec = Encoder(EMB_DIM, args.norm).to(device), Decoder(EMB_DIM).to(device)
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(Xtr) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch)
    g = torch.Generator().manual_seed(SEED)

    def loss_fn(xb, mb):
        xin = xb * (1 - mb) + 0.02 * torch.randn_like(xb)
        rec = dec(enc(xin, mb))
        err = (rec - xb) ** 2
        return (err * mb).sum() / mb.sum().clamp(min=1) + 0.3 * (err * (1 - mb)).sum() / (1 - mb).sum().clamp(min=1)

    Xtr_t = torch.from_numpy(Xtr); Xval_t = torch.from_numpy(Xval).to(device)
    for ep in range(args.epochs):
        enc.train(); dec.train(); t0 = time.time(); tot = 0.0
        perm = torch.randperm(len(Xtr_t), generator=g)
        for s in range(steps_per_epoch):
            idx = perm[s * args.batch:(s + 1) * args.batch]
            xb = Xtr_t[idx].to(device)
            mb = random_block_mask(len(idx), generator=g).to(device)
            loss = loss_fn(xb, mb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        enc.eval(); dec.eval()
        with torch.no_grad():
            vm = random_block_mask(len(Xval_t), generator=torch.Generator().manual_seed(0)).to(device)
            vl = loss_fn(Xval_t, vm).item()
        print(f"epoch {ep + 1}/{args.epochs}  train {tot / steps_per_epoch:.4f}  val {vl:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    save_encoder(enc, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
