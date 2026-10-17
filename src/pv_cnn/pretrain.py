"""Step 3a (optional, no labels): self-supervised pretraining of the conv stack on every customer's image.

Masked-patch reconstruction: random 12-day x 12-slot patches covering ~40% of a 180-day crop are hidden
(image channel zeroed, mask channel = 1) and a small decoder must reconstruct the hidden image values from
the conv features. Loss = MSE on hidden patches + 0.3 * MSE on visible ones. The decoder is discarded;
HeatCNN.conv is saved to artifacts/pretrained_<image>.pt and loaded by train.py --pretrained.
    python pretrain.py --image relsmooth [--sets labelled unlabelled] [--steps 3000]"""
import argparse, time
import numpy as np
import torch, torch.nn as nn

from config import CACHE_DIR, ARTIFACT_DIR, SEED, IMAGE, PRETRAIN
from data import load_cache, per_customer
from images import build_image, IMAGE_CHANNELS, n_channels
from cnn_model import HeatCNN, get_device


class Decoder2D(nn.Module):
    """(32, L, 6) -> (out_ch, L, 96): undo the three slot-axis poolings"""
    def __init__(self, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=(1, 4)), nn.Conv2d(32, 32, (5, 5), padding=(2, 2)), nn.GELU(),
            nn.Upsample(scale_factor=(1, 2)), nn.Conv2d(32, 16, (5, 5), padding=(2, 2)), nn.GELU(),
            nn.Upsample(scale_factor=(1, 2)), nn.Conv2d(16, out_ch, (5, 7), padding=(2, 3)))

    def forward(self, h):
        return self.net(h)


def patch_mask(B, L, frac=None, pd=None, ps=None, generator=None):
    frac = PRETRAIN["mask_frac"] if frac is None else frac
    pd = PRETRAIN["patch"][0] if pd is None else pd; ps = PRETRAIN["patch"][1] if ps is None else ps
    gl, gs = int(np.ceil(L / pd)), int(np.ceil(96 / ps))
    m = (torch.rand(B, gl, gs, generator=generator) < frac).float()
    return m.repeat_interleave(pd, 1).repeat_interleave(ps, 2)[:, :L, :96]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=IMAGE["mode"], choices=list(IMAGE_CHANNELS))
    ap.add_argument("--sets", nargs="+", default=["labelled", "unlabelled"])
    ap.add_argument("--steps", type=int, default=PRETRAIN["steps"])
    ap.add_argument("--batch", type=int, default=PRETRAIN["batch"])
    ap.add_argument("--crop", type=int, default=PRETRAIN["crop"])
    ap.add_argument("--lr", type=float, default=PRETRAIN["lr"])
    ap.add_argument("--calendar", default=IMAGE["calendar"], type=lambda s: s.lower() in ("1", "true", "yes"), help="true|false: day-of-year channels")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cal = bool(args.calendar)
    out = args.out or str(ARTIFACT_DIR / f"pretrained_{args.image}{'' if cal else '_nocal'}.pt")
    nc = IMAGE_CHANNELS[args.image]
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); g = torch.Generator().manual_seed(SEED)

    imgs = []
    for name in args.sets:
        p = CACHE_DIR / f"{name}.npz"
        if p.exists():
            imgs += [build_image(d["X"], d["days"], args.image, cal, with_mask=False) for d in per_customer(load_cache(p)).values()]
    imgs = [im for im in imgs if im.shape[1] >= args.crop]
    device = get_device()
    print(f"{len(imgs)} customers with >= {args.crop} days   image={args.image}   device {device}")
    enc = HeatCNN(in_ch=n_channels(args.image, cal)).conv.to(device); dec = Decoder2D(nc).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=PRETRAIN["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    t0 = time.time(); run = 0.0
    for step in range(1, args.steps + 1):
        xs = []
        for i in rng.integers(0, len(imgs), args.batch):
            im = imgs[i]; a = int(rng.integers(0, im.shape[1] - args.crop + 1)); xs.append(im[:, a:a + args.crop])
        x = torch.from_numpy(np.stack(xs)).to(device)                        # (B, C[+2], L, 96)
        m = patch_mask(args.batch, args.crop, generator=g).to(device)        # (B, L, 96)
        target = x[:, :nc]
        xin = torch.cat([x[:, :nc] * (1 - m[:, None]), x[:, nc:], m[:, None]], dim=1)
        err = ((dec(enc(xin)) - target) ** 2).mean(1)
        loss = (err * m).sum() / m.sum().clamp(min=1) + PRETRAIN["visible_loss_weight"] * (err * (1 - m)).sum() / (1 - m).sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run += loss.item()
        if step % 250 == 0:
            print(f"step {step}/{args.steps}  loss {run / 250:.4f}  ({time.time() - t0:.0f}s)", flush=True); run = 0.0
    ARTIFACT_DIR.mkdir(exist_ok=True)
    torch.save({"conv": enc.cpu().state_dict(), "image": args.image, "calendar": cal}, out); print("saved", out)


if __name__ == "__main__":
    main()
