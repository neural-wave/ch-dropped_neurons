"""Self-supervised pretraining of the 2-D CNN conv stack (cnn_model.HeatCNN.conv) on ALL customers' images.
Masked-patch reconstruction: random rectangles of the (days x slots) image are hidden and reconstructed.
Output: artifacts/heatcnn_ssl.pt with the conv stack weights, used by  train_downstream.py --clf cnn --cnn-pretrained ...
"""
import argparse, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from config import CACHE_DIR, ARTIFACT_DIR, SEED
from data import load_cache, per_customer, day_calendar
from features import customer_image, IMAGE_CHANNELS
from cnn_model import HeatCNN
from ssl_model import get_device


class Decoder2D(nn.Module):
    """(32, L, 6) -> (out_ch, L, 96): undo the three slot-axis poolings."""
    def __init__(self, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=(1, 4)), nn.Conv2d(32, 32, (5, 5), padding=(2, 2)), nn.GELU(),
            nn.Upsample(scale_factor=(1, 2)), nn.Conv2d(32, 16, (5, 5), padding=(2, 2)), nn.GELU(),
            nn.Upsample(scale_factor=(1, 2)), nn.Conv2d(16, out_ch, (5, 7), padding=(2, 3)))

    def forward(self, h):
        return self.net(h)


def load_images(sets, image="heat"):
    imgs = []
    for name in sets:
        p = CACHE_DIR / f"{name}.npz"
        if not p.exists():
            continue
        for cid, d in per_customer(load_cache(p)).items():
            px = customer_image(d["X"], d["days"], image); _, doy = day_calendar(d["days"])
            ds = np.repeat(np.sin(2 * np.pi * doy / 365.25).astype(np.float32)[:, None], 96, 1)
            dc = np.repeat(np.cos(2 * np.pi * doy / 365.25).astype(np.float32)[:, None], 96, 1)
            imgs.append(np.concatenate([px, ds[None], dc[None]]))   # (C+2, L, 96); mask channel added per batch
    return imgs


def patch_mask(B, L, frac=0.4, pd=12, ps=12, generator=None):
    """rectangular patch mask (B, L, 96): 1 where hidden"""
    gl, gs = int(np.ceil(L / pd)), int(np.ceil(96 / ps))
    m = (torch.rand(B, gl, gs, generator=generator) < frac).float()
    m = m.repeat_interleave(pd, 1).repeat_interleave(ps, 2)[:, :L, :96]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--crop", type=int, default=180)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sets", nargs="+", default=["labelled", "unlabelled"])
    ap.add_argument("--out", default=None, help="default artifacts/heatcnn_ssl.pt (heat) or heatcnn_ssl_<image>.pt")
    ap.add_argument("--image", default="heat", choices=list(IMAGE_CHANNELS), help="image representation (see features.customer_image)")
    args = ap.parse_args()
    args.out = args.out or str(ARTIFACT_DIR / ("heatcnn_ssl.pt" if args.image == "heat" else f"heatcnn_ssl_{args.image}.pt"))
    nc = IMAGE_CHANNELS[args.image]
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED); g = torch.Generator().manual_seed(SEED)
    imgs = [im for im in load_images(args.sets, args.image) if im.shape[1] >= args.crop]
    print(f"{len(imgs)} customers with >= {args.crop} days   image={args.image} ({nc} channel(s))")
    device = get_device(); print("device", device)
    enc = HeatCNN(in_ch=nc + 3).conv.to(device); dec = Decoder2D(nc).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps)
    t0 = time.time(); run = 0.0
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(imgs), args.batch); xs = []
        for i in idx:
            im = imgs[i]; a = int(rng.integers(0, im.shape[1] - args.crop + 1)); xs.append(im[:, a:a + args.crop])
        x = torch.from_numpy(np.stack(xs)).to(device)                       # (B, C+2, L, 96)
        m = patch_mask(args.batch, args.crop, generator=g).to(device)       # (B, L, 96)
        target = x[:, :nc]
        xin = torch.cat([x[:, :nc] * (1 - m[:, None]), x[:, nc:], m[:, None]], dim=1)   # (B, C+3, L, 96)
        rec = dec(enc(xin))
        err = ((rec - target) ** 2).mean(1)
        loss = (err * m).sum() / m.sum().clamp(min=1) + 0.3 * (err * (1 - m)).sum() / (1 - m).sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); run += loss.item()
        if step % 250 == 0:
            print(f"step {step}/{args.steps}  loss {run / 250:.4f}  ({time.time() - t0:.0f}s)", flush=True); run = 0.0
    torch.save({"conv": enc.cpu().state_dict(), "image": args.image}, args.out); print("saved", args.out)


if __name__ == "__main__":
    main()
