"""Where does the 2-D CNN look?  Grad-CAM and day attention on the (days x hours) image.

    python plot_attention.py --model model            # all 40 labelled customers
    python plot_attention.py --model <dir> <customer_id> ... --set labelled
    python plot_attention.py --model <dir> <customer_id> ... --set unlabelled

Per customer -> plots/attention/<set>/attn_<id>.png with four panels:
  1. the image the CNN sees (relsmooth: blue = midday deficit), true date (black dashed) and model date (green)
  2. Grad-CAM evidence FOR "PV"      (gradient of the PV logit w.r.t. the last conv feature maps, ReLU, upsampled)
  3. Grad-CAM evidence AGAINST "PV"  (same with the sign flipped)
  4. the network's own attention over days (which days the customer decision pooled from)
Plus plots/attention/<set>/sheet_<set>.png: one Grad-CAM-for-PV thumbnail per customer.
Only the model's view is shown; the overlay is normalised per customer (red = strongest evidence)."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter

from config import CACHE_DIR, ARTIFACT_DIR, LABELS_CSV, PROJECT_DIR
from data import load_cache, per_customer, read_labels, date_to_day, day_to_date
from pipeline import PVPipeline
from plot_heatmap import _dnum, DCMAP

OUT = PROJECT_DIR / "plots" / "attention"


def cnn_forward_parts(net, x):
    """replicates HeatCNN.forward but keeps the conv feature maps for Grad-CAM"""
    feats = net.conv(x); feats.retain_grad()
    B, C, L, S = feats.shape
    h = net.time(feats.permute(0, 1, 3, 2).reshape(B, C * S, L))
    day_logit = net.day_head(h).squeeze(1)
    w = torch.softmax(net.att(h).squeeze(1), dim=1)
    cust_logit = net.cust_head((h * w.unsqueeze(1)).sum(2)).squeeze(-1)
    return feats, day_logit, cust_logit, w


def grad_cam(model, img):
    """img (C+3, L, 96) -> dict with prob, cam_pv, cam_no (L, 96 in [0, 1]), attention (L,), day_prob (L,)"""
    net = model.net.eval()
    x = torch.from_numpy(img[None]).to(model.device).requires_grad_(True)
    feats, day_logit, cust_logit, w = cnn_forward_parts(net, x)
    net.zero_grad(); cust_logit.sum().backward()
    g = feats.grad[0]                                   # (32, L, 6)
    f = feats[0].detach()
    L = img.shape[1]

    def cam(sign):
        wts = (sign * g).mean(dim=(1, 2))               # global-average-pooled gradients -> channel weights
        m = torch.relu((wts[:, None, None] * f).sum(0))  # (L, 6)
        m = F.interpolate(m[None, None], size=(L, 96), mode="bilinear", align_corners=False)[0, 0]
        m = m.cpu().numpy(); return m / max(float(m.max()), 1e-8)
    return {"prob": float(torch.sigmoid(cust_logit)[0]), "cam_pv": cam(1.0), "cam_no": cam(-1.0),
            "attention": w[0].detach().cpu().numpy(), "day_prob": torch.sigmoid(day_logit)[0].detach().cpu().numpy()}


def dense(days, Z, fill=np.nan):
    grid = np.arange(days.min(), days.max() + 1)
    G = np.full((len(grid), Z.shape[1]) if Z.ndim == 2 else (len(grid),), fill, np.float32)
    G[days - days.min()] = Z
    return grid, G


def overlay(ax, grid, base, heat, dates, title):
    ax.imshow(base, aspect="auto", cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest", extent=[0, 24, dates[-1] + 1, dates[0]])
    ax.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap="jet", vmin=0, vmax=1, alpha=0.55, interpolation="bilinear",
              extent=[0, 24, dates[-1] + 1, dates[0]])
    ax.set_ylim(dates[-1] + 1, dates[0]); ax.set_title(title, loc="left", fontsize=10)


def plot_customer(cid, d, prep, model, truth, out_dir, pred_date=None):
    img = model._image(cid, prep); r = grad_cam(model, img)
    days = prep[cid]["days"]
    grid, base = dense(days, img[0]); _, cam_pv = dense(days, r["cam_pv"]); _, cam_no = dense(days, r["cam_no"])
    _, att = dense(days, r["attention"], 0.0); _, dp = dense(days, r["day_prob"], np.nan)
    dates = _dnum(grid)
    fig, axes = plt.subplots(1, 4, figsize=(17, 8), dpi=110, sharey=True, gridspec_kw={"width_ratios": [4, 4, 4, 1.3]})
    fig.patch.set_facecolor("white")
    axes[0].imshow(base, aspect="auto", cmap=DCMAP, norm=TwoSlopeNorm(0.0, -1.0, 1.0), interpolation="nearest", extent=[0, 24, dates[-1] + 1, dates[0]])
    axes[0].set_ylim(dates[-1] + 1, dates[0]); axes[0].set_title("input image (relsmooth, blue = midday deficit)", loc="left", fontsize=10)
    overlay(axes[1], grid, base, cam_pv, dates, "Grad-CAM: evidence FOR PV")
    overlay(axes[2], grid, base, cam_no, dates, "Grad-CAM: evidence AGAINST PV")
    axes[3].fill_betweenx(dates + 0.5, 0, att / max(att.max(), 1e-8), color="#7b2cbf", alpha=0.6, step="mid", label="day attention")
    axes[3].plot(np.nan_to_num(dp), dates + 0.5, color="#0b7a3b", lw=0.8, label="day PV prob")
    axes[3].set_xlim(0, 1.02); axes[3].set_title("attention / day prob", loc="left", fontsize=10); axes[3].legend(fontsize=7, loc="lower right")
    tr = truth.get(cid)
    for ax in axes[:3]:
        if tr is not None and tr[0] and tr[1]:
            ax.axhline(_dnum(date_to_day(tr[1])), color="#1a1a1a", lw=1.6, ls="--")
        if pred_date:
            ax.axhline(_dnum(date_to_day(pred_date)), color="#0b7a3b", lw=1.6)
        ax.set_xticks(range(0, 25, 6)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 6)]); ax.set_xlabel("hour")
        for s in ax.spines.values():
            s.set_visible(False)
    tick_days = [g for g in grid if str(day_to_date(g))[8:10] == "01" and str(day_to_date(g))[5:7] in ("01", "07")]
    axes[0].set_yticks(_dnum(np.array(tick_days))); axes[0].set_yticklabels([str(day_to_date(g))[:7] for g in tick_days])
    tt = f"truth: {'PV from ' + tr[1] if tr and tr[0] else 'no PV'}" if tr is not None else "truth: unknown"
    fig.suptitle(f"{cid}   model: {'PV' if r['prob'] >= 0.5 else 'no PV'} (prob {r['prob']:.2f}){'  from ' + pred_date if pred_date else ''}     {tt}",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_dir / f"attn_{cid}.png"); plt.close(fig)
    return r["prob"], grid, base, cam_pv, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--model", default=str(ARTIFACT_DIR), help="directory with encoder.pt + downstream.joblib of a --clf cnn run")
    ap.add_argument("--set", default="labelled", choices=["labelled", "unlabelled"])
    ap.add_argument("--no-sheet", action="store_true")
    a = ap.parse_args()
    pipe = PVPipeline.load(Path(a.model))
    if pipe.clf != "cnn" or pipe.seq_model is None:
        raise SystemExit(f"{a.model} holds a '{pipe.clf}' pipeline; Grad-CAM needs a --clf cnn model (train_downstream.py --clf cnn ... --save-to DIR)")
    model = pipe.seq_model
    data = per_customer(load_cache(CACHE_DIR / f"{a.set}.npz"))
    ids = a.ids or (sorted(data, key=lambda s: int(s[3:])) if a.set == "labelled" else [])
    if not ids:
        raise SystemExit("give customer ids for --set unlabelled")
    missing = [c for c in ids if c not in data]
    if missing:
        raise SystemExit(f"not in {a.set}.npz: {missing}")
    truth = {}
    if a.set == "labelled":
        for _, r in read_labels(LABELS_CSV).iterrows():
            truth[r["customer_id"]] = (bool(r["pv_installed"]), r["pv_installation_date"])
    out_dir = OUT / a.set; out_dir.mkdir(parents=True, exist_ok=True)
    prep = pipe.prepare({c: data[c] for c in ids})
    pred = pipe.predict(prep).set_index("customer_id")
    print(f"model {a.model} (image={model.image})  device {model.device}  customers {len(ids)}")
    thumbs = []
    for i, c in enumerate(ids):
        pd_ = pred.at[c, "pv_installation_date"] if bool(pred.at[c, "pv_installed"]) else None
        thumbs.append((c,) + plot_customer(c, data[c], prep, model, truth, out_dir, pd_ or None))
        print(f"  {i + 1}/{len(ids)} {c}", end="\r", flush=True)
    print(f"\nwrote {len(ids)} images to {out_dir}")
    if not a.no_sheet and len(ids) > 1:
        ncol = 8; nrow = int(np.ceil(len(thumbs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2), dpi=120); fig.patch.set_facecolor("white")
        for ax, (c, prob, grid, base, cam, dates) in zip(np.ravel(axes), thumbs):
            overlay(ax, grid, base, cam, dates, ""); ax.set_xticks([]); ax.set_yticks([])
            tr = truth.get(c)
            if tr is not None and tr[0] and tr[1]:
                ax.axhline(_dnum(date_to_day(tr[1])), color="#1a1a1a", lw=1.2, ls="--")
            ax.set_title(f"{c}  {'PV' if tr and tr[0] else ('no PV' if tr else '?')}  model {prob:.2f}", fontsize=8, loc="left")
        for ax in np.ravel(axes)[len(thumbs):]:
            ax.axis("off")
        fig.suptitle("Grad-CAM evidence for PV on the relsmooth image (red = strongest), dashed = true commissioning date", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out_dir / f"sheet_{a.set}.png"); plt.close(fig)
        print("wrote", out_dir / f"sheet_{a.set}.png")


if __name__ == "__main__":
    main()
