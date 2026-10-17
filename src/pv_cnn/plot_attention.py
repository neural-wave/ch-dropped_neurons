"""Step 7b: Grad-CAM - where the CNN looks when it decides.
    python plot_attention.py --model artifacts/model_humans                       # all labelled customers + sheet
    python plot_attention.py --model artifacts/model_humans --set unlabelled <customer_id> ...
Per customer, plots/attention/<set>/<id>.png with four panels: the input image with the true (black dashed) and
predicted (green) dates; Grad-CAM evidence FOR PV (gradient of the customer logit w.r.t. the last conv feature
maps, channel-weighted, ReLU, upsampled to days x slots, normalised per customer); the same AGAINST PV; the
network's attention over days with the per-day PV probability."""
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from config import PLOTS_DIR, CACHE_DIR, ARTIFACT_DIR, LABELS_CSV
from data import load_cache, per_customer, read_labels, date_to_day, day_to_date
from pipeline import CNNPipeline
from plot_images import dnum, DCMAP

OUT = PLOTS_DIR / "attention"


def grad_cam(cnn, d):
    net = cnn.net.eval(); img = cnn.image_of(d)
    x = torch.from_numpy(img[None]).to(cnn.device).requires_grad_(True)
    feats = net.features(x); feats.retain_grad()
    day_logit, cust_logit, w = net.heads(feats)
    net.zero_grad(); cust_logit.sum().backward()
    g, f, L = feats.grad[0], feats[0].detach(), img.shape[1]

    def cam(sign):
        wts = (sign * g).mean(dim=(1, 2))
        m = torch.relu((wts[:, None, None] * f).sum(0))
        m = F.interpolate(m[None, None], size=(L, 96), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        return m / max(float(m.max()), 1e-8)
    return {"prob": float(torch.sigmoid(cust_logit)[0]), "cam_pv": cam(1.0), "cam_no": cam(-1.0), "image": img[0],
            "attention": w[0].detach().cpu().numpy(), "day_prob": torch.sigmoid(day_logit)[0].detach().cpu().numpy()}


def dense(days, Z, fill=np.nan):
    grid = np.arange(days.min(), days.max() + 1)
    G = np.full((len(grid),) + Z.shape[1:], fill, np.float32); G[days - days.min()] = Z
    return grid, G


def overlay(ax, base, heat, dates, title=""):
    ax.imshow(base, aspect="auto", cmap="gray_r", vmin=-1, vmax=1, interpolation="nearest", extent=[0, 24, dates[-1] + 1, dates[0]])
    ax.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap="jet", vmin=0, vmax=1, alpha=0.55, interpolation="bilinear",
              extent=[0, 24, dates[-1] + 1, dates[0]])
    ax.set_ylim(dates[-1] + 1, dates[0]); ax.set_title(title, loc="left", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_customer(cid, d, cnn, truth, pred_date, out_dir):
    r = grad_cam(cnn, d); days = d["days"]
    grid, base = dense(days, r["image"]); _, cam_pv = dense(days, r["cam_pv"]); _, cam_no = dense(days, r["cam_no"])
    _, att = dense(days, r["attention"], 0.0); _, dp = dense(days, r["day_prob"])
    dates = dnum(grid)
    fig, axes = plt.subplots(1, 4, figsize=(17, 8), dpi=110, sharey=True, gridspec_kw={"width_ratios": [4, 4, 4, 1.3]})
    fig.patch.set_facecolor("white")
    axes[0].imshow(base, aspect="auto", cmap=DCMAP, norm=TwoSlopeNorm(0.0, -1.0, 1.0), interpolation="nearest", extent=[0, 24, dates[-1] + 1, dates[0]])
    axes[0].set_ylim(dates[-1] + 1, dates[0]); axes[0].set_title("input image (relsmooth, blue = midday deficit)", loc="left", fontsize=10)
    overlay(axes[1], base, cam_pv, dates, "Grad-CAM: evidence FOR PV")
    overlay(axes[2], base, cam_no, dates, "Grad-CAM: evidence AGAINST PV")
    axes[3].fill_betweenx(dates + 0.5, 0, att / max(att.max(), 1e-8), color="#7b2cbf", alpha=0.6, step="mid", label="day attention")
    axes[3].plot(np.nan_to_num(dp), dates + 0.5, color="#0b7a3b", lw=0.8, label="day PV prob")
    axes[3].set_xlim(0, 1.02); axes[3].set_title("attention / day prob", loc="left", fontsize=10); axes[3].legend(fontsize=7, loc="lower right")
    t = truth.get(cid)
    for ax in axes[:3]:
        if t and t[0] and t[1]:
            ax.axhline(dnum(date_to_day(t[1])), color="#1a1a1a", lw=1.6, ls="--")
        if pred_date:
            ax.axhline(dnum(date_to_day(pred_date)), color="#0b7a3b", lw=1.6)
        ax.set_xticks(range(0, 25, 6)); ax.set_xticklabels([f"{h:02d}" for h in range(0, 25, 6)]); ax.set_xlabel("hour")
    ticks = [g for g in grid if str(day_to_date(g))[8:10] == "01" and str(day_to_date(g))[5:7] in ("01", "07")]
    axes[0].set_yticks(dnum(np.array(ticks))); axes[0].set_yticklabels([str(day_to_date(g))[:7] for g in ticks])
    tt = f"truth: {'PV from ' + t[1] if t[0] else 'no PV'}" if t else "truth: unknown"
    fig.suptitle(f"{cid}   model: {'PV' if r['prob'] >= 0.5 else 'no PV'} (prob {r['prob']:.2f}){'  from ' + pred_date if pred_date else ''}     {tt}",
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out_dir / f"{cid}.png"); plt.close(fig)
    return r["prob"], base, cam_pv, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--model", default=str(ARTIFACT_DIR / "model_humans"))
    ap.add_argument("--set", default="labelled", choices=["labelled", "unlabelled"])
    a = ap.parse_args()
    pipe = CNNPipeline.load(a.model); cnn = pipe.cnn
    data = per_customer(load_cache(CACHE_DIR / f"{a.set}.npz"))
    ids = a.ids or (sorted(data, key=lambda s: int(s[3:])) if a.set == "labelled" else [])
    if not ids:
        raise SystemExit("give customer ids for --set unlabelled")
    truth = {}
    if a.set == "labelled":
        for _, r in read_labels(LABELS_CSV).iterrows():
            truth[r["customer_id"]] = (bool(r["pv_installed"]), r["pv_installation_date"])
    out_dir = OUT / a.set; out_dir.mkdir(parents=True, exist_ok=True)
    pred = pipe.predict(data, ids).set_index("customer_id")
    print(f"model {a.model} (image={cnn.image}, calendar={cnn.calendar}, device {cnn.device})  {len(ids)} customers")
    thumbs = []
    for i, c in enumerate(ids):
        pdate = pred.at[c, "pv_installation_date"] if bool(pred.at[c, "pv_installed"]) else None
        thumbs.append((c,) + plot_customer(c, data[c], cnn, truth, pdate or None, out_dir)); print(f"  {i + 1}/{len(ids)}", end="\r", flush=True)
    print(f"\nwrote {len(ids)} images to {out_dir}")
    if len(ids) > 1:
        ncol = 8; nrow = int(np.ceil(len(thumbs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2), dpi=120); fig.patch.set_facecolor("white")
        for ax, (c, prob, base, cam, dates) in zip(np.ravel(axes), thumbs):
            overlay(ax, base, cam, dates); ax.set_xticks([]); ax.set_yticks([])
            t = truth.get(c)
            if t and t[0] and t[1]:
                ax.axhline(dnum(date_to_day(t[1])), color="#1a1a1a", lw=1.2, ls="--")
            ax.set_title(f"{c}  {'PV' if t and t[0] else ('no PV' if t else '?')}  model {prob:.2f}", fontsize=8, loc="left")
        for ax in np.ravel(axes)[len(thumbs):]:
            ax.axis("off")
        fig.suptitle("Grad-CAM evidence for PV (red = strongest); dashed = labelled commissioning date", fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out_dir / f"sheet_{a.set}.png"); plt.close(fig)
        print("wrote", out_dir / f"sheet_{a.set}.png")


if __name__ == "__main__":
    main()
