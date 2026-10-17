"""Step 7a: the images the CNN sees, with the true and predicted commissioning dates drawn on them.
    python plot_images.py --eval artifacts/eval_humans.json          # the labelled customers + overview sheet
    python plot_images.py --set unlabelled --labels artifacts/labels_unlabelled.csv <customer_id> ...
Lines: dashed black = labelled date, solid green = predicted date. Title: label, model verdict and probability."""
import argparse, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm

from config import PLOTS_DIR, CACHE_DIR, LABELS_CSV
from data import load_cache, per_customer, read_labels, date_to_day, day_to_date
from images import relsmooth_image

OUT = PLOTS_DIR / "images"
DCMAP = plt.get_cmap("RdBu_r").copy(); DCMAP.set_bad("#e6e6e6")


def dnum(day):
    return mdates.date2num(day_to_date(np.asarray(day)).astype("datetime64[ns]").astype("O"))


def dense_relsmooth(d):
    days, Z = d["days"], relsmooth_image(d["X"], d["days"])
    grid = np.arange(days.min(), days.max() + 1)
    G = np.full((len(grid), 96), np.nan, np.float32); G[days - days.min()] = Z
    return grid, G


def draw(ax, cid, d, truth, pred, thumb=False):
    grid, G = dense_relsmooth(d); dates = dnum(grid)
    im = ax.imshow(G, aspect="auto", cmap=DCMAP, norm=TwoSlopeNorm(0.0, -1.0, 1.0), interpolation="nearest",
                   extent=[0, 24, dates[-1] + 1, dates[0]])
    ax.set_ylim(dates[-1] + 1, dates[0])
    t = truth.get(cid); p = pred.get(cid)
    if t and t["pv"] and t["date"]:
        ax.axhline(dnum(date_to_day(t["date"])), color="#1a1a1a", lw=1.2 if thumb else 1.6, ls="--")
    if p and p["pv"] and p["date"]:
        ax.axhline(dnum(date_to_day(p["date"])), color="#0b7a3b", lw=1.2 if thumb else 1.8)
    for s in ax.spines.values():
        s.set_visible(False)
    lab = ("PV" + (f" {t['cap']:.0f} kWp" if t.get("cap") else "") if t and t["pv"] else "no PV") if t else "unlabelled"
    mod = f"model {'PV' if p['pv'] else 'no PV'} ({p['prob']:.2f})" if p else ""
    if thumb:
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{cid}  {lab}\n{mod}", fontsize=8, loc="left")
    else:
        ticks = [g for g in grid if str(day_to_date(g))[8:10] == "01" and str(day_to_date(g))[5:7] in ("01", "04", "07", "10")]
        ax.set_yticks(dnum(np.array(ticks))); ax.set_yticklabels([str(day_to_date(g))[:7] for g in ticks])
        ax.set_xticks(range(0, 25, 3)); ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)]); ax.set_xlabel("hour of day (local time)")
        ax.grid(axis="y", color="#7a7a7a", lw=0.5, alpha=0.5)
        ax.set_title(f"{cid}\nlabel: {lab}" + (f"   |   {mod}" + (f", predicted date {p['date']}" if p and p['pv'] else "") if p else "")
                     + "\nrelsmooth: log2 ratio of each day to its own 06-09h / 17-20h level, 15-day x 3-slot median filter (blue = midday deficit)",
                     loc="left", fontsize=10)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--set", default="labelled", choices=["labelled", "unlabelled"])
    ap.add_argument("--eval", default=None, help="eval json from train.py: draws its predictions")
    ap.add_argument("--labels", default=None, help="label csv from label.py: draws its predictions")
    a = ap.parse_args()
    data = per_customer(load_cache(CACHE_DIR / f"{a.set}.npz"))
    truth = {}
    if a.set == "labelled":
        for _, r in read_labels(LABELS_CSV).iterrows():
            truth[r["customer_id"]] = {"pv": bool(r["pv_installed"]), "date": r["pv_installation_date"], "cap": r["pv_capacity_kwp"]}
    pred = {}
    if a.eval:
        for x in json.load(open(a.eval))["per_customer"]:
            pred[x["customer_id"]] = {"pv": bool(x["pv_installed_pred"]), "date": x["pv_installation_date_pred"] or "", "prob": float(x["pv_prob"])}
    if a.labels:
        for _, r in pd.read_csv(a.labels, sep=None, engine="python", dtype=str).iterrows():
            pred[r["customer_id"]] = {"pv": r["pv_installed"].strip().upper() == "TRUE", "date": (r["pv_installation_date"] or "").strip() if isinstance(r["pv_installation_date"], str) else "", "prob": float(r["pv_prob"])}
    ids = a.ids or (sorted(data, key=lambda s: int(s[3:])) if a.set == "labelled" else sorted(pred))
    out = OUT / a.set; out.mkdir(parents=True, exist_ok=True)
    for cid in ids:
        fig, ax = plt.subplots(figsize=(9, 11), dpi=130); fig.patch.set_facecolor("white")
        im = draw(ax, cid, data[cid], truth, pred)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("log2 ratio to the shoulder level (blue = less grid draw)"); cb.outline.set_visible(False)
        fig.tight_layout(); fig.savefig(out / f"{cid}.png"); plt.close(fig)
    print(f"wrote {len(ids)} images to {out}")
    if len(ids) > 1:
        ids_s = sorted(ids, key=lambda c: (not (truth.get(c, {}).get("pv", pred.get(c, {}).get("pv", False))), int(c[3:]) if c[3:].isdigit() else 0))
        ncol = 8; nrow = int(np.ceil(len(ids_s) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2), dpi=120); fig.patch.set_facecolor("white")
        for ax, cid in zip(np.ravel(axes), ids_s):
            draw(ax, cid, data[cid], truth, pred, thumb=True)
        for ax in np.ravel(axes)[len(ids_s):]:
            ax.axis("off")
        fig.suptitle("relsmooth images (blue = midday deficit = PV). Dashed black = labelled date, green = predicted date.", fontsize=10, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(out / f"overview_{a.set}.png"); plt.close(fig)
        print("wrote", out / f"overview_{a.set}.png")


if __name__ == "__main__":
    main()
