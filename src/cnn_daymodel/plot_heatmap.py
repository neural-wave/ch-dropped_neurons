"""Date x hour-of-day heatmaps of 15-minute grid withdrawal.

  python plot_heatmap.py            # one PV customer whose commissioning date is mid-series
  python plot_heatmap.py POD13      # one given customer
  python plot_heatmap.py --all      # every labelled customer -> plots/heatmap_<id>_<PV info>.png + plots/overview_all40.png
  add --pred <eval json>          # draw the model's predicted commissioning date as a second line
                                  # (e.g. --pred artifacts/eval_loocv.json: out-of-fold predictions) -> plots/dates/
  add --minmax to any form        # per-customer 0-1 scaling (1st-99th percentile) -> plots/minmax/
  add --log to any form           # log colour scale from the customer's 2nd to 98th percentile -> plots/log/
                                  # (linear: white = 0 kW; log: white = the customer's own low level, so a
                                  #  flat high load such as 800-1000 kW still shows its structure)
  add --relative to any form      # each day as log2 ratio to its own shoulder level (06-09h and 17-20h) -> plots/relative/
                                  # removes the load level: a 15% midday deficit looks the same on 1 MW and on 10 kW
  add --relsmooth to any form     # --relative followed by a 15-day x 3-slot median filter -> plots/relsmooth/
                                  # removes day stripes (holidays, load changes), keeps the sun-shaped PV blob
  add --yoy to any form           # each cell minus the median of the same slot one year earlier (+-10 days) -> plots/yoy/
                                  # removes seasonality and stable lunch dips; only what CHANGED survives; first year is grey
  python plot_heatmap.py --compare   # plots/compare_all40.png: every customer in the linear, relative, relsmooth and yoy views
File names carry the label, e.g. heatmap_POD13_PV_174kWp_2024-02-05_ext407kWp.png or heatmap_POD48_noPV.png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, LogNorm, Normalize, TwoSlopeNorm
from scipy.ndimage import median_filter
import matplotlib.dates as mdates

import pandas as pd
from config import CACHE_DIR, LABELS_CSV, EXT_LABELS_CSV, PROJECT_DIR
from data import load_cache, per_customer, read_labels, date_to_day, day_to_date
from ssl_model import minmax_anchors, MINMAX_PCT

OUT = PROJECT_DIR / "plots"
CMAP = LinearSegmentedColormap.from_list("heat", ["#ffffff", "#fff3b0", "#ffd23f", "#ff9f1c", "#f0522b",
                                                  "#c81d25", "#7a0c14"])
CMAP.set_bad("#e6e6e6")          # missing days
GAMMA = 0.6                      # < 1 stretches the low end so small changes stand out
LOG = "--log" in sys.argv
MINMAX = "--minmax" in sys.argv          # per-customer 0-1 scaling between its 1st and 99th percentile
MODE = ("minmax" if MINMAX else "log" if LOG else "relative" if "--relative" in sys.argv
        else "relsmooth" if "--relsmooth" in sys.argv else "yoy" if "--yoy" in sys.argv else "linear")
SUBDIR = {"minmax": "minmax", "log": "log", "relative": "relative", "relsmooth": "relsmooth", "yoy": "yoy", "linear": ""}[MODE]
DIVERGING = {"relative", "relsmooth", "yoy"}
DCMAP = plt.get_cmap("RdBu_r").copy(); DCMAP.set_bad("#e6e6e6")     # blue = less grid draw than the reference (PV deficit)
SHOULDER = np.r_[24:36, 68:80]           # 06:00-09:00 and 17:00-20:00
MID_SLOTS = np.r_[44:56]                 # 11:00-14:00
DIV_LIM = 1.0                            # colour range: +-1 = factor 2 above / below the reference
SCALE_TXT = {"minmax": f"0-1 scaled between the customer's {MINMAX_PCT[0]}th and {MINMAX_PCT[1]}th percentile",
             "log": "log colour scale, 2nd-98th percentile", "linear": "linear colour scale from 0",
             "relative": "log2 ratio of each day to its own 06-09h / 17-20h shoulder level (blue = below)",
             "relsmooth": "day / shoulder log2 ratio, 15-day x 3-slot median filter (blue = below)",
             "yoy": "log2 ratio to the same slot one year earlier (blue = less than last year; grey = no reference)"}
OVERVIEW_TXT = {"minmax": f"Each customer min-max scaled to 0-1 between its {MINMAX_PCT[0]}th and {MINMAX_PCT[1]}th percentile. ",
                "log": "Log colour scale per customer, from its 2nd to its 98th percentile. ",
                "linear": "Colour scale is per customer (white = 0, dark red = that customer's 95th percentile). ",
                "relative": "Each day as log2 ratio to its own shoulder level (06-09h, 17-20h): blue = less grid draw than the shoulders, red = more; +-1 = factor 2. ",
                "relsmooth": "Day / shoulder log2 ratio after a 15-day x 3-slot median filter: blue = midday deficit, day stripes removed; +-1 = factor 2. ",
                "yoy": "Each cell as log2 ratio to the same slot one year earlier (median of +-10 days): blue = less than last year, grey = no reference year; +-1 = factor 2. "}
if "--pred" in sys.argv:
    SUBDIR = "dates"
if "--outdir" in sys.argv:
    SUBDIR = sys.argv[sys.argv.index("--outdir") + 1]


def grid_kw(d):
    """dense (days x 96) matrix of average kW, NaN rows for missing days"""
    X, days = d["X"], d["days"]
    grid_days = np.arange(days.min(), days.max() + 1)
    Z = np.full((len(grid_days), 96), np.nan, np.float32)
    Z[days - days.min()] = X
    return grid_days, Z * 4


def _dnum(day):
    return mdates.date2num(day_to_date(np.asarray(day)).astype("datetime64[ns]").astype("O"))


def read_extensions():
    """customer_id -> list of (extension_date, capacity_after_kwp); tolerant to mixed , / ; separators"""
    try:
        df = pd.read_csv(EXT_LABELS_CSV, sep=r"[;,]", engine="python", dtype=str)
    except Exception:
        return {}
    df.columns = [c.strip() for c in df.columns]; out = {}
    for _, r in df.dropna().iterrows():
        out.setdefault(r["customer_id"].strip(), []).append((r["extension_date"].strip(), float(r["pv_capacity_after_extension_kwp"])))
    return out


EXT = {}
PRED = {}          # customer_id -> predicted date string (from --pred <eval json>)
PROB = {}          # customer_id -> predicted PV probability (from --pred <eval json>)


def read_predictions(path):
    import json
    r = json.load(open(path)); out = {}
    for x in r["per_customer"]:
        if x.get("pv_prob") is not None:
            PROB[x["customer_id"]] = float(x["pv_prob"])
        d = x.get("pv_installation_date_pred")
        if x.get("pv_installed_pred") and d and str(d) not in ("", "nan", "None"):
            out[x["customer_id"]] = str(d)[:10]
    return out


def enhance(kw, mode):
    """(days x 96) kW -> log2 ratio image for the diverging modes. NaN where there is no reference."""
    lk = np.log2(np.maximum(kw, 1e-3))
    if mode in ("relative", "relsmooth"):
        Z = lk - np.nanmean(lk[:, SHOULDER], 1, keepdims=True)
        if mode == "relsmooth":
            Z = median_filter(np.where(np.isfinite(Z), Z, 0.0), size=(15, 3))
            Z[~np.isfinite(kw).all(1)] = np.nan
        return Z.astype(np.float32)
    if mode == "yoy":
        Z = np.full_like(lk, np.nan)
        for day in range(365, lk.shape[0]):
            ref = lk[max(0, day - 375):day - 355]
            if np.isfinite(ref).all(1).sum() >= 5:
                Z[day] = lk[day] - np.nanmedian(ref, 0)
        return Z.astype(np.float32)
    raise ValueError(mode)


def draw(ax, cid, d, r, thumb=False, mode=None):
    mode = mode or MODE
    grid_days, kw = grid_kw(d)
    dates = _dnum(grid_days)
    cmap = CMAP
    if mode in DIVERGING:
        kw = enhance(kw, mode)
        norm = TwoSlopeNorm(0.0, -DIV_LIM, DIV_LIM); cmap = DCMAP
    elif mode == "minmax":
        lo, hi = minmax_anchors(kw[np.isfinite(kw).all(1)])
        kw = np.clip((kw - lo) / (hi - lo), 0, 1)
        norm = Normalize(vmin=0, vmax=1)
    elif mode == "log":
        vmax = max(float(np.nanpercentile(kw, 98)), 1e-2)
        vmin = max(float(np.nanpercentile(kw, 2)), 0.01 * vmax, 1e-3)
        kw = np.where(np.isfinite(kw), np.maximum(kw, vmin), np.nan)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmax = max(float(np.nanpercentile(kw, 95)), 1e-3)
        norm = PowerNorm(GAMMA, vmin=0, vmax=vmax)
    im = ax.imshow(kw, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest", extent=[0, 24, dates[-1] + 1, dates[0]])
    ax.set_ylim(dates[-1] + 1, dates[0])
    pv = bool(r.pv_installed) and bool(r.pv_installation_date)
    if pv:
        y = _dnum(date_to_day(r.pv_installation_date))
        ax.axhline(y, color="#1a1a1a", lw=1.2 if thumb else 1.5, ls="--")
    pred = PRED.get(cid)
    if pred:
        yp = _dnum(date_to_day(pred))
        ax.axhline(yp, color="#0b7a3b", lw=1.2 if thumb else 1.8, ls="-")
        if not thumb:
            err = (date_to_day(pred) - date_to_day(r.pv_installation_date)) if pv else None
            txt = f"predicted {pred}" + (f"  ({err:+d} days)" if err is not None else "  (label: no PV)")
            ax.text(23.7, yp, txt, color="#0b7a3b", fontsize=9, ha="right", va="top", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    exts = EXT.get(cid, [])
    for ext_date, ext_kwp in exts:
        ye = _dnum(date_to_day(ext_date))
        ax.axhline(ye, color="#1b4fd8", lw=1.2 if thumb else 1.5, ls=":")
        if not thumb:
            ax.text(23.7, ye, f"extension {ext_date}: {ext_kwp:.0f} kWp total", color="#1b4fd8", fontsize=9,
                    ha="right", va="bottom", fontweight="bold",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
    for s in ax.spines.values():
        s.set_visible(False)
    if thumb:
        ax.set_xticks([]); ax.set_yticks([])
        ttl = f"{cid}  " + (f"PV {r.pv_capacity_kwp:.0f} kWp" if pv else "no PV")
        if exts:
            ttl += f" -> {exts[-1][1]:.0f}"
        if pred and pv:
            ttl += f"  err {date_to_day(pred) - date_to_day(r.pv_installation_date):+d}d"
        elif pred:
            ttl += "  FP"
        elif pv and PRED:
            ttl += "  missed"
        if cid in PROB:
            ttl += f"\nmodel p={PROB[cid]:.2f} -> {'PV' if PROB[cid] >= 0.5 else 'no PV'}"
        ax.set_title(ttl, fontsize=8, loc="left")
        return im
    tick_days = [g for g in grid_days if str(day_to_date(g))[8:10] == "01" and str(day_to_date(g))[5:7] in ("01", "04", "07", "10")]
    ax.set_yticks(_dnum(np.array(tick_days)))
    ax.set_yticklabels([np.datetime_as_string(day_to_date(g), unit="D")[:7] for g in tick_days])
    ax.grid(axis="y", color="#7a7a7a", lw=0.5, alpha=0.5)
    ax.set_xticks(range(0, 25, 3)); ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xlabel("Hour of day (local time)")
    ax.tick_params(colors="#4b4b4b", labelsize=9)
    if pv:
        ax.text(0.3, y, f"PV commissioned {r.pv_installation_date}", color="#1a1a1a", fontsize=9, ha="left",
                va="bottom", fontweight="bold", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
        sub = f"PV customer, {r.pv_capacity_kwp:.0f} kWp nominal"
        if exts:
            sub += "; extended to " + ", ".join(f"{k:.0f} kWp on {dt}" for dt, k in exts)
    else:
        sub = "no PV"
    model_txt = ""
    if cid in PROB:
        model_txt = f"\nmodel: {'PV' if PROB[cid] >= 0.5 else 'no PV'} (probability {PROB[cid]:.2f})" + (f", predicted date {pred}" if pred else "")
    scale_txt = SCALE_TXT[mode]
    ax.set_title(f"{cid}: grid withdrawal by date and hour of day\nlabel: {sub}{model_txt}\n{scale_txt}",
                 loc="left", fontsize=10, color="#1f1f1f")
    return im


def file_name(cid, r):
    """e.g. heatmap_POD13_PV_174kWp_2024-02-05_ext407kWp.png  or  heatmap_POD48_noPV.png"""
    if bool(r.pv_installed) and bool(r.pv_installation_date):
        name = f"heatmap_{cid}_PV_{r.pv_capacity_kwp:.0f}kWp_{r.pv_installation_date}"
        for _, k in EXT.get(cid, []):
            name += f"_ext{k:.0f}kWp"
        return name + ".png"
    return f"heatmap_{cid}_noPV.png"


def plot_one(cid, d, r):
    fig, ax = plt.subplots(figsize=(9, 11), dpi=150); fig.patch.set_facecolor("white")
    im = draw(ax, cid, d, r)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Normalised withdrawal, 0 = customer low, 1 = customer high" if MINMAX else
                 "log2 ratio to the reference (blue = less grid draw)" if MODE in DIVERGING else
                 "Average power drawn from the grid (kW)" + (", log scale" if LOG else "")); cb.outline.set_visible(False)
    out_dir = OUT / SUBDIR if SUBDIR else OUT; out_dir.mkdir(exist_ok=True)
    fig.tight_layout(); out = out_dir / file_name(cid, r); fig.savefig(out); plt.close(fig)
    return out


def plot_overview(data, labels):
    cids = sorted(data, key=lambda c: (not bool(labels.loc[c].pv_installed), c))   # PV customers first
    ncol = 8; nrow = int(np.ceil(len(cids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 3.2), dpi=120)
    fig.patch.set_facecolor("white")
    for ax, cid in zip(axes.flat, cids):
        draw(ax, cid, data[cid], labels.loc[cid], thumb=True)
    for ax in list(axes.flat)[len(cids):]:
        ax.axis("off")
    scale_txt = OVERVIEW_TXT[MODE]
    fig.suptitle("All 40 labelled customers: grid withdrawal by date (down) and hour of day (across). " + scale_txt
                 + "\nDashed black line = labelled PV commissioning date; dotted blue line = labelled capacity extension"
                 + ("; solid green line = predicted commissioning date." if PRED else "."),
                 fontsize=10, x=0.01, ha="left")
    out_dir = OUT / SUBDIR if SUBDIR else OUT; out_dir.mkdir(exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); out = out_dir / "overview_all40.png"; fig.savefig(out); plt.close(fig)
    return out


COMPARE_MODES = ["linear", "relative", "relsmooth", "yoy"]


def plot_compare(data, labels, per_row=5):
    """plots/compare_all40.png: one block of four thumbnails (linear, relative, relsmooth, yoy) per customer"""
    cids = sorted(data, key=lambda c: (not bool(labels.loc[c].pv_installed), c))   # PV customers first
    nview = len(COMPARE_MODES); ncol = per_row * nview; nrow = int(np.ceil(len(cids) / per_row))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.7, nrow * 2.6), dpi=110)
    fig.patch.set_facecolor("white")
    for k, cid in enumerate(cids):
        row, col0 = divmod(k, per_row)
        for j, mode in enumerate(COMPARE_MODES):
            ax = axes[row, col0 * nview + j]
            draw(ax, cid, data[cid], labels.loc[cid], thumb=True, mode=mode)
            if j:
                ax.set_title(mode, fontsize=7, loc="left", color="#555555")
    for k in range(len(cids), nrow * per_row):
        row, col0 = divmod(k, per_row)
        for j in range(nview):
            axes[row, col0 * nview + j].axis("off")
    fig.suptitle("All 40 labelled customers, PV first. Per customer, left to right: linear kW (white = 0, dark red = 95th percentile); "
                 "relative = each day as log2 ratio to its own 06-09h/17-20h shoulder level;\nrelsmooth = the same after a 15-day x 3-slot "
                 "median filter; yoy = log2 ratio to the same slot one year earlier (grey = no reference year). "
                 "Diverging views: blue = less grid draw than the reference, red = more, +-1 = factor 2. "
                 "Dashed black line = labelled PV commissioning date; dotted blue = capacity extension.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.965)); out = OUT / "compare_all40.png"; fig.savefig(out); plt.close(fig)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    labels = read_labels(LABELS_CSV).set_index("customer_id")
    data = per_customer(load_cache(CACHE_DIR / "labelled.npz"))
    EXT.update(read_extensions())
    if "--pred" in sys.argv:
        PRED.update(read_predictions(sys.argv[sys.argv.index("--pred") + 1]))
    if "--compare" in sys.argv:
        print("saved", plot_compare(data, labels))
        return
    if "--all" in sys.argv:
        for cid in sorted(data):
            plot_one(cid, data[cid], labels.loc[cid])
        print(f"saved {len(data)} heatmaps to {OUT / SUBDIR if SUBDIR else OUT}")
        print("saved", plot_overview(data, labels))
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--") and (sys.argv[sys.argv.index(a) - 1] not in ("--pred", "--outdir"))]
    if args:
        cid = args[0]
    else:
        best = None
        for c, d in data.items():
            r = labels.loc[c]
            if r.pv_installed and r.pv_installation_date:
                frac = (date_to_day(r.pv_installation_date) - d["days"].min()) / max(1, d["days"].max() - d["days"].min())
                if 0.2 < frac < 0.8 and (best is None or abs(frac - 0.5) < best[0]):
                    best = (abs(frac - 0.5), c)
        cid = best[1]
    print("saved", plot_one(cid, data[cid], labels.loc[cid]))


if __name__ == "__main__":
    main()
