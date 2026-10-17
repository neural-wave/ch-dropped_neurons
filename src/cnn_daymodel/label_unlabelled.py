"""Step 4: pseudo-label the unlabelled customers with the fitted pipeline.

For every customer in cache/unlabelled.npz (1147 POD ids) this writes whether PV is present,
the estimated commissioning date and the estimated kWp, plus the model probability, using the
models saved in artifacts/ by train_downstream.py.  The encoder runs on the GPU when one exists.

    python label_unlabelled.py                      # -> artifacts/unlabelled_pseudo_labels.csv
    python label_unlabelled.py --limit 50           # quick smoke test on 50 customers

    python label_unlabelled.py --artifacts model \
        --exclude artifacts/human_labels.csv --out artifacts/unlabelled_labels_cnn.csv
                                                    # the human-trained CNN on every customer the humans did not label

Output columns (same layout as pv_labels_dev.csv, so data.read_labels() can read it back):
    customer_id; pv_installed (TRUE/FALSE); pv_installation_date; pv_capacity_kwp; pv_prob; label_confidence; confident
pv_prob          = model probability that the customer has PV
label_confidence = probability that the written pv_installed label is right = max(pv_prob, 1 - pv_prob)
confident        = TRUE when |pv_prob - threshold| >= --margin (default 0.2)
Only aggregate statistics are printed.
"""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

from config import CACHE_DIR, ARTIFACT_DIR, OUTPUT_SEP
from data import load_cache, per_customer
from pipeline import PVPipeline
from ssl_model import get_device

COLS = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp", "pv_prob", "label_confidence", "confident"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE_DIR / "unlabelled.npz"), help="daily profiles from prepare_data.py")
    ap.add_argument("--artifacts", default=str(ARTIFACT_DIR), help="folder with encoder.pt + downstream.joblib")
    ap.add_argument("--out", default=str(ARTIFACT_DIR / "unlabelled_pseudo_labels.csv"))
    ap.add_argument("--margin", type=float, default=0.2, help="|prob - threshold| needed for confident=TRUE")
    ap.add_argument("--chunk", type=int, default=100, help="customers per embed/predict chunk (progress + memory)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N customers (smoke test)")
    ap.add_argument("--exclude", nargs="*", default=[], help="label files whose customers are skipped (e.g. the human labels)")
    args = ap.parse_args()

    t0 = time.time()
    pipe = PVPipeline.load(Path(args.artifacts))
    print(f"device: {get_device()}   classifier: {pipe.clf}   threshold: {pipe.threshold:.3f}")
    data = per_customer(load_cache(args.cache))
    cids = sorted(data)
    if args.exclude:
        from data import read_labels
        skip = set(pd.concat([read_labels(p) for p in args.exclude])["customer_id"])
        n0 = len(cids); cids = [c for c in cids if c not in skip]
        print(f"skipping {n0 - len(cids)} customers already labelled in {args.exclude}")
    if args.limit:
        cids = cids[:args.limit]
    print(f"{len(cids)} customers, {sum(len(data[c]['days']) for c in cids):,} customer-days")

    preds = []
    for i in range(0, len(cids), args.chunk):
        batch = {c: data[c] for c in cids[i:i + args.chunk]}
        preds.append(pipe.predict(pipe.prepare(batch)))
        print(f"  {min(i + args.chunk, len(cids))}/{len(cids)} customers  ({time.time() - t0:.0f}s)", flush=True)
    pred = pd.concat(preds, ignore_index=True)
    pred["confident"] = (pred["pv_prob"] - pipe.threshold).abs() >= args.margin
    pred["label_confidence"] = np.maximum(pred["pv_prob"], 1 - pred["pv_prob"]).round(3)

    out = pred[COLS].copy()
    out["pv_installed"] = np.where(out["pv_installed"], "TRUE", "FALSE")
    out["confident"] = np.where(out["confident"], "TRUE", "FALSE")
    out["pv_capacity_kwp"] = [f"{v:.1f}" if np.isfinite(v) else "" for v in out["pv_capacity_kwp"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, sep=OUTPUT_SEP, index=False, encoding="utf-8")

    # ---- aggregate summary only (no per-customer values are printed) ----
    pv = pred["pv_installed"].to_numpy(bool); conf = pred["confident"].to_numpy(bool)
    n = len(pred)
    print(f"\nwrote {args.out}")
    print(f"customers {n}   predicted PV {pv.sum()} ({pv.mean():.1%})   no PV {(~pv).sum()}")
    lc = pred["label_confidence"].to_numpy()
    print(f"label_confidence: mean {lc.mean():.3f}   >= 0.9: {(lc >= 0.9).sum()}   >= 0.8: {(lc >= 0.8).sum()}   < 0.6: {(lc < 0.6).sum()}")
    print(f"confident (margin {args.margin}): {conf.sum()} total = {int((conf & pv).sum())} PV + {int((conf & ~pv).sum())} no-PV")
    hist, edges = np.histogram(pred["pv_prob"], bins=np.linspace(0, 1, 11))
    print("pv_prob histogram:", "  ".join(f"[{a:.1f}-{b:.1f}) {h}" for a, b, h in zip(edges[:-1], edges[1:], hist)))
    if pv.any():
        years = pd.to_datetime(pred.loc[pv, "pv_installation_date"]).dt.year.value_counts().sort_index()
        print("predicted commissioning year:", "  ".join(f"{y}: {c}" for y, c in years.items()))
        cap = pred.loc[pv, "pv_capacity_kwp"].to_numpy(float); cap = cap[np.isfinite(cap)]
        if len(cap):
            q = np.percentile(cap, [10, 50, 90])
            print(f"predicted kWp: p10 {q[0]:.1f}  median {q[1]:.1f}  p90 {q[2]:.1f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
