"""Step 6: label customers with a saved model.
    python label.py --model artifacts/model_humans [--set unlabelled] [--exclude artifacts/human_labels.csv] [--out FILE]
Output (same layout as the labels file + extras): customer_id; pv_installed; pv_installation_date; pv_capacity_kwp;
pv_prob; label_confidence (= max(p, 1-p): probability that the written label is right). Aggregates only are printed."""
import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

from config import CACHE_DIR, ARTIFACT_DIR, OUTPUT_SEP
from data import load_cache, per_customer, read_labels
from pipeline import CNNPipeline

COLS = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp", "pv_prob", "label_confidence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ARTIFACT_DIR / "model_humans"))
    ap.add_argument("--set", default="unlabelled", choices=["labelled", "unlabelled"])
    ap.add_argument("--exclude", nargs="*", default=[], help="label files whose customers are skipped")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = args.out or str(ARTIFACT_DIR / f"labels_{args.set}.csv")
    t0 = time.time()
    pipe = CNNPipeline.load(args.model)
    data = per_customer(load_cache(CACHE_DIR / f"{args.set}.npz"))
    cids = sorted(data)
    if args.exclude:
        skip = set(pd.concat([read_labels(p) for p in args.exclude])["customer_id"])
        cids = [c for c in cids if c not in skip]; print(f"skipping {len(skip)} customers listed in {args.exclude}")
    if args.limit:
        cids = cids[:args.limit]
    print(f"model {args.model} (image={pipe.cnn.image}, calendar={pipe.cnn.calendar}, device {pipe.cnn.device})  {len(cids)} customers")
    pred = pipe.predict(data, cids)
    pred["label_confidence"] = np.maximum(pred["pv_prob"], 1 - pred["pv_prob"]).round(3)
    o = pred[COLS].copy()
    o["pv_installed"] = np.where(o["pv_installed"], "TRUE", "FALSE")
    o["pv_capacity_kwp"] = [f"{v:.1f}" if np.isfinite(v) else "" for v in o["pv_capacity_kwp"]]
    Path(out).parent.mkdir(parents=True, exist_ok=True); o.to_csv(out, sep=OUTPUT_SEP, index=False)
    pv = pred["pv_installed"].to_numpy(bool); lc = pred["label_confidence"].to_numpy()
    print(f"wrote {out}: {len(o)} customers, PV {pv.sum()} ({pv.mean():.1%}), no PV {(~pv).sum()}")
    print(f"label_confidence mean {lc.mean():.3f}   >= 0.9: {(lc >= 0.9).sum()}   >= 0.8: {(lc >= 0.8).sum()}   < 0.6: {(lc < 0.6).sum()}")
    hist, edges = np.histogram(pred["pv_prob"], bins=np.linspace(0, 1, 11))
    print("pv_prob histogram:", "  ".join(f"[{a:.1f}-{b:.1f}) {h}" for a, b, h in zip(edges[:-1], edges[1:], hist)))
    if pv.any():
        years = pd.to_datetime(pred.loc[pv, "pv_installation_date"]).dt.year.value_counts().sort_index()
        print("predicted commissioning year:", "  ".join(f"{y}: {c}" for y, c in years.items()))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
