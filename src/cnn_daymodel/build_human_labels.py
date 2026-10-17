"""Merge the manual labelling files into one training label file (same columns as pv_labels_dev.csv).

    python build_human_labels.py humanA.csv humanB.csv [--min-conf 0] [--out artifacts/human_labels.csv]

Keeps a customer only when every human who labelled it agrees on PV yes/no and every one of their confidences
is >= --min-conf. The 40 labelled development customers are dropped (they are the test set). Date = the mean of
the humans' dates. Capacity is unknown for human labels and left empty. Only aggregate counts are printed."""
import argparse
import numpy as np
import pandas as pd

from config import ARTIFACT_DIR, LABELS_CSV, OUTPUT_SEP
from data import read_labels


def read_any(path):
    d = pd.read_csv(path, sep=None, engine="python", dtype=str)
    d.columns = [c.strip().lower() for c in d.columns]
    d["customer_id"] = d["customer_id"].str.strip()
    d["pv"] = d["pv_installed"].str.strip().str.upper().isin(["TRUE", "1", "YES"])
    d["date"] = pd.to_datetime(d["pv_installation_date"].fillna("").str.strip().str[:10], errors="coerce")
    d["conf"] = pd.to_numeric(d["confidence"], errors="coerce") if "confidence" in d else 100.0
    return d.drop_duplicates("customer_id")[["customer_id", "pv", "date", "conf"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("humans", nargs="+")
    ap.add_argument("--min-conf", type=float, default=0.0)
    ap.add_argument("--out", default=str(ARTIFACT_DIR / "human_labels.csv"))
    a = ap.parse_args()
    allh = pd.concat([read_any(p) for p in a.humans], ignore_index=True)
    dev = set(read_labels(LABELS_CSV)["customer_id"])
    n_dev = allh["customer_id"].isin(dev).sum()
    allh = allh[~allh["customer_id"].isin(dev)]
    g = allh.groupby("customer_id")
    agg = pd.DataFrame({"n_humans": g.size(), "pv_mean": g["pv"].mean(), "conf_min": g["conf"].min(), "date": g["date"].mean()})
    agree = agg["pv_mean"].isin([0.0, 1.0]); conf_ok = agg["conf_min"] >= a.min_conf
    keep = agg[agree & conf_ok].copy(); keep["pv"] = keep["pv_mean"] == 1.0
    out = pd.DataFrame({
        "customer_id": keep.index,
        "pv_installed": np.where(keep["pv"], "TRUE", "FALSE"),
        "pv_installation_date": [d.strftime("%Y-%m-%d") if (pv and pd.notna(d)) else "" for pv, d in zip(keep["pv"], keep["date"])],
        "pv_capacity_kwp": "",
        "n_humans": keep["n_humans"].values, "conf_min": keep["conf_min"].values,
    })
    out.to_csv(a.out, sep=OUTPUT_SEP, index=False)
    print(f"input rows {len(allh) + n_dev} ({n_dev} belong to the 40 labelled dev customers -> dropped)")
    print(f"unlabelled customers with a human label: {len(agg)}   humans contradict: {(~agree).sum()}   below min-conf {a.min_conf:g}: {(agree & ~conf_ok).sum()}")
    print(f"kept {len(out)}: {int(keep['pv'].sum())} PV ({int((keep['pv'] & keep['date'].notna()).sum())} with a date), {int((~keep['pv']).sum())} no PV   -> {a.out}")


if __name__ == "__main__":
    main()
