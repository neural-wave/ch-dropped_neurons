"""Score a prediction / label file against the labelled customers and show the per-customer date table.

    python score_labels.py artifacts/labels_40_run1.csv [--model model]

--model prints the saved model's settings (classifier, date source, threshold) from downstream.joblib."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import LABELS_CSV
from data import read_labels
from scoring import main_score, format_score

pd.set_option("display.width", 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    if a.model:
        s = joblib.load(Path(a.model) / "downstream.joblib")
        print(f"model {a.model}: classifier={s.get('clf')}  date_from={s.get('date_from', 'gbm (default)')}  "
              f"date_signal={s.get('date_signal')}  threshold={s.get('threshold')}  capacity model={'yes' if s.get('cap_model') is not None else 'none'}")
    pred = read_labels(a.labels); truth = read_labels(LABELS_CSV)
    r = main_score(pred, truth)
    print(format_score(r))
    m = truth.merge(pred, on="customer_id", suffixes=("_true", "_pred"))
    m = m[m["pv_installed_true"]].copy()
    err = (pd.to_datetime(m["pv_installation_date_pred"], errors="coerce") - pd.to_datetime(m["pv_installation_date_true"])).dt.days
    m["date_err_days"] = err
    cols = ["customer_id", "pv_installed_pred", "pv_installation_date_true", "pv_installation_date_pred", "date_err_days"]
    if "pv_prob" in m:
        cols.insert(2, "pv_prob")
    print(m[cols].to_string(index=False))
    ok = err.dropna().abs()
    print(f"dated {len(ok)}: within 1 day {(ok <= 1).sum()}, within 7 days {(ok <= 7).sum()}, within 30 days {(ok <= 30).sum()}, "
          f"median |err| {ok.median():.0f} d")


if __name__ == "__main__":
    main()
