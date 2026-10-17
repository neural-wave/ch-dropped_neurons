"""Submission entry point for the CNN + day model pipeline.

    python inference.py path/to/test_set.csv

Reads the test consumption CSV (customer_id, energy_consumption_kwh, timestamp_local, timestamp_epoch; comma or
semicolon separated), does all preprocessing, loads the saved model from model/ (fallback artifacts/) and writes to the current working directory:
  predictions.csv                customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
                                 exactly one row per customer_id of the test CSV, TRUE/FALSE, YYYY-MM-DD,
                                 empty date and capacity when pv_installed = FALSE
  pv_extensions_predictions.csv  customer_id;extension_date;pv_capacity_after_extension_kwp (header only)
No network access, no weather / location / irradiation data; only the consumption values and their timestamps.
Runs on CPU; uses a GPU when one is available."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import load_timeseries, per_customer, date_to_day            # noqa: E402
from pipeline import PVPipeline                                          # noqa: E402
from features import capacity_features                                   # noqa: E402

SEP = ";"
MAIN_COLS = ["customer_id", "pv_installed", "pv_installation_date", "pv_capacity_kwp"]
EXT_COLS = ["customer_id", "extension_date", "pv_capacity_after_extension_kwp"]
MODEL_DIRS = [HERE / "model", HERE / "artifacts"]
PEAK_TO_KWP = 1.3     # fallback only: nominal kWp ~ 1.3 x the 95th percentile of the summer midday peak reduction (kW)
CHUNK = 100


def find_model():
    for d in MODEL_DIRS:
        if (d / "downstream.joblib").is_file() and (d / "encoder.pt").is_file():
            return d
    raise SystemExit(f"no model found: expected encoder.pt + downstream.joblib in one of {[str(d) for d in MODEL_DIRS]}")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python inference.py path/to/test_set.csv")
    test_csv = Path(sys.argv[1])
    if not test_csv.is_file():
        raise SystemExit(f"not a file: {test_csv}")
    t0 = time.time()
    model_dir = find_model()
    pipe = PVPipeline.load(model_dir)
    print(f"model: {model_dir}  (classifier={pipe.clf}, date from={pipe.date_from if pipe.clf == 'cnn' else 'day model'}, "
          f"threshold={pipe.threshold:.2f}, capacity model: {'yes' if pipe.cap_model is not None else 'fallback rule'})")

    raw = load_timeseries(test_csv, verbose=False, require_pod=False)      # every non-null id is kept
    all_ids = [str(c) for c in raw["all_ids"]]
    data = per_customer(raw)
    print(f"test file: {len(all_ids)} customers, {len(data)} with at least one usable day "
          f"({raw['stats']['days_kept']} days), read in {time.time() - t0:.0f}s")

    cids = sorted(data); preds = []
    for i in range(0, len(cids), CHUNK):
        batch = {c: data[c] for c in cids[i:i + CHUNK]}
        prep = pipe.prepare(batch)
        p = pipe.predict(prep)
        for _, r in p[p["pv_installed"]].iterrows():                      # capacity fallback when no regressor is stored
            if not np.isfinite(r["pv_capacity_kwp"]):
                c = r["customer_id"]
                f = capacity_features(prep[c]["X"], prep[c]["days"], date_to_day(r["pv_installation_date"]), prep[c]["scale"], prep[c]["sun"])
                p.loc[p["customer_id"] == c, "pv_capacity_kwp"] = round(max(PEAK_TO_KWP * f["peak_red_kw"], 1.0), 1)
        preds.append(p)
    pred = pd.concat(preds, ignore_index=True).set_index("customer_id") if preds else pd.DataFrame(columns=MAIN_COLS).set_index("customer_id")

    rows = []
    for cid in all_ids:
        if cid in pred.index:
            r = pred.loc[cid]; pv = bool(r["pv_installed"])
            date = str(r["pv_installation_date"])[:10] if pv else ""
            cap_s = f"{float(r['pv_capacity_kwp']):.1f}" if pv else ""
        else:                                                              # no usable day for this id
            pv, date, cap_s = False, "", ""
        rows.append({"customer_id": cid, "pv_installed": "TRUE" if pv else "FALSE",
                     "pv_installation_date": date, "pv_capacity_kwp": cap_s})
    out = pd.DataFrame(rows, columns=MAIN_COLS)
    out.to_csv("predictions.csv", sep=SEP, index=False, encoding="utf-8")
    pd.DataFrame(columns=EXT_COLS).to_csv("pv_extensions_predictions.csv", sep=SEP, index=False, encoding="utf-8")
    n_pv = int((out["pv_installed"] == "TRUE").sum())
    print(f"wrote predictions.csv ({len(out)} rows, {n_pv} PV) and pv_extensions_predictions.csv (header only) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
