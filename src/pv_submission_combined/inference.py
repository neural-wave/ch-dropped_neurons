"""Usage: python inference.py path/to/test_set.csv [out_dir]  -> predictions.csv, pv_extensions_predictions.csv (default: CWD).

Rule detector + bell gate, encoder-score date, kWp = 50/50 log blend of the C9v2 kWp (deficit ridge + year-over-year) and
the kWp network, GatedGuard extensions. Any per-customer failure falls back to the classical date / the median kWp / the
C9v2 kWp alone / no extension; a global failure writes everything as FALSE.
"""
import os
import pickle
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

from pvsteps import (capacity, capacity_yoy, date, day_features, day_score, detect_gate, encoder, extensions_gated, kwp_net,
                     load, write)

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices on sparse customers are handled downstream (capacity_yoy relies on it)
MODEL = HERE / "model"
COLS = ["customer_id", "pv_installation_date", "pv_capacity_kwp"]
NET_WEIGHT = 0.5  # share of the kWp network in the log-space kWp blend


def warn(msg):
    print(f"[inference] WARNING: {msg}", file=sys.stderr)


def scored_days(state, df):
    """Day encoder -> day features -> day score, per customer; a customer that fails is skipped (its date comes from the rule detector)."""
    tabs, bad = {}, 0
    try:
        enc = encoder.load_encoder(MODEL / "encoder_shipped.pt")
        days = load.ail_days(df)
    except Exception as e:
        warn(f"encoder path failed ({type(e).__name__}); all dates use the classical date")
        return tabs
    for cid, d in days.items():
        try:
            scale = encoder.customer_scale(d["X"])
            tab = day_features.day_table(d["X"], d["days"], encoder.encode_days(enc, d["X"], scale), scale)
            tabs[cid] = day_score.score_days(state["ail"]["day_model"], tab)
        except Exception:
            bad += 1
    if bad:
        warn(f"{bad} customer(s) without encoder date, using the classical date")
    return tabs


def base_kwp(state, dms, cid):
    med = state["live"]["det"]["kwp"]
    try:
        k = float(capacity.kwp(state["live"]["cap"], dms[cid]))
    except Exception:
        k = float(med)
    return k if np.isfinite(k) and k > 0 else float(med)


def blend_kwp(df, pred):
    """C9v2 kWp blended with the network kWp at the same date; customers the network cannot score keep the C9v2 kWp."""
    try:
        net = kwp_net.kwp(MODEL / "kwp_net.pt", df, dict(zip(pred["customer_id"], pred["pv_installation_date"])))
    except Exception as e:
        warn(f"kWp network failed ({type(e).__name__}); C9v2 kWp used for every customer")
        return pred["pv_capacity_kwp"]
    missing = int((~pred["customer_id"].isin(list(net))).sum())
    if missing:
        warn(f"{missing} customer(s) without network kWp, C9v2 kWp used")
    return [float(np.exp((1 - NET_WEIGHT) * np.log(k) + NET_WEIGHT * np.log(net[c]))) if c in net else k
            for c, k in zip(pred["customer_id"], pred["pv_capacity_kwp"])]


def predict(state, df):
    dms, tabs = load.daily_matrix(df), scored_days(state, df)
    found = detect_gate.rule_detect(state["live"]["det"], dms)
    off = detect_gate.bell_gate(dms, found)
    rows, base = [], {}
    for c in sorted(found["pv"]):
        base[c] = base_kwp(state, dms, c)
        if c in off:
            continue
        try:
            d = date.ssl_date(tabs[c]) if c in tabs else found["date"][c]
        except Exception:
            d = found["date"][c]
        try:
            k = capacity_yoy.kwp_at(state["yoy"], dms[c], d, base[c])
        except Exception:
            k = base[c]
        if not (np.isfinite(k) and k > 0):
            k = base[c]
        rows.append((c, pd.Timestamp(d), k))
    pred = pd.DataFrame(rows, columns=COLS)
    pred["pv_capacity_kwp"] = blend_kwp(df, pred)
    try:
        ext = extensions_gated.gated_extensions(dms, base, off)
    except Exception as e:
        warn(f"extension prediction failed ({type(e).__name__}); header-only extensions file")
        ext = pd.DataFrame(columns=extensions_gated.COLS)
    return pred, ext


def main():
    if len(sys.argv) not in (2, 3) or not os.path.isfile(sys.argv[1]):
        sys.exit("usage: python inference.py path/to/test_set.csv [out_dir]")
    out_dir = sys.argv[2] if len(sys.argv) == 3 else "."
    ids = load.customer_ids(sys.argv[1])
    pred, ext = pd.DataFrame(columns=COLS), pd.DataFrame(columns=extensions_gated.COLS)
    try:
        with open(MODEL / "state.pkl", "rb") as f:
            state = pickle.load(f)
        pred, ext = predict(state, load.read_series(sys.argv[1]))
    except Exception as e:
        warn(f"loading/prediction failed ({type(e).__name__}); everything falls back to FALSE")
    write.write_outputs(out_dir, ids, pred, ext)
    print(f"[inference] {len(ids)} customers, {len(pred)} PV, {len(ext)} extensions")


if __name__ == "__main__":
    np.random.seed(0)
    main()
