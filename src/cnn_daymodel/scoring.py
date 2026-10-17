"""Exact main score from the challenge PDF, section 4.2:
   main = F1 * (70 + 15*D + 15*P)
   D_i = max(0, 1 - |date error days| / 30), P_i = max(0, 1 - APE / 50%),
   both averaged over correctly detected PV customers (missing/invalid -> 0)."""
import numpy as np
import pandas as pd


def _parse_date(s):
    try:
        if s is None or str(s).strip() == "" or str(s).lower() == "nan":
            return None
        return pd.Timestamp(str(s).strip()[:10])
    except Exception:
        return None


def main_score(pred, truth):
    """pred/truth: DataFrames with customer_id, pv_installed (bool), pv_installation_date (str),
    pv_capacity_kwp (float). Returns dict with F1, D, P, score and counts."""
    t = truth.set_index("customer_id"); p = pred.set_index("customer_id").reindex(t.index)
    yt = t["pv_installed"].astype(bool).values
    yp = p["pv_installed"].fillna(False).astype(bool).values
    tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    D, P, rows = [], [], []
    for cid in t.index[yt & yp]:
        td, pdate = _parse_date(t.loc[cid, "pv_installation_date"]), _parse_date(p.loc[cid, "pv_installation_date"])
        err_days = abs((pdate - td).days) if (td is not None and pdate is not None) else None
        d_i = max(0.0, 1 - err_days / 30) if err_days is not None else 0.0
        tc = float(t.loc[cid, "pv_capacity_kwp"]); pc = p.loc[cid, "pv_capacity_kwp"]
        try:
            pc = float(pc); ape = abs(pc - tc) / abs(tc) if np.isfinite(pc) and tc != 0 else None
        except Exception:
            ape = None
        p_i = max(0.0, 1 - ape / 0.5) if ape is not None else 0.0
        D.append(d_i); P.append(p_i)
        rows.append({"customer_id": cid, "date_err_days": err_days, "D_i": d_i,
                     "cap_true": tc, "cap_pred": pc, "ape": ape, "P_i": p_i})
    Dm = float(np.mean(D)) if D else 0.0; Pm = float(np.mean(P)) if P else 0.0
    score = f1 * (70 + 15 * Dm + 15 * Pm) if tp > 0 else 0.0
    return {"F1": f1, "D": Dm, "P": Pm, "score": score, "tp": tp, "fp": fp, "fn": fn,
            "n": int(len(yt)), "details": pd.DataFrame(rows)}


def format_score(r):
    return (f"score {r['score']:.2f} / 100   F1 {r['F1']:.3f} (tp {r['tp']} fp {r['fp']} fn {r['fn']}, n {r['n']})"
            f"   D {r['D']:.3f}   P {r['P']:.3f}")
