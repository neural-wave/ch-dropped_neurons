"""Steps 3-4 together: CNNPipeline = CNN presence decision + date from the CNN day scores + kWp regression.

fit(data, labels):
  1. CNNPVModel on the training customers (presence + per-day scores)
  2. capacity: ridge regression in log space on postprocess.capacity_features (features computed with the TRUE
     dates), alpha by leave-one-out; skipped when fewer than 3 training customers have a kWp (e.g. human labels)
predict(data) -> DataFrame customer_id, pv_installed, pv_installation_date, pv_capacity_kwp, pv_prob
  presence  = P(PV) >= threshold (0.5)
  date      = postprocess.estimate_date on the CNN's per-day scores
  capacity  = exp(ridge(features after the predicted date)) or NaN without a capacity model"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import TRAIN
from data import date_to_day, day_to_date
from images import customer_scale
from cnn_model import CNNPVModel
from postprocess import estimate_date, capacity_features, CAP_FEATS


class CNNPipeline:
    def __init__(self, threshold=None, **cnn_kwargs):
        self.threshold = TRAIN["threshold"] if threshold is None else threshold
        self.cnn_kwargs = cnn_kwargs
        self.cnn = None
        self.cap_model = None
        self.info = {}

    def fit(self, data, labels, verbose=True):
        labels = labels[labels["customer_id"].isin(data)].reset_index(drop=True)
        self.cnn = CNNPVModel(**self.cnn_kwargs).fit(data, labels)
        self.info = {"n_train": int(len(labels)), "n_train_pv": int(labels["pv_installed"].sum()), **self.cnn_kwargs}
        pv = labels[labels["pv_installed"] & labels["pv_installation_date"].ne("")]
        pvc = pv[pv["pv_capacity_kwp"].notna() & (pv["pv_capacity_kwp"] > 0)]
        if len(pvc) < 3:
            self.cap_model = None; self.info["capacity"] = f"disabled ({len(pvc)} training customers with a kWp)"
            if verbose:
                print(f"  CNN trained on {len(labels)} customers ({int(labels['pv_installed'].sum())} PV); capacity model {self.info['capacity']}")
            return self
        Xk = np.array([[capacity_features(data[c]["X"], data[c]["days"], date_to_day(d), customer_scale(data[c]["X"]))[k]
                        for k in CAP_FEATS] for c, d in zip(pvc["customer_id"], pvc["pv_installation_date"])])
        yk = np.log(pvc["pv_capacity_kwp"].to_numpy(dtype=float))
        best = None
        for a in [0.1, 0.3, 1.0, 3.0, 10.0]:
            oof = np.zeros(len(yk))
            for tr, te in LeaveOneOut().split(Xk):
                oof[te] = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xk[tr], yk[tr]).predict(Xk[te])
            P = np.mean(np.maximum(0, 1 - np.abs(np.exp(oof) - np.exp(yk)) / np.exp(yk) / 0.5))
            if best is None or P > best[0]:
                best = (P, a)
        self.cap_model = make_pipeline(StandardScaler(), Ridge(alpha=best[1])).fit(Xk, yk)
        self.info["capacity"] = f"ridge alpha {best[1]}, leave-one-out P {best[0]:.3f} with true dates"
        if verbose:
            print(f"  CNN trained on {len(labels)} customers ({int(labels['pv_installed'].sum())} PV); capacity {self.info['capacity']}")
        return self

    def predict(self, data, cids=None):
        cids = list(data) if cids is None else cids
        prob, day, att = self.cnn.predict(data, cids)
        rows = []
        for c in cids:
            p = prob[c]; pv = bool(p >= self.threshold)
            date, cap = "", np.nan
            if pv:
                d0, _ = estimate_date(data[c]["days"], day[c])
                date = str(day_to_date(d0))
                if self.cap_model is not None:
                    f = capacity_features(data[c]["X"], data[c]["days"], d0, customer_scale(data[c]["X"]))
                    cap = round(float(np.exp(self.cap_model.predict(np.array([[f[k] for k in CAP_FEATS]]))[0])), 1)
            rows.append({"customer_id": c, "pv_installed": pv, "pv_installation_date": date,
                         "pv_capacity_kwp": cap, "pv_prob": round(float(p), 3)})
        return pd.DataFrame(rows)

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cnn": self.cnn, "cap_model": self.cap_model, "threshold": self.threshold, "cnn_kwargs": self.cnn_kwargs},
                    d / "model.joblib")
        with open(d / "info.json", "w") as f:
            json.dump(self.info, f, indent=1, default=str)

    @classmethod
    def load(cls, d):
        s = joblib.load(Path(d) / "model.joblib")
        p = cls(threshold=s["threshold"], **s["cnn_kwargs"])
        p.cnn, p.cap_model = s["cnn"], s["cap_model"]
        return p
