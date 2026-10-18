"""End-to-end PV pipeline on top of the frozen self-supervised encoder.

fit():
  1. day-level PV classifier (gradient boosting) on embeddings + calendar + physics cues,
     labels = days after the true commissioning date of PV customers (out-of-fold by customer)
  2. customer-level logistic regression on summaries of the day-score sequence
     (C and decision threshold picked by leave-one-customer-out F1)
  3. commissioning date = change point of the day-score sequence (or the midday-dip signal,
     whichever scores better D on the training customers)
  4. kWp = ridge regression in log space on midday-reduction features
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from config import SEED
from data import date_to_day, day_to_date
from ssl_model import customer_scale, normalize_days, embed, load_encoder, save_encoder
from features import (day_table, customer_features, estimate_date, capacity_features, add_pixels,
                      DAY_FEATS, LIN_FEATS, CUST_FEATS, CAP_FEATS)


def _day_model(kind="gbm", C=0.05):
    if kind == "linear":      # logistic regression on heatmap pixels
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, min_samples_leaf=50, random_state=SEED)


class PVPipeline:
    def __init__(self, encoder, sun=None, clf="gbm", seq_kwargs=None, threshold="tune", cust_C=None, date_from="gbm"):
        self.encoder = encoder
        self.sun = sun                      # SunIndex or None. NOT used in the submission: a learned
                                            # irradiation proxy counts as weather data under the rules.
        self.clf = clf                      # "gbm": day GBM + customer LR; "lstm": LSTM+attention for both;
                                            # "hybrid": LSTM customer decision + GBM day scores for the date;
                                            # "linear": logistic regression on heatmap pixels + customer LR
        self.date_from = date_from          # clf="cnn": day-score sequence for the change-point date comes from the
                                            # "gbm" day model (default, validated) or from the "cnn" day head
        self.day_feats = LIN_FEATS if clf == "linear" else DAY_FEATS
        self.day_kind = "linear" if clf == "linear" else "gbm"
        self.seq_kwargs = seq_kwargs or {}
        self.seq_model = None
        self.day_model = None
        self.cust_model = None
        self.threshold = 0.5
        self.threshold_mode = threshold          # "tune" (leave-one-out F1 on train) or a fixed float
        self.cust_C = cust_C                     # None: pick C by leave-one-out on train; float: fixed
        self.date_signal = "score"
        self.cap_model = None
        self.info = {}

    # ---------- shared preprocessing (label-free) ----------
    def prepare(self, data):
        """data: cid -> {'days','X'}  ->  cid -> {'days','X','scale','tab'}"""
        prep = {}
        for cid, d in data.items():
            scale = customer_scale(d["X"])
            emb = embed(self.encoder, normalize_days(d["X"], scale, getattr(self.encoder, "norm_mode", "linear"), d["days"]))
            sun = self.sun.get(d["days"]) if self.sun is not None else None
            tab = day_table(d["X"], d["days"], emb, scale, sun)
            if self.clf == "linear":
                tab = add_pixels(tab, d["X"])
            prep[cid] = {"days": d["days"], "X": d["X"], "scale": scale, "sun": sun, "tab": tab}
        return prep

    # ---------- helpers ----------
    def _day_labels(self, prep, labels):
        rows, ys, groups = [], [], []
        for _, r in labels.iterrows():
            cid = r["customer_id"]
            if cid not in prep:
                continue
            tab = prep[cid]["tab"]
            if r["pv_installed"] and r["pv_installation_date"]:
                y = (tab["day"].to_numpy() >= date_to_day(r["pv_installation_date"])).astype(int)
            else:
                y = np.full(len(tab), int(bool(r["pv_installed"])))
            rows.append(tab[self.day_feats].to_numpy()); ys.append(y); groups.append(np.full(len(tab), cid))
        return np.concatenate(rows), np.concatenate(ys), np.concatenate(groups)

    def _score_days(self, prep, cids, model):
        for cid in cids:
            tab = prep[cid]["tab"]
            tab["score"] = model.predict_proba(tab[self.day_feats].to_numpy())[:, 1].astype(np.float32)

    @staticmethod
    def _cust_matrix(prep, cids):
        return np.array([[customer_features(prep[c]["tab"])[k] for k in CUST_FEATS] for c in cids])

    # ---------- fit ----------
    def fit(self, prep, labels, verbose=True):
        labels = labels[labels["customer_id"].isin(prep)].reset_index(drop=True)
        cids = labels["customer_id"].tolist()
        y_c = labels["pv_installed"].astype(int).to_numpy()

        if self.clf in ("lstm", "hybrid", "cnn", "ensemble"):
            from seq_model import SeqPVModel
            from cnn_model import CNNPVModel
            Model = CNNPVModel if self.clf == "cnn" else SeqPVModel
            self.seq_model = Model(**self.seq_kwargs).fit(prep, labels)
            if self.clf != "ensemble":
                self.threshold = 0.5
            if self.clf == "lstm":
                cp, dp = self.seq_model.predict(prep, cids)
                for c in cids:
                    prep[c]["tab"]["score"] = dp[c]
                self._fit_date_and_capacity(prep, labels, verbose)
                return self

        # 1. day-level model with out-of-fold scores for the training customers
        Xd, yd, gd = self._day_labels(prep, labels)
        n_splits = min(4, int(y_c.sum()), int((1 - y_c).sum()))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        cust_y = dict(zip(cids, y_c))
        for tr, te in cv.split(Xd, yd, gd):
            m = _day_model(self.day_kind).fit(Xd[tr], yd[tr])
            self._score_days(prep, sorted(set(gd[te])), m)
        self.day_model = _day_model(self.day_kind).fit(Xd, yd)
        if self.clf in ("hybrid", "cnn"):
            if self.clf == "cnn" and self.date_from == "cnn":
                _, dp = self.seq_model.predict(prep, cids)
                for c in cids:
                    prep[c]["tab"]["score"] = dp[c]
            if verbose:
                print(f"  day model: {len(yd):,} training days ({n_splits}-fold OOF); customer decision: {self.clf}, threshold 0.5"
                      + (f"; date from the {self.date_from} day scores" if self.clf == "cnn" else ""))
            self._fit_date_and_capacity(prep, labels, verbose)
            return self

        # 2. customer-level classifier, C + threshold by leave-one-out F1
        Xc = self._cust_matrix(prep, cids)
        best = None
        for C in ([self.cust_C] if self.cust_C else [0.03, 0.1, 0.3, 1.0, 3.0]):
            oof = np.zeros(len(cids))
            for tr, te in LeaveOneOut().split(Xc):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xc[tr], y_c[tr])
                oof[te] = m.predict_proba(Xc[te])[:, 1]
            thr, f1 = self._best_threshold(oof, y_c)
            if best is None or f1 > best[0] + 1e-9:
                best = (f1, C, thr, oof)
        f1_loo, C, self.threshold, oof_prob = best
        if self.threshold_mode != "tune":
            self.threshold = float(self.threshold_mode)
            f1_loo = f1_score(y_c, (oof_prob >= self.threshold).astype(int), zero_division=0)
        self.cust_model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(Xc, y_c)
        self.info["cust_loo_f1"] = f1_loo; self.info["cust_C"] = C; self.info["threshold"] = self.threshold
        self.info["cust_oof_prob"] = dict(zip(cids, oof_prob.round(3).tolist()))

        if verbose:
            print(f"  day model: {len(yd):,} training days from {len(cids)} customers ({n_splits}-fold OOF)")
            print(f"  customer clf: LOO F1 {f1_loo:.3f}  C={C}  threshold={self.threshold:.3f}")
        self._fit_date_and_capacity(prep, labels, verbose)
        return self

    def _fit_date_and_capacity(self, prep, labels, verbose):
        # 3. date signal choice on the training PV customers (uses OOF day scores)
        pv = labels[labels["pv_installed"] & labels["pv_installation_date"].ne("")]
        d_scores = {}
        for sig in ["score", "dip"]:
            ds = []
            for _, r in pv.iterrows():
                est, _ = estimate_date(prep[r["customer_id"]]["tab"], sig)
                ds.append(max(0.0, 1 - abs(est - date_to_day(r["pv_installation_date"])) / 30))
            d_scores[sig] = float(np.mean(ds)) if ds else 0.0
        # Fixed to the day-score signal: on all 40 labelled customers it gives D 0.78 vs 0.69 for
        # the raw midday-dip signal (see date_experiments.py); the train-only choice was too noisy.
        self.date_signal = "score"
        self.info["date_D_train"] = d_scores; self.info["date_signal"] = self.date_signal

        # 4. capacity regressor (features computed with the TRUE dates for training)
        pvc = pv[pv["pv_capacity_kwp"].notna() & (pv["pv_capacity_kwp"] > 0)]
        if len(pvc) < 3:                       # e.g. human labels: no capacity known -> no kWp predictions (P = 0)
            self.cap_model = None; self.info["cap_loo_P_true_dates"] = None
            if verbose:
                print(f"  date signal: {self.date_signal}  (train D: {d_scores})")
                print(f"  capacity: only {len(pvc)} training customers with a known kWp -> capacity model disabled")
            return
        Xk = np.array([[capacity_features(prep[c]["X"], prep[c]["days"], date_to_day(d), prep[c]["scale"], prep[c]["sun"])[k]
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
        self.info["cap_loo_P_true_dates"] = float(best[0]); self.info["cap_alpha"] = best[1]

        if verbose:
            print(f"  date signal: {self.date_signal}  (train D: {d_scores})")
            print(f"  capacity: LOO P {best[0]:.3f} with true dates, alpha={best[1]}")

    @staticmethod
    def _best_threshold(prob, y):
        cands = np.unique(np.concatenate([[0.5], (np.sort(prob)[:-1] + np.sort(prob)[1:]) / 2]))
        best = (-1, 0.5)
        for t in cands:
            f = f1_score(y, (prob >= t).astype(int), zero_division=0)
            if f > best[0] + 1e-9 or (abs(f - best[0]) < 1e-9 and abs(t - 0.5) < abs(best[1] - 0.5)):
                best = (f, t)
        return best[1], best[0]

    # ---------- predict ----------
    def predict(self, prep):
        cids = list(prep)
        if self.clf == "lstm":
            cp, dp = self.seq_model.predict(prep, cids)
            for c in cids:
                prep[c]["tab"]["score"] = dp[c]
            prob = np.array([cp[c] for c in cids])
        elif self.clf in ("hybrid", "cnn"):
            self._score_days(prep, cids, self.day_model)
            cp, dp = self.seq_model.predict(prep, cids)
            if self.clf == "cnn" and self.date_from == "cnn":
                for c in cids:
                    prep[c]["tab"]["score"] = dp[c]
            prob = np.array([cp[c] for c in cids])
        elif self.clf == "ensemble":                      # mean of the LR probability and the LSTM probability
            self._score_days(prep, cids, self.day_model)
            p_lr = self.cust_model.predict_proba(self._cust_matrix(prep, cids))[:, 1]
            cp, _ = self.seq_model.predict(prep, cids)
            prob = 0.5 * (p_lr + np.array([cp[c] for c in cids]))
        else:
            self._score_days(prep, cids, self.day_model)
            prob = self.cust_model.predict_proba(self._cust_matrix(prep, cids))[:, 1]
        rows = []
        for cid, p in zip(cids, prob):
            pv = bool(p >= self.threshold)
            date, cap, diag = "", np.nan, {}
            if pv:
                d0, diag = estimate_date(prep[cid]["tab"], self.date_signal)
                date = str(day_to_date(d0))
                if self.cap_model is not None:
                    f = capacity_features(prep[cid]["X"], prep[cid]["days"], d0, prep[cid]["scale"], prep[cid]["sun"])
                    cap = float(np.exp(self.cap_model.predict(np.array([[f[k] for k in CAP_FEATS]]))[0]))
                    cap = round(cap, 1); diag["peak_red_kw"] = round(f["peak_red_kw"], 1); diag["cap_sun_kw"] = round(f["cap_sun_kw"], 1)
            rows.append({"customer_id": cid, "pv_installed": pv, "pv_installation_date": date,
                         "pv_capacity_kwp": cap, "pv_prob": round(float(p), 3), **diag})
        return pd.DataFrame(rows)

    # ---------- persistence ----------
    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        save_encoder(self.encoder, d / "encoder.pt")
        if self.sun is not None:
            self.sun.save(d / "sun_index.npz")
        joblib.dump({"day_model": self.day_model, "cust_model": self.cust_model, "cap_model": self.cap_model,
                     "threshold": self.threshold, "date_signal": self.date_signal, "clf": self.clf,
                     "date_from": self.date_from, "seq_model": self.seq_model}, d / "downstream.joblib")
        with open(d / "fit_info.json", "w") as f:
            json.dump(self.info, f, indent=1, default=float)

    @classmethod
    def load(cls, d):
        d = Path(d)
        p = cls(load_encoder(d / "encoder.pt"))
        s = joblib.load(d / "downstream.joblib")
        p.day_model, p.cust_model, p.cap_model = s["day_model"], s["cust_model"], s["cap_model"]
        p.threshold, p.date_signal = s["threshold"], s["date_signal"]
        p.clf, p.seq_model, p.date_from = s.get("clf", "gbm"), s.get("seq_model"), s.get("date_from", "gbm")
        p.day_feats = LIN_FEATS if p.clf == "linear" else DAY_FEATS
        p.day_kind = "linear" if p.clf == "linear" else "gbm"
        return p
