# C9v2 + kWp network submission: notes for the team

Combined version of 2026-10-18. Built from the C9v2 freeze of 2026-10-17 (tree sha256 a7133f67217aaeb1); the only
change is the kWp: geometric mean of the C9v2 kWp and the kWp network (`pvsteps/kwp_net.py`, `model/kwp_net.pt`
sha256 912f8416f10704cc), at the same detected date. Presence, dates and extensions are byte-identical to C9v2.

Outputs on the 40 labelled customers (CPU): `predictions.csv` md5 3ea24e409d5e27f57f562a7f85587746 (C9v2 alone:
b90bd411e9897027b83f3d311b843a21), extensions md5 99b0e71ebf7e631849738e3efed020d5 (unchanged).

## Quick check

```
pip install -r requirements.txt
python inference.py ../challenge_set/energy_consumption_dev.csv /tmp/out
# expected: [inference] 40 customers, 21 PV, 2 extensions
```

## kWp blend (what changed)

Paired 5-fold CV on the 22 labelled PV customers, 2 repeats, both kWp models refitted inside identical folds
(C9v2 ridges re-implemented from `train_pipeline.py`; its `g` table could not be refitted and is the shipped one):

| kWp | true dates: MAPE / P | LSTM dates: MAPE / P |
|---|---|---|
| C9v2 | 26.0% / 0.524 | 24.6% / 0.527 |
| network | 27.6% / 0.515 | 28.1% / 0.510 |
| blend 50/50 (shipped) | 23.1% / 0.588 | 23.3% / 0.570 |

C9v2 and the network are tied (C9v2 better on 11 and 10 of 22 sites; 90% bootstrap CI of the MAPE difference spans
zero); their log errors correlate 0.75 / 0.43, so the blend beats both. ΔP ≈ +0.05 to +0.06, i.e. roughly +0.7 to +0.9
main-score points at F1 0.977. The 50/50 weight was fixed before looking. The network's training used public UK PV
generation data (openclimatefix/uk_pv) to build synthetic examples; see README "Training data used". It is not used
at inference.

The C9v2 scores below are unchanged C9v2 figures (its P = 0.548 is on its own 21 detected sites and dates).

## Scores (40 labelled customers, 5 folds x 5 repeats, out-of-fold)

| Metric | Value |
|---|---|
| Main score F1 x (70 + 15 D + 15 P) | 87.72 |
| F1 / precision / recall | 0.977 / 1.000 / 0.955 (21 TP, 0 FP, 1 FN per repeat) |
| D (date) | 0.772, 17 of 21 dates within 7 days |
| P (kWp) | 0.548, MAPE 23%, median APE 20% |
| Estimate for the hidden 20 | about 82 to 84, +-8 (one error on 20 customers is 3 to 4 points) |

Thresholds and models are refitted inside each fold, but the design was chosen on the same 40 customers, so the
out-of-fold score is somewhat optimistic.

External detection checks (models fitted on the 40, nothing tuned on these sets):

| Dataset | F1 | Accuracy | False positives |
|---|---|---|---|
| EKZ single-family houses (495, 60 PV) | 0.897 | 0.976 | 4 of 435 |
| EKZ all meters (2447, 77 PV) | 0.532 | 0.955 | 97 |
| HEAPO households (650, 516 PV) | 0.895 | 0.845 | 17 of 134 |

## Alternatives tested against it (same folds, paired)

| Candidate | Main score | Note |
|---|---|---|
| C12 (learned detector: day encoder + logistic regression) | 85.60 | one PV fewer on the 40; many false positives on EKZ/HEAPO |
| C12 with encoder trained on the pool only | 85.35 | same detection as C12 |
| LSTM (team model, retrained in our folds, 1 repeat) | 76.93 | 3 false positives |
| CNN detector, alone or OR-ed with C9v2 | fails | adds false positives |
| Classical date instead of the encoder date | 86.93 | encoder date kept (+0.8) |
| Single-constant physical kWp formula | 83.38 / 84.98 | regressions kept |
| Thresholds fitted on the manual pool labels | 83.82 / 82.96 | worse on the 40 |
| Best non-promoted tweaks (readout, clipping) | 87.84 / 87.79 | +0.11 / +0.07, noise (P 0.72 / 0.58) |

## Where to start reading

1. `inference.py`: the whole flow in about 100 lines, with the per-customer fallbacks.
2. `pvsteps/detect_gate.py`: PV yes/no. Fixed rules: change-point gain threshold, midday zero-share path (outage days
   ignored), bell-shape gate.
3. `pvsteps/date.py` with `encoder.py`, `day_features.py`, `day_score.py`: commissioning date.
4. `pvsteps/capacity.py`, `capacity_yoy.py`: kWp (DC).
5. `pvsteps/extensions_gated.py`: extensions.

All fitted values are in `model/state.pkl` (plain dict) and `model/encoder_shipped.pt`.

## Known limits

- PV present from the first day of data without midday zero readings has no before/after and can be missed.
- A PV whose midday zeros all fall on days with zero night load (battery full, whole-day zeros) loses the zero path.
- If scikit-learn is missing, `state.pkl` cannot be read and every customer is written FALSE; install the requirements.
