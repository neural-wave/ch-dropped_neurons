# AIL PV detection: improved LSTM notebook

Open `pv_lstm.ipynb` and use `.venv/bin/python` as the notebook kernel. The notebook retains the original baseline and appends the complete improvement study, current metrics, reproducible training functions, and offline export.

**95/100 was not reached.** The selected development cross-validation score is **85.81/100**, up from **76.39/100** (+9.42 points). Detection F1 improved from **0.9302** to **0.9767**. The final selected pipeline uses separate LSTM checkpoints for presence and the other outputs, plus consumption-only date and capacity refinements.

## Measured results

| Metric | Original | Selected |
|---|---:|---:|
| Main challenge score | 76.39 | 85.81 |
| PV F1 | 0.9302 | 0.9767 |
| Precision | 0.9524 | 1.0000 |
| Recall | 0.9091 | 0.9545 |
| Date component D | 0.5617 | 0.7635 |
| Capacity component P | 0.2465 | 0.4268 |
| Date MAE, days, true positives | 77.35 | 19.57 |
| Capacity MAPE, true positives | 73.08% | 69.46% |

The selected model has 21 true positives, one false negative, no false positives, and 18 true negatives across the 40 development customers. Capacity remains the main weakness. The capacity prior improves the challenge's clipped capacity score but does not eliminate large relative errors.

Each reported prediction excludes its customer from that fold's neural training, scaler, shared factor, capacity regressors, and prior. However, model choices were compared using these same folds. These are **exploratory development model-selection results**, not an unbiased hidden-test estimate. The hidden test data/labels are unavailable. No training-set score or extension bonus is used to claim 95.

## Run and reproduce

Python 3.12 was used. On a new machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The notebook expects `AIL_data_for_neural_wave_v1.1/challenge_set/` beside it. Update `DATA` if necessary. Installation needs network access; training and inference work offline.

The original baseline cells retrain the original 30-epoch model. The improvement section defaults to verified cached validation results and included final artifacts. It verifies source/data fingerprints before accepting the cache. Set `RECOMPUTE_STUDY=True` to retrain the five improved folds, and `RETRAIN_FINAL=True` to recreate the selected artifacts. The checkpoint schedule is 30 epochs for date/capacity and 60 for presence, hidden size 24, seed 42, threshold 0.5, CPU. Full retraining takes several minutes. The displayed implementation cells export to the support modules.

## What changed

The presence head benefits from 60 epochs while the earlier date/capacity checkpoint generalizes better. Commissioning dates combine neural localization, a night-adjusted local change point, and a shared daytime withdrawal factor learned from training consumption and labels. Initial capacity combines neural, shared-factor, regularized linear and tree estimates, then shrinks halfway toward a training-only prior. The prior maximizes expected training capacity score. Development comparisons selected these choices.

Inputs remain consumption and derived calendar information only. IDs are join keys, never model features. Epoch timestamps control local calendar features and DST. Invalid energy is masked; duplicates across chunks are rejected; leading-zero IDs are preserved. Capacity refers to the initial installation. The optional extension detector and unlabelled pretraining are not implemented; the extension file is still emitted with its header.

## Files

- `pv_lstm.ipynb`: executed baseline and improvement study, all implementation and export cells.
- `lstm_pipeline.py`, `pv_refinement.py`, `pv_training.py`: reusable implementation and training functions.
- `lstm_submission.zip`: code and trained artifacts only, ready for offline inference.
- `validation/metrics.json`, `validation/experiment_results.csv`: aggregate results and complete-pipeline comparisons.
- `experiments/`: local experiment scripts, original baseline backup and private caches/diagnostics; do not submit this folder.
- `verify_lstm.py`, `verify_refinement.py`: synthetic correctness and serialization checks.

Run inference from `lstm_submission/`:

```bash
python inference.py path/to/test_set.csv
```

Both outputs follow PDF §4.1: UTF-8, semicolon separators, string IDs, TRUE/FALSE, ISO dates, and empty date/capacity for non-PV customers. No development labels are read at inference.

Keep customer data, feature caches and out-of-fold diagnostics local. The archive includes no development CSVs, labels or per-customer diagnostics.
