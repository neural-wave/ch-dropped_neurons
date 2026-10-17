# CNN + day model: PV detection from 15-minute consumption

Solution for the AIL × Neural Wave challenge: for each customer, does it have a PV plant, since when, and how large.
Two models share the work. A 2-D CNN reads the customer's whole (days × hours) image and decides **PV yes / no**.
A per-day gradient-boosting model, fed by a self-supervised day encoder, scores every day as "PV active" and the
**commissioning date** is the change point of that score sequence. Capacity comes from a ridge regression on
scale-aware cues (only fitted when training labels carry a kWp).

Only the supplied consumption values and their timestamps are used: no weather, irradiation, location or sun
geometry. Calendar features (day of year, month, fixed clock windows) are derived from the timestamps in the data.

## Results

Protocol: train on 218 human-labelled customers of the unlabelled file, test on all 40 labelled customers
(never used for training). Seed 2 of `run_all.sh`:

| | value |
|---|---|
| F1 (PV yes / no) | 0.977 |
| false positives / misses | 0 / 1 |
| D (date score) | 0.611 |
| predicted dates within 1 day of the label | 13 of 21 |
| main score, training run (no capacity, P = 0) | 77.3 |
| main score, `inference.py` with the kWp fallback rule (P = 0.17) | 79.8 |

`results/eval_seed2.json` holds the per-customer predictions of that run and `results/overview_seed2.png` the
dates overview (dashed black = labelled date, green = predicted date). Repeated runs land at F1 0.93–0.98 with
0 false positives; the misses are late installs with a short PV tail or a small plant on a very large load.

## How it works

| Stage | File | What it does |
|---|---|---|
| Data | `data.py`, `prepare_data.py` | CSV → one 96-slot profile per (customer, local day). Keeps `POD*` ids, repairs DST days, drops days with < 80 measured slots. Cached in `cache/*.npz`. |
| Day encoder | `ssl_model.py`, `ssl_pretrain.py` | 1-D CNN, 96 slots → 16-d embedding, trained to reconstruct masked blocks of each day on all customers (no labels). Days are divided by the customer's own median level. |
| Day features | `features.py` | Per day: embedding + day-of-year sine/cosine + midday-vs-shoulder cues. Also builds the `relsmooth` image: log2 ratio of each slot to the day's 06–09 h / 17–20 h shoulder level, 15-day × 3-slot median filter. |
| PV yes / no | `cnn_model.py` | 2-D CNN (84 k parameters) on the relsmooth image: conv stack pretrained by masked-patch reconstruction (`ssl2d_pretrain.py`), 7-day temporal convolution, per-day head, attention pooling over days, customer head. Threshold 0.5. |
| Date | `pipeline.py`, `features.estimate_date` | Gradient boosting on the day features (positives = days on/after the commissioning date, out-of-fold by customer) scores every day; 7-day smoothing and a single mean-shift fit give the date. If PV is present from the start, the first day of data is returned. |
| Capacity | `pipeline.py` | Ridge regression in log space on peak midday reduction, saturation and load level; fitted only when kWp labels exist. |
| Score | `scoring.py` | Exact challenge score `F1 · (70 + 15·D + 15·P)`. |
| Entry point | `inference.py` | `python inference.py test.csv` → `predictions.csv` + `pv_extensions_predictions.csv` (semicolons, one row per id). |

Training the CNN: one batch per epoch holding every training customer as a random window of 120–365 days; the
window label is 1 only if it contains a PV day; loss = BCE(customer) + BCE(days); AdamW, cosine schedule, 150 epochs.
Full details in the docstrings of `cnn_model.py` and `pipeline.py`.

## Setup

```bash
bash setup_server.sh          # Linux: creates .venv (uv or python3 -m venv), installs requirements.txt, runs check_env.py
python check_env.py           # any machine: python, GPU, libraries, data files
```
Python 3.12 or newer with **numpy >= 2 and scikit-learn >= 1.9** (see `requirements.txt`): the shipped `model/` was
pickled with those and older versions cannot load it. Verified: a fresh Python 3.13 venv from `requirements.txt`
runs `inference.py` on the labelled file in about 10 s on CPU and reproduces the table above.
Data: the three challenge CSVs are expected in `../challenge_set/`; set `PV_DATA_DIR=<folder>` to point elsewhere.
Torch uses CUDA when available, otherwise Apple MPS, otherwise CPU (`PV_DEVICE=cuda|mps|cpu` to force).

## Reproduce

```bash
bash run_all.sh               # seed 2; SEED=5 bash run_all.sh for another seed
```
Steps: cache → 1-D encoder (30 epochs) → 2-D conv stack (relsmooth) → training labels → train + test on the 40 →
`model/` → summary → `plots/dates_seed2/overview_all40.png`. Steps whose output exists are skipped.
About 15 min on an RTX 4090 after the cache exists. GPU training is not bit-reproducible; expect a few
tenths of a point of F1 between runs of the same seed.

Single steps:
```bash
python build_human_labels.py labels/manual/annotator1.csv labels/manual/annotator2.csv --out labels/human_labels.csv
python train_downstream.py --clf cnn --image relsmooth --cnn-pretrained artifacts/heatcnn_ssl_relsmooth.pt \
    --train-labels labels/human_labels.csv --test-all-labelled --no-save --eval-tag cnn_seed2 --save-to model --seed 2
python summarize_evals.py cnn_seed                 # table of artifacts/eval_*.json
python plot_heatmap.py --all --relsmooth --pred artifacts/eval_holdout_cnn_seed2.json --outdir dates_seed2
python plot_attention.py --model model             # Grad-CAM maps: where the CNN looks (plots/attention/)
python label_unlabelled.py --artifacts model --exclude labels/human_labels.csv --out artifacts/labels_unlabelled.csv
python inference.py ../challenge_set/energy_consumption_dev.csv   # writes predictions.csv in the current folder
python score_labels.py predictions.csv --model model   # scores that file against the 40 labels, prints the date table
```

## Labels

`labels/manual/annotator1.csv`, `annotator2.csv`: manual labels of unlabelled customers by two team members
(`customer_id, pv_installed, pv_installation_date, confidence`). `build_human_labels.py` keeps the customers on
which every annotator agrees, drops the 40 labelled customers, averages the dates and writes
`labels/human_labels.csv` (218 customers, 66 PV). No capacity is known for these, so the capacity regressor
is disabled under this protocol; `inference.py` then falls back to a peak-reduction rule for kWp.

## Layout

```
run_all.sh            full reproduction            model/      shipped model (encoder.pt, downstream.joblib)
inference.py          challenge entry point        labels/     annotator files + merged training labels
train_downstream.py   training / evaluation        results/    seed-2 evaluation and overview plot
pipeline.py           day model + date + capacity  splits/     32/8 split used by the earlier protocol
cnn_model.py          2-D CNN (PV yes / no)        artifacts/  (generated) pretrained weights, eval_*.json
features.py           day features, images, date   cache/      (generated) daily profiles
data.py               CSV parsing                  plots/      (generated) heatmaps, Grad-CAM
ssl_model.py, ssl_pretrain.py, ssl2d_pretrain.py   self-supervised pretraining
plot_heatmap.py, plot_attention.py                 visual checks
build_human_labels.py, label_unlabelled.py, score_labels.py, summarize_evals.py, scoring.py, check_env.py
```
