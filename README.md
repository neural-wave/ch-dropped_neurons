# PV detection from electricity consumption profiles

Team **Dropped Neurons**, AIL × Neural Wave challenge. For every customer the pipeline decides whether a photovoltaic
system is present, estimates its commissioning date and nominal capacity (kWp), and reports later capacity
extensions, using only the 15-minute grid-withdrawal series and its timestamps. No weather, irradiation or location
data is read at inference, and no LLM is part of the pipeline.

## Repository layout

| Path | Content |
|---|---|
| `src/pv_submission_combined/` | **The submission.** `inference.py` + `components/` (code and weights of the three models below), requirements. Self-contained. |
| `src/cnn_daymodel/` | Tasks 1–2: 2-D CNN presence detector and day encoder + gradient-boosting date model, with full training. |
| `src/kWp/` | Task 3: kWp network (training, standalone inference in `inference/`). |
| `src/pv_extension/` | Task 4: extension detector (training, standalone inference in `inference/`). |
| `src/labeling/` | Tool used to label the unlabelled series by hand. |
| `data/` | Manual labels of unlabelled customers and CNN-inferred labels used during training. No AIL time series. |
| `assets/` | Figures, e.g. `extension_model_predictions/` (leave-one-out inspection of the extension model). |
| `test/` | Scripts to test the submission: sample cutter, synthetic input generator, output validator, full check. |
| `requirements.txt` | Environment for running and testing from the repository root. |
| `docs/TESTING.md` | Step-by-step runbook: organizer dry run, verification commands with expected outputs, retraining. |

## 1. Running the submission

Requirements: Python 3.12+, CPU only, no network access after installation. Inside `src/pv_submission_combined/`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python inference.py path/to/test_set.csv
```

On Linux without a GPU, install the CPU torch wheel first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

- **Input:** the challenge time-series CSV (`customer_id;energy_consumption_kwh;timestamp_local;timestamp_epoch`,
  `;` or `,` separated). Days with 92 or 100 intervals (daylight saving) are handled.
- **Output:** `predictions.csv` and `pv_extensions_predictions.csv` in the **current working directory**. UTF-8, `;`
  separated, `TRUE`/`FALSE`, dates `YYYY-MM-DD`, kWp with 2 decimals, customer ids kept as strings, empty date and
  capacity when `pv_installed` is `FALSE`, exactly one row per customer in input order. The extension file holds one
  row per predicted extension with the total capacity after it, or only its header.
- **Runtime:** about 35 s for 40 customers on CPU.
- The script prints one `[inference]` line per step and ends with
  `[inference] wrote predictions.csv (N rows, M PV) and pv_extensions_predictions.csv (K rows) in Ts`. A line with
  `WARNING` means a step fell back to a simpler estimate (see `src/pv_submission_combined/README.md`).

## 2. Testing the submission

All commands are run from the repository root. `CHALLENGE` stands for the folder holding `energy_consumption_dev.csv`,
`pv_labels_dev.csv` and `pv_extensions_labels_dev.csv`.

**Full check in one command** (copies the submission to a scratch folder, builds a fresh Python 3.12 venv from its
`requirements.txt`, then runs the organizers' one-argument command on: a synthetic file, 5 random real customers, the
same 5 with leading-zero ids, the labelled extension customer POD13, and all 40 customers with an md5 check; every
output is validated against the challenge format rules):

```bash
python test/run_checks.py CHALLENGE
```

It ends with `ALL CHECKS PASSED` or stops at the first failure. Without the argument it runs the synthetic check only.

**Individual tools:**

| Command | Purpose |
|---|---|
| `python test/make_sample.py CHALLENGE/energy_consumption_dev.csv sample.csv 2026 [n]` | Cuts `n` (default 5) randomly chosen customers out of the dev file into a valid test CSV. Seed 2026 picks POD22, POD63, POD36, POD23, POD26. |
| `python test/make_mock.py mock_dir` | Writes a synthetic `test_set.csv` in the challenge schema (4 customers, one id with leading zeros, real DST days) plus label files. Contains no AIL data. |
| `python test/validate.py test.csv out_dir labels.csv ext_labels.csv` | Checks both output files against every format rule of the challenge document and computes the main score and extension bonus with the official formulas. |

Reference results of the shipped submission on the development data (verified 2026-10-18; all format checks pass):

| Input | Main task | Extensions |
|---|---|---|
| 5 random customers, seed 2026 | TP 4, FP 0, FN 0; F1 1.000, D 0.742, P 0.716; main score 91.9 | 1 predicted, 0 matched |
| POD13 (the labelled extension customer) | F1 1.000, D 0.000, P 0.888; main score 83.3 | matched: D_ext 0.767, P_ext 0.247, bonus 8.5 |
| All 40 labelled customers | TP 21, FP 0, FN 1; F1 0.977, D 0.611, P 0.744; main score 88.2 | 3 predicted, 1 matched; bonus 4.3 |

`predictions.csv` md5 on all 40: `32ccb951673530650a817af9ae08181f`, extensions `e6c913cd7b8920ca634fc6f62b1869cc`.
Detection and dates come from a model trained without these 40 customers, but the kWp network was fine-tuned on
them, so P is in-sample. With the cross-validated kWp (P ≈ 0.51) the realistic main-score estimate is about **84.8**.

## 3. Method

| Task | Method | Fitted on |
|---|---|---|
| 1. Presence | 2-D CNN on the customer's (days × hours) image of each slot relative to the day's shoulder level; 7-day temporal convolution, per-day head, attention pooling over days, customer head. | conv stack: label-free masked-patch pretraining; classifier: 218 hand-labelled customers of the unlabelled pool |
| 2. Date | A 1-D CNN day encoder (masked-reconstruction pretraining, no labels) embeds every day; gradient boosting scores each day as "PV active"; the date is the change point of the smoothed score sequence (first day of data when PV is present from the start). | same 218 hand-labelled customers |
| 3. Capacity | Per-day 1-D CNN + 4-layer transformer over the 56 days before and 336 days after the date; attention pooling and a head give log kWp, averaged over ±14-day date shifts and calibrated. | synthetic sites (see 4.2), then fine-tuned on the 22 labelled PV customers |
| 4. Extensions | Weekly CNN + bidirectional GRU giving per week p(nothing / PV installed / PV extended); peaks of p(extended) above a threshold, not at the installation; capacity after = previous kWp × growth of a midday PV-strength index. | 11 hand-labelled extensions + the official dev extension, dev customers as negatives, synthetic extensions and installs built from AIL data |

Details, fallbacks and the dev-set check: `src/pv_submission_combined/README.md`.

**Label conflict to know about.** For 3 dev customers the team's manual labels see PV from the start of the data and
an extension later, while the official labels call that later date the initial installation. The presence/date model
(trained on the team's labels) follows the manual view, so on such customers the official scorer sees a wrong date
and a false-alarm extension (2 of the 3 extensions predicted on the dev set).

## 4. Reproducing the training

`src/pv_submission_combined/components/` holds copies of the three models. After retraining any of them, refresh the
copies with `bash src/pv_submission_combined/build.sh` and re-run the checks of section 2.

| Component in the submission | Source | Training |
|---|---|---|
| `components/cnn_daymodel/` (code + `model/encoder.pt`, `model/downstream.joblib`) | `src/cnn_daymodel/` | 4.1 |
| `components/kwp/` (`inference.py`, `kwp.pt`) | `src/kWp/inference/` | 4.2 |
| `components/pv_extension/` (`inference.py`, `ext_net.pt`) | `src/pv_extension/inference/` | 4.3 |

### 4.1 Presence and date (`src/cnn_daymodel/`)

Full reproduction script; the three challenge CSVs are expected in `src/challenge_set/` or in `PV_DATA_DIR`.

```bash
bash src/cnn_daymodel/setup_server.sh                   # venv from src/cnn_daymodel/requirements.txt
PV_DATA_DIR=CHALLENGE bash src/cnn_daymodel/run_all.sh  # cache -> 1-D encoder -> 2-D conv stack -> labels -> train/test -> model/ -> plots
```

About 15 minutes on one GPU after the data cache exists. Steps and single commands: `src/cnn_daymodel/README.md`.

### 4.2 kWp network (`src/kWp/`)

Trained on synthetic sites, then fine-tuned on the 22 labelled PV customers. A synthetic site is a PV-free period
of an unlabelled customer from which real PV generation per kWp, taken from the public dataset
`openclimatefix/uk_pv` (Sheffield Solar, CC-BY-4.0), times a random capacity is subtracted and clipped at zero. The
UK data is used only to build training examples; it is not shipped and not read at inference.

Requirements: Python 3.10+, `numpy pandas torch`, plus `polars` for the library step. A GPU is recommended for
pretraining. Paths default to `/workspace/data`; every one is an argument. `DATA` below is any working folder.

```bash
# UK PV library (once): download openclimatefix/uk_pv from Hugging Face; the script globs the HF cache snapshot
python src/kWp/build_pv_library.py --years 2022,2023,2024 --out DATA/pv_library_15min.npz
# daily consumption cache from both challenge time-series files (also used by 4.3)
python src/kWp/train.py prepare --ts CHALLENGE/energy_consumption_dev.csv --ts CHALLENGE/energy_consumption_unlabelled_dev.csv --cons DATA/cons.npz
# pretrain on synthetic sites, cross-validate, fine-tune on all 22 -> src/kWp/inference/kwp.pt
python src/kWp/train.py pretrain --out pretrained.pt --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --cnn_labels data/cnn_inferenced_label.csv --library DATA/pv_library_15min.npz
python src/kWp/train.py cv       --init pretrained.pt --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --cnn_labels data/cnn_inferenced_label.csv --library DATA/pv_library_15min.npz
python src/kWp/train.py final    --init pretrained.pt --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --cnn_labels data/cnn_inferenced_label.csv --library DATA/pv_library_15min.npz
```

`data/cnn_inferenced_label.csv` marks which unlabelled customers are PV-free (used to pick synthetic hosts).
Expected cross-validation (5 folds × 2, true dates): MAPE 27.6 %, P 0.515; with estimated dates MAPE 28.1 %, P 0.510.
Standalone use: `python src/kWp/inference/inference.py consumption.csv detections.csv` (see `src/kWp/inference/HOWTO.md`).

### 4.3 Extension detector (`src/pv_extension/`)

Trained on AIL data only, with the manual extension labels in `data/manual_labels/extension_with_startdates.csv`
and the official dev extension. Needs the `cons.npz` cache from 4.2.

```bash
for K in $(seq 0 10); do python src/pv_extension/train.py loo --fold $K --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --official_ext CHALLENGE/pv_extensions_labels_dev.csv --out_dir DATA/ext; done
python src/pv_extension/train.py summary --out_dir DATA/ext
python src/pv_extension/train.py final --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --official_ext CHALLENGE/pv_extensions_labels_dev.csv   # -> src/pv_extension/inference/ext_net.pt
```

Leave-one-out over the 11 labelled extensions: 4 found (median date error 5 days), 1 false alarm on 36 negatives,
event F1 0.50 at the shipped threshold 2.5. Standalone use: `src/pv_extension/inference/HOWTO.md`.

### 4.4 Labels of the unlabelled series

`src/labeling/label.py` is the interactive tool with which team members marked PV start dates on heat-maps of the
unlabelled series. Their merged labels are in `data/manual_labels/` and `src/cnn_daymodel/labels/`; the CNN-inferred
labels of the remaining customers are in `data/cnn_inferenced_label.csv`. These label files are inputs to 4.1 to 4.3.

## 5. Data and tooling disclosure

- Inference uses only the provided consumption series and their timestamps.
- Development labels and the unlabelled series were used for training; the unlabelled series were partly labelled
  by hand.
- Public PV generation data (`openclimatefix/uk_pv`) was used only to synthesise training examples for the kWp
  network; it is not shipped and not used at inference.
- LLM coding assistants were used to write and document code. No LLM is part of the inference pipeline. During
  development, a coding assistant was shown the development labels file and a few raw rows of the dev time series;
  afterwards it was given aggregate statistics only.
