# PV detection from electricity consumption profiles

Team **Dropped Neurons**, AIL × Neural Wave challenge. For every customer the pipeline decides whether a photovoltaic
system is present, estimates its commissioning date and nominal capacity (kWp), and reports later capacity
extensions, using only the 15-minute grid-withdrawal series and its timestamps. No weather, irradiation or location
data is used, and no LLM is part of the pipeline.

## Repository layout

| Path | Content |
|---|---|
| `src/pv_submission_combined/` | **The submission.** `inference.py`, model artifacts, supporting modules, requirements. Self-contained. |
| `src/kWp/` | Training code of the kWp network shipped in the submission. |
| `src/pv_extension/` | Training code of a learned PV-extension detector (research line, not used by the submission). |
| `src/cnn_daymodel/` | Research line for presence and date (2-D CNN + day encoder); origin of the date design. |
| `src/labeling/` | Tool used to label the unlabelled series by hand. |
| `data/` | Manual labels of unlabelled customers and CNN-inferred labels used during training. No AIL time series. |
| `test/` | Scripts to test the submission: sample cutter, synthetic input generator, output validator, full check. |
| `requirements.txt` | Environment for running and testing from the repository root (includes the submission's pins). |
| `docs/TESTING.md` | Step-by-step runbook of every command in this README, with expected outputs. |

## 1. Running the submission

Requirements: Python 3.12, CPU only, no network access after installation. Inside `src/pv_submission_combined/`:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python inference.py path/to/test_set.csv
```

On Linux install the CPU torch wheel first: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`.

- **Input:** the challenge time-series CSV (`customer_id;energy_consumption_kwh;timestamp_local;timestamp_epoch`,
  `;` separated). Days with 92 or 100 intervals (daylight saving) are handled.
- **Output:** `predictions.csv` and `pv_extensions_predictions.csv` in the current working directory. UTF-8, `;`
  separated, `TRUE`/`FALSE`, dates `YYYY-MM-DD`, customer ids kept as strings, empty date and capacity when
  `pv_installed` is `FALSE`, exactly one row per customer in input order. The extension file holds one row per
  predicted extension with the total capacity after it, or only its header.
- **Runtime:** about 40 s for 40 customers on a laptop CPU. Deterministic.
- The script prints `[inference] N customers, M PV, K extensions`. A line starting with `[inference] WARNING:` on
  stderr means a component fell back to a simpler estimate (details in `src/pv_submission_combined/README.md`).

An optional second argument sets the output folder: `python inference.py test_set.csv out_dir`.

## 2. Testing the submission

All commands are run from the repository root in a Python 3.12 environment built from the root `requirements.txt`
(the same pins as the submission):

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`CHALLENGE` stands for the folder holding `energy_consumption_dev.csv`, `pv_labels_dev.csv` and
`pv_extensions_labels_dev.csv`.

**Full check in one command** (fresh venv in a scratch folder, synthetic input, random real sample, leading-zero ids,
the labelled extension customer, all 40 customers with checksum, format validation of every output):

```bash
python test/run_checks.py CHALLENGE
```

It ends with `ALL CHECKS PASSED` or stops at the first failure. Without the argument it runs the synthetic check only.

**Individual tools:**

| Command | Purpose |
|---|---|
| `python test/make_sample.py CHALLENGE/energy_consumption_dev.csv sample.csv 2026 [n]` | Cuts `n` (default 5) randomly chosen customers out of the dev file into a valid test CSV. Seed 2026 picks POD22, POD63, POD36, POD23, POD26. |
| `python test/make_mock.py mock_dir` | Writes a synthetic `test_set.csv` in the challenge schema (4 customers, 2 with PV, one with an extension, one id with leading zeros, real DST days) plus matching label files. Contains no AIL data. |
| `python test/validate.py test.csv out_dir labels.csv ext_labels.csv` | Checks both output files against every format rule of the challenge document (separator, encoding, header order, one row per customer, booleans, dates, empty fields) and computes the main score and extension bonus with the official formulas. |

Reference results of the shipped artifacts on the development data (all format checks pass):

| Input | Result |
|---|---|
| 5 random customers, seed 2026 | 4 PV detected, 0 errors; F1 1.000, D 0.975, P 0.859; main score 97.5 |
| All 40 labelled customers (in-sample) | 21 PV, 0 false positives, 1 miss; F1 0.977, D 0.765, P 0.730; main score 90.3; `predictions.csv` md5 `3ea24e409d5e27f57f562a7f85587746` |
| All 40, 5-fold × 5 out-of-fold (models refitted per fold) | F1 0.977, D 0.772, P 0.548; main score 87.7 |

The in-sample score is higher than the out-of-fold one because detection thresholds were fitted on these 40
customers; the out-of-fold figure is the realistic estimate.

## 3. Method

| Task | Method | Fitted on |
|---|---|---|
| Presence | Change-point gain of the daily profiles against a fitted threshold, or persistent midday zero readings (PV that predates the data); a bell-shape gate rejects changes that are not centred on noon. | the 40 labelled customers (threshold); unlabelled pool (gate) |
| Date | A frozen 1-D CNN day encoder turns each day into an embedding; a gradient-boosting model scores every day as "PV active"; the commissioning date is the change point of that score sequence. | encoder: label-free masked reconstruction on all series; day model: the 40 labelled customers |
| Capacity | Geometric mean of (a) a ridge regression on pre/post midday deficit features blended with a year-over-year peak deficit and (b) a neural network (1-D CNN per day + transformer over the 56 days before and 336 days after the date) with a calibration factor. | (a) the 40 labelled customers; (b) synthetic sites, then fine-tuned on the 22 labelled PV customers |
| Extensions | A second same-sign step at least 240 days after the commissioning date; the two strongest in the file are reported with capacity scaled from the initial kWp. | nothing (rule) |

Details, fallbacks and per-module notes: `src/pv_submission_combined/README.md` and `TEAM_NOTES.md`.

## 4. Reproducing the training

Artifacts shipped in `src/pv_submission_combined/model/`:

| File | Content | Training code |
|---|---|---|
| `kwp_net.pt` | kWp network weights, calibration factor, window length | `src/kWp/` (section 4.1) |
| `encoder_shipped.pt` | day encoder weights | label-free masked-autoencoder pretraining; the same design is in `src/cnn_daymodel/ssl_pretrain.py` (section 4.3) |
| `state.pkl` | detection thresholds, capacity ridge regressions, year-over-year table, day model | fitted by cross-validation on the 40 labelled customers; shipped as a frozen artifact, the fitting harness is not part of this repository |

### 4.1 kWp network (`src/kWp/`)

Trained on synthetic sites, then fine-tuned on the 22 labelled PV customers. A synthetic site is a PV-free period
of an unlabelled customer from which real PV generation per kWp, taken from the public dataset
`openclimatefix/uk_pv` (Sheffield Solar, CC-BY-4.0), times a random capacity is subtracted and clipped at zero. The
UK data is used only to build training examples; it is not shipped and not read at inference.

Requirements: Python 3.10+, `numpy pandas torch`, plus `polars` for the library step. A GPU is recommended for
pretraining. Paths default to `/workspace/data`; every one is an argument. `DATA` below is any working folder.

```bash
# UK PV library (once): download openclimatefix/uk_pv from Hugging Face; the script globs the HF cache snapshot
python src/kWp/build_pv_library.py --years 2022,2023,2024 --out DATA/pv_library_15min.npz
# daily consumption cache from both challenge time-series files
python src/kWp/train.py prepare --ts CHALLENGE/energy_consumption_dev.csv --ts CHALLENGE/energy_consumption_unlabelled_dev.csv --cons DATA/cons.npz
# pretrain on synthetic sites, cross-validate, fine-tune on all 22 -> src/kWp/inference/kwp.pt
python src/kWp/train.py pretrain --out pretrained.pt --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --cnn_labels data/cnn_inferenced_label.csv --library DATA/pv_library_15min.npz
python src/kWp/train.py cv       --init pretrained.pt --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --cnn_labels data/cnn_inferenced_label.csv --library DATA/pv_library_15min.npz
python src/kWp/train.py final    --init pretrained.pt --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --cnn_labels data/cnn_inferenced_label.csv --library DATA/pv_library_15min.npz
cp src/kWp/inference/kwp.pt src/pv_submission_combined/model/kwp_net.pt
```

`data/cnn_inferenced_label.csv` marks which unlabelled customers are PV-free (used to pick synthetic hosts).
Expected cross-validation: network alone MAPE 27.6 %, P 0.515; blended with the ridge estimate MAPE 23.1 %, P 0.588.
Standalone use: `python src/kWp/inference/inference.py consumption.csv detections.csv` (see `src/kWp/inference/HOWTO.md`).

### 4.2 Extension detector (`src/pv_extension/`, research line)

Weekly CNN + BiGRU classifying each week as nothing / PV installed / PV extended, trained on AIL data only, with the
manual extension labels in `data/manual_labels/extension_with_startdates.csv` and the official dev extension. Needs
the `cons.npz` cache from 4.1.

```bash
for K in $(seq 0 10); do python src/pv_extension/train.py loo --fold $K --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --official_ext CHALLENGE/pv_extensions_labels_dev.csv --out_dir DATA/ext; done
python src/pv_extension/train.py summary --out_dir DATA/ext
python src/pv_extension/train.py final --cons DATA/cons.npz --labels CHALLENGE/pv_labels_dev.csv --official_ext CHALLENGE/pv_extensions_labels_dev.csv   # -> src/pv_extension/inference/ext_net.pt
python src/pv_extension/inference/inference.py consumption.csv predictions.csv                                                                     # standalone use
```

The submission uses the rule-based extension step instead of this model.

### 4.3 Presence and date research line (`src/cnn_daymodel/`)

Complete alternative pipeline (2-D CNN for presence, self-supervised day encoder plus gradient boosting for the
date) with a full reproduction script. The three challenge CSVs are expected in `src/challenge_set/` or in
`PV_DATA_DIR`.

```bash
bash src/cnn_daymodel/setup_server.sh                   # venv from src/cnn_daymodel/requirements.txt
PV_DATA_DIR=CHALLENGE bash src/cnn_daymodel/run_all.sh  # cache -> 1-D encoder -> 2-D conv stack -> labels -> train/test -> model/ -> plots
```

Steps and single commands are documented in `src/cnn_daymodel/README.md`. About 15 minutes on one GPU after the
data cache exists.

### 4.4 Labels of the unlabelled series

`src/labeling/label.py` is the interactive tool with which two team members marked PV start dates on heat-maps of
the unlabelled series. Their merged labels are in `data/manual_labels/` and `src/cnn_daymodel/labels/`; the CNN-inferred
labels of the remaining customers are in `data/cnn_inferenced_label.csv`. These label files are inputs to 4.1 to 4.3.

## 5. Data and tooling disclosure

- Inference uses only the provided consumption series and their timestamps.
- Development labels and the unlabelled series were used for training; the unlabelled series were partly labelled
  by hand.
- Public PV generation data (`openclimatefix/uk_pv`) was used only to synthesise training examples for the kWp
  network. External datasets of other utilities were used only to check detection generalisation, never for fitting.
- LLM coding assistants were used to write and document code. No LLM is part of the inference pipeline, and AIL data
  was not uploaded to any external service.
