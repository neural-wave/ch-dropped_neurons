# PV detection submission

One entry point for the four challenge tasks, using only the customer's 15-minute consumption and its timestamps
(no weather, irradiance or location data, no network access).

| Task | Model | Source folder |
|---|---|---|
| 1. PV yes / no | 2-D CNN on the customer's (days × hours) image, attention pooling over days | `src/cnn_daymodel` |
| 2. Installation date | self-supervised day encoder + gradient-boosting day score, change point | `src/cnn_daymodel` |
| 3. Initial kWp | per-day CNN + transformer around the installation date | `src/kWp` |
| 4. Extensions | weekly CNN + BiGRU: nothing / PV installed / PV extended per week | `src/pv_extension` |

## Setup (Python 3.12+)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Linux without a GPU, install the CPU torch wheel first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Run

```bash
python inference.py path/to/test_set.csv
```

Writes `predictions.csv` and `pv_extensions_predictions.csv` to the **current working directory**. CPU is enough
(about 30 s for 40 customers); a GPU is used when present.

## Input

The challenge time-series CSV (`;` or `,`): `customer_id`, `energy_consumption_kwh`, `timestamp_local`,
`timestamp_epoch` (end of the 15-minute interval).

## Output

UTF-8, `;`-separated, `TRUE`/`FALSE`, dates `YYYY-MM-DD`:

- `predictions.csv` — exactly one row per `customer_id` of the input:
  `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp` (date and kWp empty when `FALSE`)
- `pv_extensions_predictions.csv` — one row per predicted extension:
  `customer_id;extension_date;pv_capacity_after_extension_kwp` (total capacity after the extension); header only
  when there is none

## How the steps connect

1. `cnn_daymodel` decides PV yes / no and the installation date for every customer.
2. For each PV customer, the kWp network reads 56 days before and 336 days after that date and outputs the initial
   kWp (averaged over the date shifted by −14…+14 days, calibrated). If it fails for a customer, cnn_daymodel's
   peak-reduction rule is used instead.
3. For each PV customer, the extension model scores every week; extensions are reported where p(extended) is high
   (threshold 2.5 on a 0–3 score), at most 2 per customer, 26 weeks apart, and not within 60 days after the
   installation date of step 1. Capacity after = previous kWp × growth of a midday PV-strength index.

If the kWp or extension step fails entirely, the run still completes (fallback kWp / header-only extension file) and
prints a warning.

## Dev-set check (40 labelled customers, CPU)

| | Value | Note |
|---|---|---|
| Detection | 21 TP, 0 FP, 1 FN → **F1 0.977** | cnn_daymodel was trained without these 40 |
| Dates | **D 0.611**, 13 of 21 within 7 days, median error 1 day | idem |
| kWp | P 0.744, MAPE 18% | **in-sample** (the kWp network was fine-tuned on these sites); 5-fold CV gives P ≈ 0.51, MAPE ≈ 28% |
| Extensions | 3 predicted, 1 matches the official dev extension → bonus 4.3 (F1_ext 0.50, D_ext 0.767, P_ext 0.247) | the other 2 are on customers whose official label puts the initial installation at that date (see below) |

Main score of this run (official formulas, `test/validate.py`): **88.2**. Realistic estimate with the cross-validated
kWp: F1 0.977 × (70 + 15 × 0.611 + 15 × 0.51) ≈ **84.8**. Output md5 on the 40: `predictions.csv`
`32ccb951673530650a817af9ae08181f`, `pv_extensions_predictions.csv` `e6c913cd7b8920ca634fc6f62b1869cc`.

Full pre-hand-in check from the repository root: `python test/run_checks.py CHALLENGE_FOLDER` (fresh venv, organizers'
one-argument command on synthetic and real inputs, format validation, md5); step by step in `docs/TESTING.md`.

**Label conflict:** for 3 dev customers the team's manual labels see PV from the start of the data and an extension
later, while the official labels call that later date the initial installation. cnn_daymodel (trained on the team's
human labels) follows the manual view, so on such customers the official scorer would see a wrong installation date
and a false-alarm extension.

## Layout and rebuilding

```
inference.py        entry point (orchestrates the three components)
requirements.txt
build.sh            refreshes components/ from src/cnn_daymodel, src/kWp/inference, src/pv_extension/inference
components/
  cnn_daymodel/     config, data, features, pipeline, ssl_model, cnn_model, seq_model + model/ (encoder.pt, downstream.joblib)
  kwp/              inference.py + kwp.pt
  pv_extension/     inference.py + ext_net.pt
```

After retraining a component, run `./build.sh` to copy its current code and weights in. Training instructions are in
each source folder (`src/cnn_daymodel/README.md`, `src/kWp/train.py`, `src/pv_extension/inference/HOWTO.md`).
