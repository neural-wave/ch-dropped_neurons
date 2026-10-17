# PV detection: inference (C9v2 + kWp network)

Detects behind-the-meter PV from 15-minute grid-withdrawal series, then estimates the commissioning date and the nominal
kWp (DC, sum of panel ratings), and reports later capacity extensions. At inference the only input is the consumption
file: no network access, no LLM, no weather, irradiance or location data.

This is the C9v2 pipeline with one change: the kWp is the 50/50 log-space blend of the C9v2 estimate and a second,
independent estimate from a neural network. Detection, dates and extensions are exactly those of C9v2.

## Setup (Python 3.12)

```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Linux install the CPU torch wheel first: `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`.

## Run

```
python inference.py path/to/test_set.csv [out_dir]
```

Runs on CPU, deterministic, about 40 s for 40 customers.

## Input

The `;`-separated time-series CSV of the challenge:

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | string | Customer identifier (kept as text, leading zeros preserved) |
| `energy_consumption_kwh` | decimal | Energy withdrawn from the grid in the 15-minute interval |
| `timestamp_local` | local datetime | Read but not used |
| `timestamp_epoch` | integer | Unix time of the end of the interval (used for ordering, days and slots) |

## Output

Written to `out_dir` (default: current directory), UTF-8, `;`, `TRUE`/`FALSE`, dates `YYYY-MM-DD`, kWp with 2 decimals:

- `predictions.csv`: one row per customer of the input, in file order.
  `customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`, date and kWp empty when `pv_installed` is `FALSE`.
- `pv_extensions_predictions.csv`: one row per predicted extension,
  `customer_id;extension_date;pv_capacity_after_extension_kwp` (total capacity after the extension). May hold only its
  header.

A customer that cannot be processed is written as `FALSE`. Warnings (without customer ids) go to stderr.

## Method

- **Detection:** PV if the classical change-point gain reaches a fitted threshold, or if midday readings are at zero often
  enough in some 56-day window (PV that predates the data; days with zero night load are outages and are ignored). A bell
  gate drops detections whose post/pre change is not bell-shaped around noon (`pvsteps/detect_gate.py`).
- **Date:** change point of the day-score sequence from a frozen 1D-CNN day encoder plus a gradient-boosting day model
  (fallback: the classical date).
- **kWp**, geometric mean of two estimates at the detected date:
  1. C9v2: pre/post deficit ridge regression (unclipped solar hours, edge of the midday zero run, midday zero share)
     blended with a year-over-year peak deficit (`pvsteps/capacity.py`, `capacity_yoy.py`).
  2. Network (`pvsteps/kwp_net.py`): a 1D-CNN reads each day's load curve, a transformer compares the 56 days before
     and 336 days after the date, attention pooling and a head give log kWp; averaged over the date shifted by
     −14 … +14 days and multiplied by a fitted calibration factor.
  If the network cannot score a customer, the C9v2 kWp is used alone.
- **Extensions:** a second same-sign step at least 240 days after the first; the two strongest in the file are reported
  (capacity from the C9v2 base kWp).

## Training data used (disclosure)

- C9v2 parts: the 40 labelled customers and the unlabelled pool only.
- kWp network: pretrained on synthetic sites, then fine-tuned on the 22 labelled PV customers. A synthetic site is a
  PV-free period of a pool customer from which real PV generation per kWp, from the public UK dataset
  `openclimatefix/uk_pv` (Sheffield Solar, CC-BY-4.0), times a random capacity is subtracted and clipped at zero. This
  UK generation data is used **only to build training examples**; it is not shipped and not read at inference.
  Training code: `nw26_dropped_neurons/src/kWp/train.py` and `build_pv_library.py`.

## Accuracy (kWp, development data)

5-fold cross-validation on the 22 labelled PV customers, 2 repeats, both models refitted inside the same folds:

| kWp model | Dates used | MAPE | Median APE | P |
|---|---|---|---|---|
| C9v2 | true | 26.0% | 19.3% | 0.524 |
| Network | true | 27.6% | 22.5% | 0.515 |
| **Blend (shipped)** | true | **23.1%** | **15.8%** | **0.588** |
| C9v2 | LSTM estimates (70 days mean error) | 24.6% | 22.9% | 0.527 |
| Network | LSTM estimates | 28.1% | 19.5% | 0.510 |
| **Blend (shipped)** | LSTM estimates | **23.3%** | **16.7%** | **0.570** |

The blend weight (50/50) was fixed in advance, not tuned. With 22 customers these figures carry about ±2 points of noise
and model choices were made on the same customers, so treat them as optimistic. Detection and dating scores are those of
C9v2 (see `TEAM_NOTES.md`).

## Contents

- `inference.py`: entry point, per-customer fallbacks, kWp blend
- `pvsteps/`: load and 96-slot grid, encoder, day features and score, date, detection, kWp (C9v2: two files; network:
  `kwp_net.py`), extensions, writer
- `model/state.pkl`: C9v2 fitted parameters (dict of scalars, small arrays, one sklearn model)
- `model/encoder_shipped.pt`: C9v2 day-encoder weights
- `model/kwp_net.pt`: kWp network weights, calibration factor and window length
