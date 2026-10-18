# kWp inference — how to use

Estimates the **nominal installed PV capacity (kWp)** of a customer from its 15-minute grid-withdrawal data only.
It does not detect PV or date the installation: it takes those from a detection model (in the submission,
`src/pv_submission_combined`, they come from `src/cnn_daymodel`)
and fills in the capacity.

## Files

| File | What it is |
|---|---|
| `inference.py` | Everything needed to predict: data reading, features, network, command line |
| `kwp.pt` | Trained weights + global calibration factor + window length |
| `HOWTO.md` | This file |

## Requirements

Python 3.10+ with `numpy`, `pandas`, `torch` (CPU is fine; a GPU is used automatically if present).

```bash
pip install numpy pandas torch
```

## Usage

```bash
python inference.py consumption.csv detections.csv --out predictions.csv
```

Options: `--weights path/to/kwp.pt` (default: `kwp.pt` next to `inference.py`), `--out` (default `predictions.csv`).

## Input 1 — `consumption.csv`

The challenge's time-series format. Separator `;` or `,` (detected automatically). Extra columns are ignored.

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | string | Customer identifier (kept as text, leading zeros preserved) |
| `energy_consumption_kwh` | decimal | Energy withdrawn from the grid in the 15-minute interval |
| `timestamp_epoch` | integer | Unix time of the **end** of the interval |

`timestamp_local` may be present but is not needed: the epoch timestamp avoids daylight-saving ambiguity.

## Input 2 — `detections.csv`

One row per customer, from the PV detection / dating model (same layout as the challenge `predictions.csv`):

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | string | Customer identifier |
| `pv_installed` | `TRUE`/`FALSE` | Whether the customer has PV |
| `pv_installation_date` | `YYYY-MM-DD` | Estimated commissioning date (empty if no PV) |

Other columns (e.g. an old `pv_capacity_kwp`) are ignored and overwritten.

## Output — `predictions.csv`

UTF-8, `;`-separated, one row per row of `detections.csv`, in the challenge format:

```
customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
POD13;TRUE;2024-02-05;181.3
POD48;FALSE;;
```

`pv_capacity_kwp` is filled for every `TRUE` customer that has a date and consumption data; it is empty otherwise
(the script reports how many PV customers were skipped).

## How it works (short)

1. Each customer's data becomes a day × 96-slot matrix (UTC, kW). A window of 56 days before and up to
   one year after the commissioning date is taken.
2. A small CNN reads each day's load curve; a transformer compares the days (before/after installation, seasons,
   sunny/cloudy days); attention pooling picks the informative days; the head outputs log(kWp).
3. The prediction is averaged over the commissioning date shifted by −14, −7, 0, +7, +14 days (robust to date
   errors) and multiplied by a calibration factor fitted on the labelled sites.

Training (see `../train.py`) used synthetic sites — PV-free AIL consumption minus real UK PV generation for a
random capacity, clipped at zero because the meter only records withdrawals — then fine-tuning on the 22
labelled PV sites. **At inference only the customer's own consumption and timestamps are used.**

## Accuracy

5-fold cross-validation on the 22 labelled PV sites (fine-tune and calibrate on 4 folds, predict the 5th;
two different fold splits averaged):

| Commissioning dates used | MAPE | Median APE | Challenge P score |
|---|---|---|---|
| True dates | 27.6% | 22.5% | 0.515 |
| LSTM out-of-fold dates (mean error 70 days) | 28.1% | 19.5% | 0.510 |

The model is almost insensitive to date errors. With 22 sites the uncertainty on these numbers is about ±1–2
points, and model choices were compared on the same sites, so treat them as optimistic.

## Limitations

- The capacity is that of the **initial** installation; extensions are not modelled.
- Errors shrink predictions toward a typical capacity/load ratio: very large PV relative to the load tends to be
  under-estimated, very small PV over-estimated.
- Validated on 22 sites only (27–959 kWp); small residential systems (< 20 kWp) were never seen with labels.
