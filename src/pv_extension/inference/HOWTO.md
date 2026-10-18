# PV extension detection — how to use

Finds **PV capacity extensions** (an existing PV system made bigger) in 15-minute grid-withdrawal data and writes the
challenge file `pv_extensions_predictions.csv`: one row per extension with its date and the total capacity after it.
It uses the customer's own consumption and timestamps only.

## Files

| File | What it is |
|---|---|
| `inference.py` | Everything needed to predict: data reading, features, network, event logic, command line |
| `ext_net.pt` | Trained weights and the decision threshold (1 MB) |
| `HOWTO.md` | This file |

## Requirements

Python 3.10+ with `numpy`, `pandas`, `torch`. Runs on CPU (about 10 s for 40 customers); a GPU is used if present.

```bash
pip install numpy pandas torch
```

## Usage

Recommended: run after the main pipeline and give it the main `predictions.csv`.

```bash
python inference.py consumption.csv predictions.csv
```

| Variant | Command | Behaviour |
|---|---|---|
| With main predictions (recommended) | `python inference.py consumption.csv predictions.csv` | Only customers with `pv_installed=TRUE`; capacity after extension filled from their initial kWp; extensions within 60 days of the predicted installation date are dropped (that change is the installation itself) |
| Keep events at the install date | `... predictions.csv --allow-at-install` | As above, but keeps extensions even right at the predicted installation date (see "Label conflict" below) |
| Stand-alone | `python inference.py consumption.csv` | All customers; dates only, capacity left empty |

Options: `--out FILE` (default `pv_extensions_predictions.csv` in the current directory), `--weights FILE`
(default `ext_net.pt` next to `inference.py`).

## Input 1 — `consumption.csv`

The challenge time-series file. Separator `;` or `,` (detected automatically); extra columns are ignored.

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | string | Customer identifier (kept as text, leading zeros preserved) |
| `energy_consumption_kwh` | decimal | Energy withdrawn from the grid in the 15-minute interval |
| `timestamp_epoch` | integer | Unix time of the **end** of the interval |

## Input 2 (optional) — `predictions.csv`

The main-task output (`customer_id;pv_installed;pv_installation_date;pv_capacity_kwp`), e.g. the `predictions.csv`
of the submission (`src/pv_submission_combined`). Used for: which customers have PV, their installation date, and
their initial kWp.

**In the submission** this model is called directly by `src/pv_submission_combined/inference.py` (task 4), with the
presence and dates of `src/cnn_daymodel` and the kWp of `src/kWp`; you only need this standalone script to test the
extension model on its own.

## Output — `pv_extensions_predictions.csv`

UTF-8, `;`-separated, header always written (no extensions = header only):

```
customer_id;extension_date;pv_capacity_after_extension_kwp
DEMO_001;2024-06-03;18.50
```

- `extension_date`: `YYYY-MM-DD`; at most 2 extensions per customer, at least 26 weeks apart.
- `pv_capacity_after_extension_kwp`: **total** capacity after the extension (not the added part); empty when no
  `predictions.csv` was given.

## How it works

1. Each customer's series becomes a sequence of **weeks**. A week is its mean daily load curve divided by the
   customer's night load (96 slots), plus the share of zero readings per slot, the share of days measured and the season.
2. A small CNN encodes each week; a bidirectional GRU reads the whole sequence; every week gets three probabilities:
   **nothing / PV installed / PV extended**. The separate "installed" class is what stops a change from *no PV* to
   *PV* being reported as an extension.
3. An extension is reported where p(extended), summed over the week and its two neighbours (score 0–3), peaks above
   the threshold **2.5**, and is not outweighed by p(installed) there. The date is then refined to the day with the
   largest rise of a daily PV-strength index (night load minus midday load) within ±14 days.
4. Capacity after = previous capacity × growth of that PV-strength index (same season one year earlier vs the 90 days
   after), clipped to 1.1–4×; 1.5× when it cannot be measured.

Training (see `../train.py`) used AIL data only: the 11 manually labelled extensions plus the official dev extension,
the dev customers as negatives, and synthetic examples made from AIL customers (their own PV output amplified after a
random date; a donor's PV output added to a customer without PV).

## Validation

Leave-one-out over the 11 manually labelled extension customers, with the 36 other dev customers spread over the
folds as negatives (scored as the challenge does: a match is within ±30 days):

| Threshold | Found | False alarms | Missed | Event F1 |
|---|---|---|---|---|
| 1.5 | 4 | 4 | 7 | 0.42 |
| **2.5 (shipped)** | **4** | **1** | **7** | **0.50** |

- When an extension is found, the date is good: median error **5 days**.
- **No false alarms on customers without PV** (19 of 19); on 17 real installations, none is reported as an extension
  at threshold 2.5.
- Missed extensions: in 5 of the 7, the year before the labelled date shows **no midday PV dip at all**, and the model
  reads the change as an installation (p(installed) 0.84–0.99).

With 11 cases these figures are rough. The capacity-after estimate is not validated (only one labelled value exists).

## Label conflict to know about

For 3 dev customers, the manual labels put an extension on the exact date that the **official** labels call the
**initial installation**. Extensions within 60 days after the installation date in `predictions.csv` are dropped
unless `--allow-at-install` is given, so the outcome depends on which installation date the main pipeline predicts:

- With the submission's presence/date model (`src/cnn_daymodel`, trained on the team's labels), the installation is
  dated well before those extensions, so they are kept: on the 40 dev customers 3 extensions are reported, 1 matches
  the official label and 2 are false alarms under the official labels.
- With a date model that follows the official labels, the installation falls on the extension date and they are
  dropped (1 extension reported); `--allow-at-install` keeps them (4).

## Retraining

```bash
cd ..   # src/pv_extension
python train.py loo --fold K     # K = 0..10, leave-one-out folds (can run 4 in parallel on one 24 GB GPU)
python train.py summary          # detection / threshold table
python train.py final            # retrain on everything -> inference/ext_net.pt
```

It needs the prepared consumption cache (`src/kWp/train.py prepare`), the dev labels and
`data/manual_labels/extension_with_startdates.csv`.
