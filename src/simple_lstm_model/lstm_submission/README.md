# Offline AIL PV inference

A multi-task LSTM pipeline predicts PV presence, initial commissioning date and initial nominal capacity. It uses a 30-epoch event/capacity checkpoint, a 60-epoch presence checkpoint, and consumption-only refinement models.

Development cross-validation: **85.81/100**, F1 **0.9767**. The **95/100 target was not reached**. These are model-selection results on the 40 development customers, not a hidden-test result. Capacity estimates remain uncertain.

## Setup and execution

Verified with Python 3.12 on macOS ARM CPU:

```bash
python -m pip install -r requirements.txt
python inference.py path/to/test_set.csv
```

Exactly one positional argument is accepted. Weights load relative to `inference.py`; outputs go to the current working directory. After installation, inference needs no network, development labels, weather, irradiation, location data or LLM.

Input columns: `customer_id`, `energy_consumption_kwh`, `timestamp_local`, `timestamp_epoch`. Comma, semicolon and tab separators are detected. IDs stay strings. Epoch timestamps must be Unix seconds on a 15-minute grid and are authoritative for ordering and Swiss local calendar features. Duplicate ID/epoch readings are rejected; invalid/negative energy is masked. DST and missing coverage are represented explicitly.

## Required package

```text
inference.py
lstm_pipeline.py
pv_refinement.py
requirements.txt
README.md
artifacts/pv_lstm.pt
artifacts/pv_presence.pt
artifacts/pv_refiner.pt
```

The parent notebook can recreate all artifacts. Neural and refinement artifacts use plain tensor parameters and load with `weights_only=True`. No original consumption files, labels, feature caches, or out-of-fold diagnostics belong in the submission. The provided archive includes exactly the files listed above.

## Outputs

UTF-8 and semicolon separators, following the official PDF §4.1:

```text
customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
```

One row per input customer, TRUE/FALSE, dates in YYYY-MM-DD and capacity in kWp. Date and capacity describe the initial installation and are blank when FALSE.

The optional extension detector is not implemented. `pv_extensions_predictions.csv` has its header only:

```text
customer_id;extension_date;pv_capacity_after_extension_kwp
```

This is a retrospective model using complete customer histories. Dates are limited to observed coverage. Nominal capacity is inferred statistically from net withdrawals; it cannot be uniquely observed from the meter alone. Model/decoder choices were compared on development folds, so those reported scores can overestimate hidden-test generalization.
