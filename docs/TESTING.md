# Runbook: test the submission, reproduce the results, retrain the models

**Part 1** is what the organizers do with the submission folder, then how we verify it before hand-in.
**Part 2** is how to reproduce our reported numbers and retrain every model in this repo.

**All our commands run from the repo main folder** (the folder that contains `src/`, `test/`, `docs/`). Open a
terminal there once and stay there; no command below changes directory. If you use conda (`(base)` in the prompt),
run `conda deactivate` first so the venv's Python is the one used.

Last verified: 2026-10-18 on macOS (Apple Silicon), Python 3.12.

---

# Part 1. The submission: `src/pv_submission_combined/`

The submission is **one folder**: `src/pv_submission_combined/`. The organizers receive only that folder (zipped or
copied), not this repo. Its `inference.py` does all four tasks (presence, date, kWp, extensions) and writes both
required files. Nothing outside the folder is needed at inference time.

## 1.1 What the organizers do (PDF section 4.4)

The PDF says the command is run **from inside the submission folder**. They unpack the folder, open a terminal in
it, and run:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python inference.py /path/to/test_set.csv
```

On Linux the CPU torch wheel goes first:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

What happens: `predictions.csv` and `pv_extensions_predictions.csv` are written into the **current working
directory**, UTF-8, `;` separated, `TRUE`/`FALSE`, dates `YYYY-MM-DD`, kWp with 2 decimals, empty date and kWp for
non-PV customers, one row per customer in file order. CPU only, no network, deterministic, about 30 to 40 s for
40 customers. Stdout ends with `[inference] N customers, M PV, K extensions`. Anything on stderr starting with
`[inference] WARNING:` means a fallback fired (missing package, encoder or kWp-network failure).

Pinned requirements: numpy 2.5.4, pandas 3.0.6, pyarrow 26.0.0, scikit-learn 1.9.1, scipy 1.18.1, torch 2.14.1.
Python 3.12 is required: `model/state.pkl` holds an sklearn model pickled with scikit-learn 1.9.1, and an older
version cannot load it (then every customer is written FALSE). If `python3.12` is missing: macOS
`brew install python@3.12`, Ubuntu `sudo apt install python3.12 python3.12-venv`.

`inference.py` also accepts an optional second argument, an output folder (`python inference.py test.csv out_dir`).
The organizers do not need it; we use it below to keep runs apart without changing directory.

## 1.2 Our verification before hand-in, command by command

Every command below is complete and copy-pasteable. The three paths used are written out for this machine; on
another machine replace them once:

| Meaning | Path used below |
|---|---|
| this repo | `/Users/robertskalvitis/Documents/repos/nw26_dropped_neurons` |
| organizers' data folder | `/Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set` |
| scratch folder for the test (anything outside the repo) | `/tmp/pvtest` |

### Step 0. THE ORGANIZERS' DRY RUN: hand-in zip, new env, their command, mock data

This is the most important check. It reproduces what the organizers do with the exact zip we hand in, using a
test file cut from the real dev data: 5 randomly drawn customers of the 40 labelled ones (seed 2026 gives
POD22, POD63, POD36, POD23, POD26; 4 with PV, 1 without), in the organizers' schema. Their labels sit beside it for
the check. (The synthetic `test/mock/test_set.csv` exists for machines without the data.)

**Preparation, one command** (whoever packages the submission runs it; it builds the zip without venvs, caches or
old outputs, unpacks it into `~/Downloads/organizer_test/` as the organizers would, cuts the 5-customer test file
from the dev data and copies the two label files beside it; change `2026` for another random draw):

```bash
O=$HOME/Downloads/organizer_test; R=/Users/robertskalvitis/Documents/repos/nw26_dropped_neurons; D=/Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set; rm -rf $O; mkdir -p $O; cd $R/src && zip -q -r $O/pv_submission_combined.zip pv_submission_combined -x '*/.venv*' '*__pycache__*' '*/predictions.csv' '*/pv_extensions_predictions.csv' && unzip -q $O/pv_submission_combined.zip -d $O && python3 $R/test/make_sample.py $D/energy_consumption_dev.csv $O/test_set.csv 2026 && cp $D/pv_labels_dev.csv $D/pv_extensions_labels_dev.csv $O/ && unzip -l $O/pv_submission_combined.zip | tail -1 && ls -a $O/pv_submission_combined
```

Must end with `23 files` and a listing of only `inference.py`, `model`, `pvsteps`, `README.md`, `requirements.txt`,
`TEAM_NOTES.md`. `~/Downloads/organizer_test/pv_submission_combined.zip` is the file to hand in.

**The organizers' commands** (PDF 4.4: inside the unpacked folder, new venv, requirements, one command). Run them
one by one in a terminal. If another venv is active, `deactivate` first.

```bash
cd ~/Downloads/organizer_test/pv_submission_combined
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference.py ../test_set.csv
```

Expected: the install takes a few minutes (a "new release of pip is available" notice is harmless); the last command
takes about 30 s, prints `[inference] 5 customers, 4 PV, 2 extensions`, nothing else, and exits 0 (`echo $?` prints
`0`). Any
`[inference] WARNING:` line means a fallback fired and is a failure. On Linux run
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu` before the requirements line.

**What they get**, in the folder the command was run in:

```bash
cat predictions.csv
cat pv_extensions_predictions.csv
```

Expected, byte for byte (seed 2026):

```
customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
POD22;TRUE;2024-10-11;143.03
POD63;FALSE;;
POD36;TRUE;2024-07-22;239.17
POD23;TRUE;2023-11-12;31.41
POD26;TRUE;2024-05-22;88.78
```

```
customer_id;extension_date;pv_capacity_after_extension_kwp
POD22;2025-06-08;434.97
POD26;2025-02-20;126.27
```

**Format check against the PDF rules and score against the real labels** (still in the same folder and venv):

```bash
python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/validate.py ../test_set.csv . ../pv_labels_dev.csv ../pv_extensions_labels_dev.csv
deactivate
```

Expected last line `RESULT: ALL FORMAT CHECKS PASSED`, with TP 4 FP 0 FN 0, F1 1.000, D 0.975, P 0.859, main score
97.51 (the two predicted extensions are false alarms, which never reduce the main score). `;` separator,
`TRUE`/`FALSE`, empty fields for the non-PV customer, both files in the folder where the command ran.

Verified 2026-10-18 on macOS: zip of 23 entries, fresh venv, the one-argument command on this 5-customer file,
exit 0, outputs byte-identical to the above, `ALL FORMAT CHECKS PASSED`.

Steps 1 to 7 below repeat the run on real data, from any directory, without `cd` and without activating anything.

### Step 1. Fresh copy of the submission, venv built the organizers' way

```bash
rm -rf /tmp/pvtest
mkdir -p /tmp/pvtest
cp -R /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/src/pv_submission_combined /tmp/pvtest/submission
rm -rf /tmp/pvtest/submission/.venv /tmp/pvtest/submission/.venv_test
python3.12 -m venv /tmp/pvtest/submission/.venv
/tmp/pvtest/submission/.venv/bin/python -m pip install --upgrade pip
/tmp/pvtest/submission/.venv/bin/python -m pip install -r /tmp/pvtest/submission/requirements.txt
/tmp/pvtest/submission/.venv/bin/python -c "import torch, sklearn, pandas, pyarrow; print(torch.__version__, sklearn.__version__, pandas.__version__, pyarrow.__version__)"
```

Expected last line: `2.14.1 1.9.1 3.0.6 26.0.0`. The install takes a few minutes (torch is large). On Linux run
`/tmp/pvtest/submission/.venv/bin/python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu`
before the requirements line. If `python3.12` is not found: macOS `brew install python@3.12`.

### Step 2. Mock input in the organizers' schema (no real data)

```bash
/tmp/pvtest/submission/.venv/bin/python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/make_mock.py /tmp/pvtest/mock
/tmp/pvtest/submission/.venv/bin/python /tmp/pvtest/submission/inference.py /tmp/pvtest/mock/test_set.csv /tmp/pvtest/run_mock
cat /tmp/pvtest/run_mock/predictions.csv
cat /tmp/pvtest/run_mock/pv_extensions_predictions.csv
/tmp/pvtest/submission/.venv/bin/python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/validate.py /tmp/pvtest/mock/test_set.csv /tmp/pvtest/run_mock /tmp/pvtest/mock/mock_labels.csv /tmp/pvtest/mock/mock_extensions_labels.csv
```

Expected (2026-10-18, 4 s): `[inference] 4 customers, 2 PV, 2 extensions`, then

```
customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
MOCK_NOPV;FALSE;;
MOCK_PV;TRUE;2022-09-15;9.37
0042;TRUE;2022-06-01;7.54
MOCK_SHORT;FALSE;;
```

and the validator ends with `RESULT: ALL FORMAT CHECKS PASSED` (F1 1.0, D 1.0, P 0.879). The mock is synthetic and
only proves the input/output contract.

### Step 3. Five random real customers

```bash
/tmp/pvtest/submission/.venv/bin/python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/make_sample.py /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/energy_consumption_dev.csv /tmp/pvtest/test_set_5.csv 2026
/tmp/pvtest/submission/.venv/bin/python /tmp/pvtest/submission/inference.py /tmp/pvtest/test_set_5.csv /tmp/pvtest/run_5
cat /tmp/pvtest/run_5/predictions.csv
cat /tmp/pvtest/run_5/pv_extensions_predictions.csv
/tmp/pvtest/submission/.venv/bin/python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/validate.py /tmp/pvtest/test_set_5.csv /tmp/pvtest/run_5 /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/pv_labels_dev.csv /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/pv_extensions_labels_dev.csv
```

The last number of the first command is the seed; `2026` picks POD22, POD63, POD36, POD23, POD26 (a different seed
gives a different draw; a further number after the seed sets how many customers). Expected, byte for byte:

```
[inference] 5 customers, 4 PV, 2 extensions
customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
POD22;TRUE;2024-10-11;143.03
POD63;FALSE;;
POD36;TRUE;2024-07-22;239.17
POD23;TRUE;2023-11-12;31.41
POD26;TRUE;2024-05-22;88.78
```

Validator: `ALL FORMAT CHECKS PASSED`, TP 4 FP 0 FN 0, F1 1.000, D 0.975, P 0.859, main score 97.51. The two
extensions it predicts (POD22, POD26) are false alarms; extension errors never reduce the main score.

### Step 4. Leading-zero customer ids (PDF: ids are strings, zeros preserved)

```bash
sed -e 's/^POD22;/00022;/' -e 's/^POD63;/00063;/' -e 's/^POD36;/0036;/' -e 's/^POD23;/023;/' -e 's/^POD26;/00026;/' /tmp/pvtest/test_set_5.csv > /tmp/pvtest/test_set_5_zeros.csv
/tmp/pvtest/submission/.venv/bin/python /tmp/pvtest/submission/inference.py /tmp/pvtest/test_set_5_zeros.csv /tmp/pvtest/run_zeros
cat /tmp/pvtest/run_zeros/predictions.csv
```

Expected: ids `00022`, `00063`, `0036`, `023`, `00026`, same dates and kWp as step 3. (If you used another seed in
step 3, adapt the five `sed` replacements to the customers it printed.)

### Step 5. The one labelled extension customer, POD13

```bash
grep -E '^(customer_id|POD13);' /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/energy_consumption_dev.csv > /tmp/pvtest/test_set_pod13.csv
/tmp/pvtest/submission/.venv/bin/python /tmp/pvtest/submission/inference.py /tmp/pvtest/test_set_pod13.csv /tmp/pvtest/run_pod13
cat /tmp/pvtest/run_pod13/predictions.csv
cat /tmp/pvtest/run_pod13/pv_extensions_predictions.csv
/tmp/pvtest/submission/.venv/bin/python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/validate.py /tmp/pvtest/test_set_pod13.csv /tmp/pvtest/run_pod13 /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/pv_labels_dev.csv /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/pv_extensions_labels_dev.csv
```

Expected: `POD13;TRUE;2024-01-30;145.92` (label 2024-02-05, 174: date 6 days off, kWp 16% off) and extension
`POD13;2025-02-23;168.39` (label 2025-11-13, 407: not matched, outside the 30-day window). Format valid.

### Step 6. All 40 customers, md5 against TEAM_NOTES

```bash
/tmp/pvtest/submission/.venv/bin/python /tmp/pvtest/submission/inference.py /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/energy_consumption_dev.csv /tmp/pvtest/run_all40
md5 -q /tmp/pvtest/run_all40/predictions.csv
md5 -q /tmp/pvtest/run_all40/pv_extensions_predictions.csv
/tmp/pvtest/submission/.venv/bin/python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/validate.py /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/energy_consumption_dev.csv /tmp/pvtest/run_all40 /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/pv_labels_dev.csv /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set/pv_extensions_labels_dev.csv
```

(Linux: `md5sum` instead of `md5 -q`.) Expected: `[inference] 40 customers, 21 PV, 2 extensions`, about 30 s,
md5 `3ea24e409d5e27f57f562a7f85587746` for predictions and `99b0e71ebf7e631849738e3efed020d5` for extensions
(the values in `src/pv_submission_combined/TEAM_NOTES.md`), validator `ALL FORMAT CHECKS PASSED`, TP 21 FP 0 FN 1,
F1 0.977, D 0.765, P 0.730, in-sample main score 90.28 (higher than the 87.72 out-of-fold figure, as expected since
the thresholds were fitted on these 40).

### Step 7. Hand-in folder clean, cleanup

```bash
ls -a /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/src/pv_submission_combined
```

Must show only `inference.py`, `model`, `pvsteps`, `README.md`, `requirements.txt`, `TEAM_NOTES.md`. Anything named
`.venv*`, `__pycache__`, `predictions.csv` or `pv_extensions_predictions.csv` must be deleted before zipping, e.g.

```bash
rm -rf /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/src/pv_submission_combined/.venv /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/src/pv_submission_combined/.venv_test
```

Finally `rm -rf /tmp/pvtest`.

Checklist:

- [ ] step 1 last line `2.14.1 1.9.1 3.0.6 26.0.0`
- [ ] steps 2, 3, 5, 6 end with `ALL FORMAT CHECKS PASSED`; no line starting with `[inference] WARNING:` anywhere
- [ ] step 4 keeps the leading zeros
- [ ] step 6 md5 values match
- [ ] step 7 folder clean
- [ ] open point for the organizer contact: PDF 4.1 text says semicolon separators (what we write, and what their
      own label file uses), but the PDF examples and the data README show comma headers

### What `test/validate.py` checks

Arguments, in order: the test CSV given to inference, the folder holding the two output files, the labels file, the
extension labels file. Each rule of PDF section 4.1 prints `OK` or `FAIL`:

- UTF-8 without BOM, LF endings, `;` separator, no commas anywhere (decimal point)
- exact header names and order of both files
- `predictions.csv`: exactly one row per customer of the test CSV, in file order, no duplicates
- `pv_installed` literally `TRUE` or `FALSE`; TRUE rows have a valid `YYYY-MM-DD` date and a positive decimal kWp;
  FALSE rows have both empty
- `pv_extensions_predictions.csv`: customer in the test set and predicted TRUE, valid date, positive capacity;
  a header-only file is valid

Then the per-customer table, the main score (PDF 4.2) and the extension bonus (PDF 4.3, ±30 day one-to-one matching).

## 1.3 Optional: the same seven steps as one Python command

`test/run_checks.py` runs steps 1 to 7 unattended and stops at the first failure. Any Python 3 can start it; it
finds Python 3.12 itself and needs no bash:

```bash
python /Users/robertskalvitis/Documents/repos/nw26_dropped_neurons/test/run_checks.py /Users/robertskalvitis/Downloads/AIL_data_for_neural_wave_v1.1/challenge_set
```

Options: `--seed 2026`, `--work /tmp/pvtest`. Without the data folder it runs the mock step only.

---

# Part 2. Reproduce the results and retrain the models

## 2.1 Which folder produces which part of the submission (verified 2026-10-18)

| Task | What ships in `src/pv_submission_combined/` | Trained where | Not in the submission |
|---|---|---|---|
| 1 presence | rule detector + bell gate, `pvsteps/detect_gate.py`, thresholds in `model/state.pkl` | C9v2 fitting code, **not in this repo** (see 2.2) | `src/cnn_daymodel` 2-D CNN detector (adds false positives); `src/pv_cnn` and `src/simple_lstm_model` (deleted from the working tree on 2026-10-18, see 2.6/2.7) |
| 2 date | day encoder + gradient-boosting day score + change point, `model/encoder_shipped.pt`, `state["ail"]["day_model"]` | same design as `src/cnn_daymodel`, but the shipped weights are a separate copy, **not** byte-identical to `src/cnn_daymodel/model/encoder.pt`; fitting code not in this repo | `src/cnn_daymodel/inference.py` |
| 3 kWp | 50/50 log blend of classical C9v2 kWp (`capacity.py`, `capacity_yoy.py`) and the kWp network `model/kwp_net.pt` | network: **`src/kWp/train.py`** (2.3); `src/kWp/inference/kwp.pt` is byte-identical to `model/kwp_net.pt`. Classical ridges: not in this repo | `src/kWp/inference/inference.py` standalone CLI |
| 4 extensions | classical second-step scan, `pvsteps/extensions_gated.py`, no fitted parameters | nothing to train | **`src/pv_extension`** CNN + BiGRU (`ext_net.pt`) is not wired in (2.4) |

## 2.2 Reproduce the submission's reported scores

The numbers in `src/pv_submission_combined/TEAM_NOTES.md` (main score 87.72, F1 0.977, D 0.772, P 0.548 on 5 folds
x 5 repeats, plus the EKZ and HEAPO external checks) came from a cross-validation harness called `train_pipeline.py`
that **was never committed to this repo**. Consequently:

- `model/state.pkl` (detector thresholds, capacity ridges, year-over-year table, day model) and
  `model/encoder_shipped.pt` cannot be refitted from this repo. They are frozen artifacts.
- What can be reproduced today is the **in-sample run on the 40 dev customers** (section 1.2 step 6, md5 check) and the kWp
  network (2.3). The out-of-fold scores cannot be regenerated until `train_pipeline.py` is recovered and added under
  `src/pv_submission_combined/`.

Action for the team: locate `train_pipeline.py` (C9v2 freeze of 2026-10-17, tree sha256 `a7133f67217aaeb1` per
TEAM_NOTES) and commit it with its data paths, or document the loss.

## 2.3 Task 3: retrain the kWp network (`src/kWp/`)

Trained on synthetic sites (PV-free AIL pool consumption minus public UK PV generation from `openclimatefix/uk_pv`,
Sheffield Solar, CC-BY-4.0), then fine-tuned on the 22 labelled PV customers. Script defaults assume a GPU server
with the data under `/workspace/data`; every path is an argument.

Environment: Python 3.10+ with `numpy pandas torch polars` (polars only for `build_pv_library.py`). A GPU is
strongly recommended for `pretrain` (12 000 steps).

```bash
# 0. one-off: download openclimatefix/uk_pv from Hugging Face (5-minutely parquet, years 2022-2024) so that it lands
#    under $HF_HOME/hub/datasets--openclimatefix--uk_pv/snapshots/<id>/5_minutely/year=YYYY/month=MM/data.parquet
#    build_pv_library.py globs that snapshot path; edit SNAP in the script if your HF cache is elsewhere.
python src/kWp/build_pv_library.py --years 2022,2023,2024 --out /workspace/data/pv_library_15min.npz

# 1. daily consumption cache from the two challenge time-series files (also used by src/pv_extension)
python src/kWp/train.py prepare --ts $DATA/energy_consumption_dev.csv --ts $DATA/energy_consumption_unlabelled_dev.csv \
    --cons /workspace/data/cons.npz

# 2. pretrain on synthetic sites only
python src/kWp/train.py pretrain --out pretrained.pt --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
    --cnn_labels data/cnn_inferenced_label.csv --library /workspace/data/pv_library_15min.npz

# 3. 5-fold x 2 cross-validation on the 22 labelled PV sites (the README table; add --eval_dates <csv> to score at estimated dates)
python src/kWp/train.py cv --init pretrained.pt --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
    --cnn_labels data/cnn_inferenced_label.csv --library /workspace/data/pv_library_15min.npz

# 4. fine-tune on all 22 and write the shipped weights
python src/kWp/train.py final --init pretrained.pt --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
    --cnn_labels data/cnn_inferenced_label.csv --library /workspace/data/pv_library_15min.npz
#    -> src/kWp/inference/kwp.pt
```

Then copy into the submission and re-run Part 1:

```bash
cp src/kWp/inference/kwp.pt src/pv_submission_combined/model/kwp_net.pt
```

`data/cnn_inferenced_label.csv` (CNN-inferred PV labels of the unlabelled pool, from `src/pv_cnn`, see 2.6) tells the
synthesiser which pool customers are PV-free. Standalone check of the network alone:
`python src/kWp/inference/inference.py consumption.csv detections.csv --out predictions.csv` (see `src/kWp/inference/HOWTO.md`).

Expected CV (README of the submission): network alone MAPE 27.6%, P 0.515; blended with C9v2 MAPE 23.1%, P 0.588.
GPU training is not bit-reproducible, so expect a point or two of MAPE drift.

## 2.4 Task 4: retrain the extension detector (`src/pv_extension/`, not yet in the submission)

Weekly CNN + BiGRU, 3 classes per week (nothing / PV installed / PV extended). AIL data only, plus the manual
extension labels in `data/manual_labels/extension_with_startdates.csv`. Needs the `cons.npz` cache from 2.3 step 1.

```bash
for K in $(seq 0 10); do                      # leave-one-out over the 11 manually labelled extension customers
  python src/pv_extension/train.py loo --fold $K --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
      --official_ext $DATA/pv_extensions_labels_dev.csv --out_dir /workspace/ext
done
python src/pv_extension/train.py summary --out_dir /workspace/ext                                   # aggregates folds, picks the threshold
python src/pv_extension/train.py final --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
    --official_ext $DATA/pv_extensions_labels_dev.csv --threshold 2.5             # -> inference/ext_net.pt
```

Standalone use after the main pipeline (`src/pv_extension/inference/HOWTO.md`):

```bash
python src/pv_extension/inference/inference.py consumption.csv predictions.csv    # writes pv_extensions_predictions.csv
```

Leave-one-out figures are in `assets/extension_model_predictions/`. To ship it, the team must replace the
`extensions_gated` call in `src/pv_submission_combined/inference.py` with this model and redo all of Part 1.

## 2.5 Tasks 1 and 2 research line: `src/cnn_daymodel/` (full reproduction script)

Separate end-to-end solution (2-D CNN presence + SSL day encoder + GBM date). Scores F1 0.977, D 0.611, main 79.8
on the 40 with its kWp fallback; kept as the origin of the date design. Expects the three challenge CSVs in
`src/challenge_set/` or `PV_DATA_DIR=$DATA`.

```bash
bash src/cnn_daymodel/setup_server.sh                      # .venv from requirements.txt (Python 3.12+, numpy >= 2, sklearn >= 1.9) + check_env.py
PV_DATA_DIR=$DATA bash src/cnn_daymodel/run_all.sh         # SEED=5 bash run_all.sh for another seed; log in artifacts/run_all.log
```

Both scripts change into their own folder themselves. `run_all.sh` steps (each skipped if its output exists; delete the file to redo):

1. `prepare_data.py`: CSVs to `cache/*.npz` (about 10 min)
2. `ssl_pretrain.py --epochs 30`: 1-D day encoder, masked reconstruction, no labels
3. `ssl2d_pretrain.py --image relsmooth`: 2-D conv stack, no labels
4. `build_human_labels.py labels/manual/annotator1.csv labels/manual/annotator2.csv --out labels/human_labels.csv`
5. `train_downstream.py --clf cnn --image relsmooth --cnn-pretrained artifacts/heatcnn_ssl_relsmooth.pt --train-labels labels/human_labels.csv --test-all-labelled --no-save --eval-tag cnn_seed2 --save-to model --seed 2`
6. `summarize_evals.py cnn_seed`
7. `plot_heatmap.py --all --relsmooth --pred artifacts/eval_holdout_cnn_seed2.json --outdir dates_seed2`

About 15 min on an RTX 4090 after the cache. Its own entry point (run from inside `src/cnn_daymodel`, it expects `model/` next to it):
`python inference.py $DATA/energy_consumption_dev.csv` then `python score_labels.py predictions.csv --model model`.

## 2.6 Labels for the unlabelled pool: `src/pv_cnn/` and `src/labeling/`

**Status 2026-10-18:** `src/pv_cnn/` and `src/simple_lstm_model/` were deleted in the working tree (uncommitted at the
time of writing). The commands below need the folder back: `git checkout b65c172 -- src/pv_cnn` (last commit that
has it). The outputs it produced are still committed: `data/cnn_inferenced_label.csv` and the human labels under
`src/cnn_daymodel/labels/`.

- `src/labeling/label.py`: Gradio tool (written for Colab) to click PV start dates on heatmaps of the unlabelled pool.
  Output `manual_labels.csv`; the merged team files are in `data/manual_labels/` (`manual_labels_combined.csv`,
  `extension_with_startdates.csv`).
- `src/pv_cnn/`: 2-D CNN trained on the manual labels, then used to label the rest of the pool:

```bash
pip install -r src/pv_cnn/requirements.txt           # paths in src/pv_cnn/config.yaml
bash src/pv_cnn/run_all.sh                            # cache -> human labels -> pretrain -> train -> label.py -> plots
```

  Produces `artifacts/labels_unlabelled.csv`; the committed result is `labels/cnn_labels_remaining.csv` and the
  repo-level `data/cnn_inferenced_label.csv` used by 2.3.

## 2.7 Baselines kept for the record

- `src/simple_lstm_model/` (deleted in the working tree on 2026-10-18; restore with `git checkout b65c172 -- src/simple_lstm_model`):
  notebook `pv_lstm.ipynb` (kernel `.venv/bin/python`, requirements in the folder),
  dev CV 85.81. `RECOMPUTE_STUDY=True` retrains the folds, `RETRAIN_FINAL=True` recreates `lstm_submission/artifacts`.
  `lstm_submission/inference.py` is a complete alternative entry point.
- `src/autoencoder.py`: Colab-only exploratory autoencoder (asserts CUDA).

## 2.8 Order for a full retrain from scratch

1. Manual labels (2.6, already in `data/manual_labels/`).
2. `src/pv_cnn` run_all to refresh `data/cnn_inferenced_label.csv` (optional, committed copy exists; folder must be
   restored from git first, see 2.6).
3. `src/kWp`: prepare, build library, pretrain, cv, final; copy `kwp.pt` into the submission (2.3).
4. `src/pv_extension`: loo x 11, summary, final (2.4). Integrate only if it beats the gated rule on the dev
   extension and on the leave-one-out figures.
5. C9v2 parts: blocked until `train_pipeline.py` is recovered (2.2).
6. Re-run all of Part 1 and update the md5 values in `TEAM_NOTES.md`, in `test/run_checks.py` and in section 1.2.
