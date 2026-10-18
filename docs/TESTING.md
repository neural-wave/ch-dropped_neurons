# Runbook: test the submission, reproduce the results, retrain the models

**Part 1** is what the organizers do with the submission folder, then how we verify it before hand-in.
**Part 2** is how to retrain every model in this repo and put it into the submission.

Set two variables once in the terminal; every command below uses them:

```bash
REPO=/path/to/nw26_dropped_neurons                                # the folder that contains src/, test/, docs/
DATA=/path/to/AIL_data_for_neural_wave_v1.1/challenge_set         # energy_consumption_dev.csv, pv_labels_dev.csv, pv_extensions_labels_dev.csv
```

If you use conda (`(base)` in the prompt), run `conda deactivate` first so the venv's Python is the one used.
`md5sum` is `md5 -q` on macOS.

Last verified: 2026-10-18 on Linux (Python 3.12, CPU).

---

# Part 1. The submission: `src/pv_submission_combined/`

The submission is **one folder**: `src/pv_submission_combined/`. The organizers receive only that folder (zipped or
copied), not this repo. Its `inference.py` does all four tasks and writes both required files:

| Task | Model (copied into `components/`) |
|---|---|
| 1. PV yes / no | `src/cnn_daymodel`: 2-D CNN on the customer's (days × hours) image |
| 2. Installation date | `src/cnn_daymodel`: day encoder + gradient-boosting day score, change point |
| 3. Initial kWp | `src/kWp`: per-day CNN + transformer around the installation date |
| 4. Extensions | `src/pv_extension`: weekly CNN + BiGRU (nothing / installed / extended) |

Nothing outside the folder is needed at inference time.

## 1.1 What the organizers do (PDF section 4.4)

From **inside the submission folder**:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python inference.py /path/to/test_set.csv
```

On Linux without a GPU, put the CPU torch wheel first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

What happens: `predictions.csv` and `pv_extensions_predictions.csv` are written into the **current working
directory**, UTF-8, `;` separated, `TRUE`/`FALSE`, dates `YYYY-MM-DD`, kWp with 2 decimals, empty date and kWp for
non-PV customers, one row per customer in file order. CPU is enough, no network, about 35 s for 40 customers. Stdout
has one `[inference]` line per step and ends with
`[inference] wrote predictions.csv (N rows, M PV) and pv_extensions_predictions.csv (K rows) in Ts`. A line with
`WARNING` means a step fell back (kWp: cnn_daymodel's peak rule; extensions: header-only file).

Requirements (`requirements.txt`): numpy ≥ 2, pandas ≥ 2.2, scikit-learn ≥ 1.9, scipy, joblib, torch. Python 3.12+ is
required: `components/cnn_daymodel/model/downstream.joblib` was pickled with numpy 2 / scikit-learn 1.9 and older
versions cannot load it. Tested with numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, torch 2.14.1.

`inference.py` takes exactly one argument, the test CSV (as the PDF requires).

## 1.2 Our verification before hand-in, command by command

Scratch folder used below: `/tmp/pvtest` (anything outside the repo).

### Step 0. The organizers' dry run: hand-in zip, new env, their command

Build the zip exactly as handed in (no venvs, caches or old outputs), unpack it as the organizers would, and cut a
5-customer test file from the dev data (seed 2026 gives POD22, POD63, POD36, POD23, POD26; 4 with PV, 1 without):

```bash
O=/tmp/organizer_test; rm -rf $O; mkdir -p $O
cd $REPO/src && zip -q -r $O/pv_submission_combined.zip pv_submission_combined -x '*/.venv*' '*__pycache__*' '*/predictions.csv' '*/pv_extensions_predictions.csv'
unzip -q $O/pv_submission_combined.zip -d $O
python3 $REPO/test/make_sample.py $DATA/energy_consumption_dev.csv $O/test_set.csv 2026
cp $DATA/pv_labels_dev.csv $DATA/pv_extensions_labels_dev.csv $O/
unzip -l $O/pv_submission_combined.zip | tail -1
ls -a $O/pv_submission_combined
```

Expected: `24 files` (about 7.0 MB unpacked, 6.0 MB zipped) and a listing of only `README.md`, `build.sh`,
`components`, `inference.py`, `requirements.txt`. `$O/pv_submission_combined.zip` is the file to hand in.

The organizers' commands, one by one (`deactivate` first if another venv is active):

```bash
cd $O/pv_submission_combined
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python inference.py ../test_set.csv
echo $?
```

Expected: the install takes a few minutes; the last command takes about 10 s, prints four `[inference]` lines ending
with `wrote predictions.csv (5 rows, 4 PV) and pv_extensions_predictions.csv (1 rows)`, no `WARNING`, and `echo $?`
prints `0`.

Check the outputs and score them against the real labels (same folder, same venv):

```bash
cat predictions.csv pv_extensions_predictions.csv
md5sum predictions.csv pv_extensions_predictions.csv
python $REPO/test/validate.py ../test_set.csv . ../pv_labels_dev.csv ../pv_extensions_labels_dev.csv
deactivate; cd $REPO
```

Expected md5: `predictions.csv` `87f6d0ca0ea01be5640707ef7f3ec3e4`, `pv_extensions_predictions.csv`
`2cc39e19542b9af571dae6e8e82b6a31`. Validator: `RESULT: ALL FORMAT CHECKS PASSED`, TP 4 FP 0 FN 0, F1 1.000,
D 0.742, P 0.716, main score 91.86; the one predicted extension is not a labelled one (extension errors never reduce
the main score).

Steps 1 to 7 repeat the run on more inputs without `cd` and without activating anything. Step 1.3 runs them all in
one command.

### Step 1. Fresh copy of the submission, venv built the organizers' way

```bash
rm -rf /tmp/pvtest && mkdir -p /tmp/pvtest
cp -R $REPO/src/pv_submission_combined /tmp/pvtest/submission
rm -rf /tmp/pvtest/submission/.venv*
python3.12 -m venv /tmp/pvtest/submission/.venv
PY=/tmp/pvtest/submission/.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -r /tmp/pvtest/submission/requirements.txt
$PY -c "import torch, sklearn, pandas, numpy; print(torch.__version__, sklearn.__version__, pandas.__version__, numpy.__version__)"
```

Expected last line: torch 2.x, scikit-learn ≥ 1.9, pandas ≥ 2.2, numpy ≥ 2 (verified: `2.14.1+cpu 1.9.1 3.0.6 2.5.4`).
The commands below run `inference.py` inside an empty folder each, because it writes to the working directory.

### Step 2. Mock input in the organizers' schema (no real data)

```bash
$PY $REPO/test/make_mock.py /tmp/pvtest/mock
mkdir -p /tmp/pvtest/run_mock && (cd /tmp/pvtest/run_mock && $PY /tmp/pvtest/submission/inference.py /tmp/pvtest/mock/test_set.csv)
cat /tmp/pvtest/run_mock/predictions.csv /tmp/pvtest/run_mock/pv_extensions_predictions.csv
$PY $REPO/test/validate.py /tmp/pvtest/mock/test_set.csv /tmp/pvtest/run_mock /tmp/pvtest/mock/mock_labels.csv /tmp/pvtest/mock/mock_extensions_labels.csv
```

Expected (about 4 s), byte for byte (md5 `1fb1d05da6f36ac94b8a0177786750c0` / `4c8aaa3feeec12a37ae88430ed905fe2`):

```
customer_id;pv_installed;pv_installation_date;pv_capacity_kwp
MOCK_NOPV;TRUE;2022-01-01;2.13
MOCK_PV;TRUE;2022-01-01;2.50
0042;TRUE;2022-03-01;3.42
MOCK_SHORT;TRUE;2023-01-01;2.13
customer_id;extension_date;pv_capacity_after_extension_kwp
MOCK_PV;2022-07-28;3.76
```

Validator: `RESULT: ALL FORMAT CHECKS PASSED`. The scores on the mock are poor (F1 0.667, D 0, P 0): its synthetic
series do not look like real customers to the CNN. The mock only proves the input/output contract (ids as strings,
`0042` keeps its zeros, DST days, empty fields).

### Step 3. Five random real customers

```bash
$PY $REPO/test/make_sample.py $DATA/energy_consumption_dev.csv /tmp/pvtest/test_set_5.csv 2026
mkdir -p /tmp/pvtest/run_5 && (cd /tmp/pvtest/run_5 && $PY /tmp/pvtest/submission/inference.py /tmp/pvtest/test_set_5.csv)
md5sum /tmp/pvtest/run_5/predictions.csv /tmp/pvtest/run_5/pv_extensions_predictions.csv
$PY $REPO/test/validate.py /tmp/pvtest/test_set_5.csv /tmp/pvtest/run_5 $DATA/pv_labels_dev.csv $DATA/pv_extensions_labels_dev.csv
```

The last number of the first command is the seed (2026 picks POD22, POD63, POD36, POD23, POD26; a further number
sets how many customers). Expected: `5 rows, 4 PV`, `1 rows` of extensions; md5 `87f6d0ca0ea01be5640707ef7f3ec3e4` /
`2cc39e19542b9af571dae6e8e82b6a31`; validator `ALL FORMAT CHECKS PASSED`, TP 4 FP 0 FN 0, F1 1.000, D 0.742,
P 0.716, main score 91.86.

### Step 4. Leading-zero customer ids (PDF: ids are strings, zeros preserved)

```bash
sed -e 's/^POD22;/00022;/' -e 's/^POD63;/00063;/' -e 's/^POD36;/0036;/' -e 's/^POD23;/023;/' -e 's/^POD26;/00026;/' /tmp/pvtest/test_set_5.csv > /tmp/pvtest/test_set_5_zeros.csv
mkdir -p /tmp/pvtest/run_zeros && (cd /tmp/pvtest/run_zeros && $PY /tmp/pvtest/submission/inference.py /tmp/pvtest/test_set_5_zeros.csv)
cut -d';' -f1 /tmp/pvtest/run_zeros/predictions.csv
diff <(cut -d';' -f2- /tmp/pvtest/run_5/predictions.csv) <(cut -d';' -f2- /tmp/pvtest/run_zeros/predictions.csv) && echo "same predictions"
```

Expected: ids `00022`, `00063`, `0036`, `023`, `00026` (zeros kept) and `same predictions`. (Another seed in step 3:
adapt the five `sed` replacements to the customers it drew.)

### Step 5. The one labelled extension customer, POD13

```bash
grep -E '^(customer_id|POD13);' $DATA/energy_consumption_dev.csv > /tmp/pvtest/test_set_pod13.csv
mkdir -p /tmp/pvtest/run_pod13 && (cd /tmp/pvtest/run_pod13 && $PY /tmp/pvtest/submission/inference.py /tmp/pvtest/test_set_pod13.csv)
md5sum /tmp/pvtest/run_pod13/predictions.csv /tmp/pvtest/run_pod13/pv_extensions_predictions.csv
$PY $REPO/test/validate.py /tmp/pvtest/test_set_pod13.csv /tmp/pvtest/run_pod13 $DATA/pv_labels_dev.csv $DATA/pv_extensions_labels_dev.csv
```

Expected: `1 rows, 1 PV`, `1 rows` of extensions; md5 `085bc736228608262728967eeb2a3ba1` /
`9998b6bc8052f9c8eab3c5049bab2854`; validator `ALL FORMAT CHECKS PASSED`, F1 1.000, D 0.000 (the installation is
dated early, as in the team's labels), P 0.888, main score 83.33; extension **matched**: F1_ext 1.000, D_ext 0.767,
P_ext 0.247, bonus 8.52.

### Step 6. All 40 customers, md5 against the reference

```bash
mkdir -p /tmp/pvtest/run_all40 && (cd /tmp/pvtest/run_all40 && $PY /tmp/pvtest/submission/inference.py $DATA/energy_consumption_dev.csv)
md5sum /tmp/pvtest/run_all40/predictions.csv /tmp/pvtest/run_all40/pv_extensions_predictions.csv
$PY $REPO/test/validate.py $DATA/energy_consumption_dev.csv /tmp/pvtest/run_all40 $DATA/pv_labels_dev.csv $DATA/pv_extensions_labels_dev.csv
```

Expected: `40 rows, 21 PV`, `3 rows` of extensions, about 35 s; md5 `32ccb951673530650a817af9ae08181f` /
`e6c913cd7b8920ca634fc6f62b1869cc` (the values in `test/run_checks.py` and the READMEs); validator
`ALL FORMAT CHECKS PASSED`, TP 21 FP 0 FN 1, F1 0.977, D 0.611, P 0.744, main score 88.23; extensions TP 1 FP 2,
F1_ext 0.500, bonus 4.26. P is in-sample (the kWp network was fine-tuned on these 40); the cross-validated P is
about 0.51, so the realistic main-score estimate is about 84.8.

### Step 7. Hand-in folder clean, cleanup

```bash
ls -a $REPO/src/pv_submission_combined
find $REPO/src/pv_submission_combined -name __pycache__ -o -name '.venv*' -o -name '*predictions.csv'
```

The first must show only `README.md`, `build.sh`, `components`, `inference.py`, `requirements.txt`; the second must
print nothing. Otherwise delete what it lists, e.g.
`find $REPO/src/pv_submission_combined -name __pycache__ -prune -exec rm -rf {} +`. Finally `rm -rf /tmp/pvtest`.

Checklist:

- [ ] step 0: 24 files, outputs md5 as listed, `ALL FORMAT CHECKS PASSED`
- [ ] steps 2, 3, 5, 6 end with `ALL FORMAT CHECKS PASSED`; no `WARNING` line anywhere
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

## 1.3 The same seven steps as one Python command

`test/run_checks.py` runs steps 1 to 7 unattended (fresh venv, organizers' one-argument command in empty folders,
md5 check on all 40) and stops at the first failure. Any Python 3 can start it; it finds Python 3.12 itself:

```bash
python $REPO/test/run_checks.py $DATA
```

Options: `--seed 2026`, `--work /tmp/pvtest`. Without the data folder it runs the mock step only. It treats any output
on stderr as a failure. Verified 2026-10-18: `ALL CHECKS PASSED`.

---

# Part 2. Retrain the models and update the submission

## 2.1 Which folder produces which part of the submission

| Task | What ships in `src/pv_submission_combined/components/` | Source and training |
|---|---|---|
| 1 presence | `cnn_daymodel/` code + `model/downstream.joblib` (2-D CNN classifier, threshold) | `src/cnn_daymodel/` (2.4) |
| 2 date | `cnn_daymodel/model/encoder.pt` (day encoder) + day model in `downstream.joblib` | `src/cnn_daymodel/` (2.4) |
| 3 kWp | `kwp/inference.py` + `kwp/kwp.pt` | `src/kWp/` (2.2) |
| 4 extensions | `pv_extension/inference.py` + `pv_extension/ext_net.pt` | `src/pv_extension/` (2.3) |

`src/pv_submission_combined/build.sh` copies the current code and weights of the three source folders into
`components/`. After retraining anything: run it, then redo Part 1 and update the md5 values (section 2.6).

```bash
bash $REPO/src/pv_submission_combined/build.sh
```

## 2.2 Task 3: retrain the kWp network (`src/kWp/`)

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

# 3. 5-fold x 2 cross-validation on the 22 labelled PV sites (add --eval_dates <csv> to score at estimated dates)
python src/kWp/train.py cv --init pretrained.pt --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
    --cnn_labels data/cnn_inferenced_label.csv --library /workspace/data/pv_library_15min.npz

# 4. fine-tune on all 22 and write the weights -> src/kWp/inference/kwp.pt
python src/kWp/train.py final --init pretrained.pt --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
    --cnn_labels data/cnn_inferenced_label.csv --library /workspace/data/pv_library_15min.npz
```

Expected CV: MAPE 27.6 %, P 0.515 at true dates; MAPE 28.1 %, P 0.510 at estimated dates. GPU training is not
bit-reproducible, so expect a point or two of drift. Then `build.sh` (2.1). Standalone check of the network:
`python src/kWp/inference/inference.py consumption.csv detections.csv --out predictions.csv`
(`src/kWp/inference/HOWTO.md`).

## 2.3 Task 4: retrain the extension detector (`src/pv_extension/`)

Weekly CNN + BiGRU, 3 classes per week (nothing / PV installed / PV extended). AIL data only, plus the manual
extension labels in `data/manual_labels/extension_with_startdates.csv`. Needs the `cons.npz` cache from 2.2 step 1.

```bash
for K in $(seq 0 10); do                      # leave-one-out over the 11 manually labelled extension customers
  python src/pv_extension/train.py loo --fold $K --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
      --official_ext $DATA/pv_extensions_labels_dev.csv --out_dir /workspace/ext
done                                          # 4 folds fit side by side on one 24 GB GPU
python src/pv_extension/train.py summary --out_dir /workspace/ext                     # aggregates folds, threshold table
python src/pv_extension/train.py final --cons /workspace/data/cons.npz --labels $DATA/pv_labels_dev.csv \
    --official_ext $DATA/pv_extensions_labels_dev.csv --threshold 2.5   # -> src/pv_extension/inference/ext_net.pt
```

Expected leave-one-out: 4 of 11 found (median date error 5 days), 1 false alarm on 36 negatives, event F1 0.50 at
threshold 2.5. Inspection figures of the folds: `assets/extension_model_predictions/`. Then `build.sh` (2.1).
Standalone use: `src/pv_extension/inference/HOWTO.md`.

## 2.4 Tasks 1 and 2: `src/cnn_daymodel/` (full reproduction script)

2-D CNN presence + self-supervised day encoder + gradient-boosting date. Trained on 218 hand-labelled customers of
the unlabelled pool, tested on the 40 (never used for training): F1 0.977, D 0.611. Expects the three challenge CSVs
in `src/challenge_set/` or `PV_DATA_DIR=$DATA`.

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

About 15 min on an RTX 4090 after the cache. Step 5 writes `src/cnn_daymodel/model/`; then `build.sh` (2.1). Its own
standalone entry point (tasks 1–2 only, from inside `src/cnn_daymodel`): `python inference.py $DATA/energy_consumption_dev.csv`
then `python score_labels.py predictions.csv --model model`.

## 2.5 Labels for the unlabelled pool: `src/pv_cnn/` and `src/labeling/`

**Status 2026-10-18:** `src/pv_cnn/` and `src/simple_lstm_model/` were deleted in the working tree. The commands
below need the folder back: `git checkout b65c172 -- src/pv_cnn` (last commit that has it). The outputs it produced
are still committed: `data/cnn_inferenced_label.csv` and the human labels under `src/cnn_daymodel/labels/`.

- `src/labeling/label.py`: Gradio tool (written for Colab) to click PV start dates on heatmaps of the unlabelled pool.
  Output `manual_labels.csv`; the merged team files are in `data/manual_labels/` (`manual_labels_combined.csv`,
  `extension_with_startdates.csv`).
- `src/pv_cnn/`: 2-D CNN trained on the manual labels, then used to label the rest of the pool:

```bash
pip install -r src/pv_cnn/requirements.txt           # paths in src/pv_cnn/config.yaml
bash src/pv_cnn/run_all.sh                            # cache -> human labels -> pretrain -> train -> label.py -> plots
```

  Produces `artifacts/labels_unlabelled.csv`; the committed result is `labels/cnn_labels_remaining.csv` and the
  repo-level `data/cnn_inferenced_label.csv` used by 2.2.

Baselines kept for the record: `src/simple_lstm_model/` (deleted in the working tree; restore with
`git checkout b65c172 -- src/simple_lstm_model`; dev CV 85.81) and `src/autoencoder.py` (Colab-only exploratory
autoencoder).

## 2.6 Order for a full retrain from scratch

1. Manual labels (2.5, already in `data/manual_labels/` and `src/cnn_daymodel/labels/`).
2. `src/pv_cnn` run_all to refresh `data/cnn_inferenced_label.csv` (optional, committed copy exists; folder must be
   restored from git first, see 2.5).
3. `src/cnn_daymodel`: run_all (2.4).
4. `src/kWp`: build library, prepare, pretrain, cv, final (2.2).
5. `src/pv_extension`: loo × 11, summary, final (2.3).
6. `bash src/pv_submission_combined/build.sh`, re-run all of Part 1, and update the md5 values in
   `test/run_checks.py`, in this runbook, in `README.md` and in `src/pv_submission_combined/README.md`.
