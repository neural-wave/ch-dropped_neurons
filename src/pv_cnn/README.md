# PV detection from 15-minute electricity consumption with a 2-D CNN

Self-contained pipeline: consumption time series -> daily profiles -> days x hours image -> 2-D CNN ->
PV presence, commissioning date, (kWp) -> evaluation. Inputs to the model are the consumption values and,
optionally, the day of year derived from the timestamps. No weather, irradiation, location or sun-position
information is used anywhere.

## Inputs

Three files, named in `config.yaml` (`paths`):

* a consumption time series of the **labelled** customers (the test set),
* a consumption time series of the **unlabelled** customers (the training pool, labelled by humans),
* the **labels** of the labelled customers.

Time series columns: `customer_id`, `energy_consumption_kwh` (energy of a 15-minute interval), `timestamp_epoch`
(end of the interval, Unix seconds). Label columns: `customer_id`, `pv_installed` (TRUE/FALSE),
`pv_installation_date` (YYYY-MM-DD), `pv_capacity_kwp`. Human label files: the same plus `confidence` (0-100).

## Bundled labels and results

| path | content |
|---|---|
| `labels/manual/annotator1.csv`, `annotator2.csv` | the two independent manual annotations (customer_id, pv_installed, pv_installation_date, confidence) |
| `labels/human_labels.csv` | merged training labels: customers where the annotators agree, labelled test customers removed |
| `labels/human_labels_conf90.csv` | the same restricted to annotator confidence >= 90 |
| `labels/cnn_labels_remaining.csv` | CNN labels for every unlabelled customer not in `human_labels.csv` (label, date, probability, label confidence) |
| `results/eval_test_run1.json` | evaluation of the trained CNN on the labelled customers (score, F1, D, P, per-customer table) |

## Pipeline

| step | file | in -> out |
|---|---|---|
| 1 | `data.py`, `prepare_data.py` | CSV rows -> per customer a float32 matrix `X[L, 96]` (L calendar days x 96 fifteen-minute slots, local time) and the day index of every row. Optional id-prefix filter, null ids dropped, the missing DST hour interpolated and the duplicated one averaged, days with fewer than `min_valid_slots` readings dropped. Cached in `cache/*.npz`. |
| 2 | `images.py` | `X` -> the image the CNN reads. `relsmooth` = log2(kW) minus the day's own shoulder level (06-09h and 17-20h), 15-day x 3-slot median filter, clipped to +-3, scaled to [-1, 1]; negative = less grid draw at midday than at the shoulders, the PV signature. Plus day-of-year sine/cosine channels (`image.calendar`, can be switched off) and a zero mask channel. No learned parameters. |
| 3a | `pretrain.py` | self-supervised pretraining of the conv stack on every customer's image, no labels: random 12-day x 12-slot patches (40% of a 180-day crop) are hidden and reconstructed (MSE). 3000 steps, batch 48, AdamW, one-cycle lr 1e-3. |
| 3b | `cnn_model.py` | the network and its training recipe (below). |
| 4 | `postprocess.py` | per-day PV scores -> commissioning date: 7-day smoothing, single upward mean-shift fit with at least 14 days per segment; PV predating the data -> first day. Capacity cues after the date -> kWp by ridge regression in log space, fitted only when training labels carry a kWp. |
| 5 | `pipeline.py`, `train.py`, `scoring.py` | `CNNPipeline` = steps 3b + 4. `train.py` trains on the human label file, evaluates on the labelled customers with the score `F1 * (70 + 15 D + 15 P)` (D: date accuracy within 30 days, P: capacity accuracy within 50%) and writes `artifacts/eval_<tag>.json`. |
| 6 | `label.py` | labels any customer set with a saved model: label, date, probability, label confidence. |
| 7 | `plot_images.py`, `plot_attention.py` | the images with true and predicted dates; Grad-CAM maps of where the CNN looks. |

## Network (`cnn_model.HeatCNN`, 83,859 parameters)

```
conv stack          Conv2d(in_ch->16, 5 days x 7 slots) GELU MaxPool(1x2)   slots 96 -> 48
                    Conv2d(16->32, 5x5)                 GELU MaxPool(1x2)   48 -> 24
                    Conv2d(32->32, 5x5)                 GELU MaxPool(1x4)   24 -> 6     -> (32, L, 6)
temporal head       Conv1d(192->32, 7 days) GELU Dropout(0.25)                          -> (32, L)
day head            Conv1d(32->1)                       one logit per day  "PV active on this day"
attention           Conv1d(32->1), softmax over days    which days the customer decision pools from
customer head       Linear(32->1) on the attention-weighted mean -> "customer has PV"
```
The day axis is never pooled: every day keeps its own feature vector, so the day head has daily resolution.
`in_ch` = image channels (1 for `relsmooth`) + 2 calendar channels (0 with `image.calendar: false`) + 1 mask channel.

Training: every epoch is one batch holding all training customers; a window length L in [120, 365] days is
drawn and each customer contributes a random window of L consecutive days. Window label = 1 only if it
contains a PV day (a window before the commissioning date is a negative example). Loss = BCE(customer) +
BCE(days). AdamW lr 2e-3, weight decay 1e-3, cosine schedule, 150 epochs, gradient clip 1.0, pretrained conv
stack at 0.2x lr. Inference: full series, threshold 0.5 on the customer probability; the date is the
change-point of the day scores. GPU training is not bit-for-bit deterministic (seed 42): expect small
differences between runs.

## Protocol

* Training set: the customers of the human label file (unlabelled-file customers annotated by two people,
  kept where the annotators agree; `build_human_labels.py`). Test set: all customers of the labelled file.
  The two sets share no customer; the labelled customers are never used for training.
* Capacity is not predicted when the training labels carry no kWp, so P = 0 in the score.
* Pretraining uses the images of all customers of both files, without labels.

## Configuration

All paths, data constants and hyperparameters live in `config.yaml` (sections paths, data, image, model, train,
pretrain, date). Command-line flags override them; `PV_CONFIG=other.yaml` selects another file.

## Run

```bash
pip install -r requirements.txt              # on Linux the PyPI torch wheel includes CUDA
python prepare_data.py                       # once: CSV -> cache/*.npz
bash run_all.sh                              # steps 2-7 on labels/manual/*.csv, log in artifacts/run_all.log
```
or step by step:
```bash
python build_human_labels.py labels/manual/annotator1.csv labels/manual/annotator2.csv   # -> artifacts/human_labels.csv
python pretrain.py --image relsmooth                                       # -> artifacts/pretrained_relsmooth.pt
python train.py --train-labels artifacts/human_labels.csv \
    --pretrained artifacts/pretrained_relsmooth.pt --tag humans --save-to artifacts/model_humans
python label.py --model artifacts/model_humans --exclude artifacts/human_labels.csv   # -> artifacts/labels_unlabelled.csv
python plot_images.py --eval artifacts/eval_humans.json                    # -> plots/images/labelled/
python plot_attention.py --model artifacts/model_humans                    # -> plots/attention/labelled/
python summarize_evals.py
```
Variant without calendar channels: `image.calendar: false` in `config.yaml`, or `--calendar false` on
`pretrain.py` and `train.py` (use `artifacts/pretrained_relsmooth_nocal.pt`).
