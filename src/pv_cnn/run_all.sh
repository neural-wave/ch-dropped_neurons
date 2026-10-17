#!/usr/bin/env bash
# Whole pipeline: CSV -> daily profiles -> images -> pretraining -> CNN trained on human labels ->
# evaluation on the labelled customers -> labels for the remaining customers -> images and Grad-CAM.
# Usage:  bash run_all.sh [humanA.csv humanB.csv]      default: the files in labels/manual/
#         (python = .venv/bin/python if present, else python3; PV_CONFIG=<file> selects the config)
set -euo pipefail
cd "$(dirname "$0")"
[[ $# -ge 2 ]] || set -- labels/manual/annotator1.csv labels/manual/annotator2.csv
PY="${PY:-$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)}"
mkdir -p artifacts
exec > >(tee -a artifacts/run_all.log) 2>&1
echo "== $(date)  python: $PY"

[[ -f cache/labelled.npz && -f cache/unlabelled.npz ]] || { echo "== 1. CSV -> cache/*.npz"; $PY prepare_data.py; }
echo "== 2. human labels (customers where the humans agree; the labelled test customers are removed)"
$PY build_human_labels.py "$@"
echo "== 3. self-supervised pretraining of the conv stack (no labels)"
$PY pretrain.py --image relsmooth
echo "== 4. train on the human labels, evaluate on the labelled customers, save the model"
$PY train.py --train-labels artifacts/human_labels.csv --pretrained artifacts/pretrained_relsmooth.pt \
    --tag humans --save-to artifacts/model_humans
echo "== 5. label the remaining unlabelled customers"
$PY label.py --model artifacts/model_humans --exclude artifacts/human_labels.csv --out artifacts/labels_unlabelled.csv
echo "== 6. images and Grad-CAM for the labelled customers"
$PY plot_images.py --eval artifacts/eval_humans.json
$PY plot_attention.py --model artifacts/model_humans
echo "== summary"; $PY summarize_evals.py
echo "== done $(date)"
