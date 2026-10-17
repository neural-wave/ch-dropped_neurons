#!/usr/bin/env bash
# Full reproduction: cache -> SSL pretraining (1-D encoder + 2-D conv stack) -> training labels ->
# train the CNN + day model on the human-labelled customers, test on the 40 labelled customers ->
# save the model to model/ -> summary table -> dates overview plot.
# Usage:  cd cnn_daymodel && bash run_all.sh            (SEED=2 by default; SEED=5 bash run_all.sh for another seed)
# Data:   ../challenge_set by default, or PV_DATA_DIR=<folder with the three challenge CSVs>
# Steps whose output already exists are skipped (delete the file to redo it). Log: artifacts/run_all.log
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-$([[ -x .venv/bin/python ]] && echo .venv/bin/python || echo python3)}"
SEED="${SEED:-2}"
mkdir -p artifacts cache plots model
exec > >(tee -a artifacts/run_all.log) 2>&1
echo "== $(date)  python: $PY  seed: $SEED"

echo "== 0. environment and data check"
$PY check_env.py || { echo "fix the MISSING items above (model files are not needed for training)"; }

if [[ -f cache/labelled.npz && -f cache/unlabelled.npz ]]; then echo "== 1. cache present"
else echo "== 1. prepare_data.py (CSVs -> cache/*.npz, ~10 min)"; $PY prepare_data.py; fi

if [[ -f artifacts/encoder.pt ]]; then echo "== 2. artifacts/encoder.pt present"
else echo "== 2. ssl_pretrain.py: 1-D day encoder, masked reconstruction, 30 epochs, no labels"; $PY ssl_pretrain.py --epochs 30; fi

if [[ -f artifacts/heatcnn_ssl_relsmooth.pt ]]; then echo "== 3. artifacts/heatcnn_ssl_relsmooth.pt present"
else echo "== 3. ssl2d_pretrain.py: 2-D conv stack, masked-patch reconstruction on relsmooth images, no labels"; $PY ssl2d_pretrain.py --image relsmooth; fi

if [[ -f labels/human_labels.csv ]]; then echo "== 4. labels/human_labels.csv present"
else echo "== 4. build_human_labels.py: merge the annotator files (agreeing customers only, the 40 test customers removed)"
     $PY build_human_labels.py labels/manual/annotator1.csv labels/manual/annotator2.csv --out labels/human_labels.csv; fi

echo "== 5. train on the human labels only, test on all 40 labelled customers, save to model/"
$PY train_downstream.py --clf cnn --image relsmooth --cnn-pretrained artifacts/heatcnn_ssl_relsmooth.pt \
    --train-labels labels/human_labels.csv --test-all-labelled --no-save \
    --eval-tag cnn_seed$SEED --save-to model --seed $SEED

echo "== 6. summary of artifacts/eval_*.json"
$PY summarize_evals.py cnn_seed

echo "== 7. dates overview of the 40 labelled customers -> plots/dates_seed$SEED/overview_all40.png"
$PY plot_heatmap.py --all --relsmooth --pred artifacts/eval_holdout_cnn_seed$SEED.json --outdir dates_seed$SEED
echo "== done $(date)"
