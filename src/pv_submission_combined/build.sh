#!/bin/bash
# Refresh components/ from the source folders in src/ (run after retraining any component).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/.."
C="$HERE/components"
rm -rf "$C" && mkdir -p "$C/cnn_daymodel/model" "$C/kwp" "$C/pv_extension"

# 1-2. detection + date: the modules the saved model needs to load and run
cp "$SRC"/cnn_daymodel/{config,data,features,pipeline,ssl_model,cnn_model,seq_model}.py "$C/cnn_daymodel/"
cp "$SRC"/cnn_daymodel/model/{encoder.pt,downstream.joblib,fit_info.json} "$C/cnn_daymodel/model/"
# 3. kWp network
cp "$SRC"/kWp/inference/{inference.py,kwp.pt} "$C/kwp/"
# 4. extensions
cp "$SRC"/pv_extension/inference/{inference.py,ext_net.pt} "$C/pv_extension/"

echo "components refreshed:"; find "$C" -type f | sed "s|$HERE/||" | sort
