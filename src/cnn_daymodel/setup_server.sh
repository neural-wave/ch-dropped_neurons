#!/usr/bin/env bash
# Creates cnn_daymodel/.venv and installs the requirements, then runs the environment check.
# Uses uv when installed (fast, no apt needed), otherwise python3 -m venv.
# Usage:  cd ~/Repositories/AIL_data_for_neural_wave_v1.1/cnn_daymodel && bash setup_server.sh
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  echo "== using uv $(uv --version)"
  [[ -x .venv/bin/python ]] || uv venv .venv --python 3.12
  uv pip install --python .venv/bin/python -r requirements.txt
else
  echo "== uv not found, using python3 -m venv (install uv with: curl -LsSf https://astral.sh/uv/install.sh | sh)"
  if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv || {
      echo "python3 -m venv failed. Either:  sudo apt install python3-venv   or install uv (see above) and re-run."; exit 1; }
  fi
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
# On Linux the default PyPI torch wheel ships with CUDA 12.x, so no special index is needed.
echo; .venv/bin/python check_env.py
