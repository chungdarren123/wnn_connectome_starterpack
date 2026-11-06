#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Users/chung/wnn_connectome_starterpack
TRAIN_PID=8050
PYTHON=/home/chung/.venvs/connectome/bin/python

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  sleep 60
done

test -f "$ROOT/experiments/models/final_gru_h64_two_seed_all_train_v1/selection.json"
"$PYTHON" "$ROOT/scripts/evaluate_final_h64_on_validation.py"
"$PYTHON" "$ROOT/scripts/package_final_h64_submission.py"
