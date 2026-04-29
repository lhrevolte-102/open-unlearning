#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/private/envs/open-unlearning-npu-venv/bin/activate
unset PYTHONPATH

python scripts/build_infocurl_grid.py

bash scripts/run_infocurl_grid_worker.sh 0 2 0 &
PID0=$!
bash scripts/run_infocurl_grid_worker.sh 1 2 1 &
PID1=$!

wait "$PID0"
wait "$PID1"

python scripts/summarize_tofu10_runs.py
python scripts/analyze_infocurl_grid.py
