#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/private/envs/open-unlearning-npu-venv/bin/activate
unset PYTHONPATH

# Machine B: 4 NPU cards
# Global sharding plan across both machines:
# - 2-card machine runs worker 0,1
# - this machine runs worker 2,3,4,5
# Total workers = 6
#
# Assumption:
# - both machines use the same repo revision
# - both machines generate the same manifest ordering via build_infocurl_grid.py
# - output paths are either shared or will be merged later

python scripts/build_infocurl_grid.py

bash scripts/run_infocurl_grid_worker.sh 2 6 0 &
PID0=$!
bash scripts/run_infocurl_grid_worker.sh 3 6 1 &
PID1=$!
bash scripts/run_infocurl_grid_worker.sh 4 6 2 &
PID2=$!
bash scripts/run_infocurl_grid_worker.sh 5 6 3 &
PID3=$!

wait "$PID0"
wait "$PID1"
wait "$PID2"
wait "$PID3"

echo "Machine-4GPU workers complete: 2,3,4,5"
echo "If results are on shared storage, you can now run:"
echo "  python scripts/summarize_tofu10_runs.py"
echo "  python scripts/analyze_infocurl_grid.py"
