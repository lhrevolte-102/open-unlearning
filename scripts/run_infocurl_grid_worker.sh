#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/private/envs/open-unlearning-npu-venv/bin/activate
unset PYTHONPATH

GRID_CSV="${GRID_CSV:-saves/unlearn/infocurl_grid_manifest.csv}"
WORKER_ID="${1:-0}"
NUM_WORKERS="${2:-2}"
RT_DEVICE="${3:-0}"

run_row() {
  local family="$1"
  local run_tag="$2"
  local trainer_name="$3"
  local mode="$4"
  local score_gamma="$5"
  local lam="$6"
  local score_subpool="$7"
  local k_steps="$8"
  local param_scope="$9"
  local retain_batch_size="${10}"
  local retain_ema_decay="${11}"
  local stage_schedule="${12}"
  local stage1_mode="${13}"
  local stage2_mode="${14}"
  local stage1_gamma="${15}"
  local stage2_gamma="${16}"
  local switch_frac="${17}"
  local transition_frac="${18}"
  local train_batch_size="${19}"
  local grad_accum="${20}"
  local seed="${21}"
  local run_eval="${22}"
  local max_steps="${23}"

  local summary_path="saves/unlearn/${run_tag}/evals/TOFU_SUMMARY.json"
  if [[ -f "${summary_path}" ]]; then
    echo "SKIP existing ${run_tag}"
    return 0
  fi

  echo "============================================================"
  echo "Worker ${WORKER_ID}/${NUM_WORKERS} on RT device ${RT_DEVICE}"
  echo "Family: ${family}"
  echo "Run   : ${run_tag}"
  echo "============================================================"

  (
    unset ASCEND_VISIBLE_DEVICES
    export ASCEND_RT_VISIBLE_DEVICES="${RT_DEVICE}"
    export RUN_TAG="${run_tag}"
    export TRAINER_NAME="${trainer_name}"
    export MODE="${mode}"
    export SCORE_GAMMA="${score_gamma}"
    export LAM="${lam}"
    export SCORE_SUBPOOL="${score_subpool}"
    export K_STEPS="${k_steps}"
    export PARAM_SCOPE="${param_scope}"
    export RETAIN_BATCH_SIZE="${retain_batch_size}"
    export RETAIN_EMA_DECAY="${retain_ema_decay}"
    export STAGE_SCHEDULE="${stage_schedule}"
    export STAGE1_MODE="${stage1_mode}"
    export STAGE2_MODE="${stage2_mode}"
    export STAGE1_GAMMA="${stage1_gamma}"
    export STAGE2_GAMMA="${stage2_gamma}"
    export SWITCH_FRAC="${switch_frac}"
    export TRANSITION_FRAC="${transition_frac}"
    export TRAIN_BATCH_SIZE=8
    export GRAD_ACCUM=2
    export SEED="${seed}"
    export RUN_EVAL="${run_eval}"
    export MAX_STEPS="${max_steps}"
    bash scripts/tofu10_infocurl_npo_single.sh
  )
}

python - <<'PY' "$GRID_CSV" "$WORKER_ID" "$NUM_WORKERS" | \
while IFS=$'\t' read -r family run_tag trainer_name mode score_gamma lam score_subpool k_steps param_scope retain_batch_size retain_ema_decay stage_schedule stage1_mode stage2_mode stage1_gamma stage2_gamma switch_frac transition_frac train_batch_size grad_accum seed run_eval max_steps; do
import csv, sys
grid_csv = sys.argv[1]
worker_id = int(sys.argv[2])
num_workers = int(sys.argv[3])
with open(grid_csv, 'r', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
for idx, row in enumerate(rows):
    if idx % num_workers != worker_id:
        continue
    cols = [
        row.get('family', ''),
        row.get('run_tag', ''),
        row.get('trainer_name', ''),
        row.get('mode', ''),
        row.get('score_gamma', ''),
        row.get('lam', ''),
        row.get('score_subpool', ''),
        row.get('K_steps', ''),
        row.get('param_scope', ''),
        row.get('retain_batch_size', ''),
        row.get('retain_ema_decay', ''),
        row.get('stage_schedule', ''),
        row.get('stage1_mode', ''),
        row.get('stage2_mode', ''),
        row.get('stage1_gamma', ''),
        row.get('stage2_gamma', ''),
        row.get('switch_frac', ''),
        row.get('transition_frac', ''),
        row.get('train_batch_size', '8'),
        row.get('grad_accum', '2'),
        row.get('seed', ''),
        row.get('run_eval', ''),
        row.get('max_steps', ''),
    ]
    print('\t'.join(cols))
PY
  run_row "$family" "$run_tag" "$trainer_name" "$mode" "$score_gamma" "$lam" "$score_subpool" "$k_steps" "$param_scope" "$retain_batch_size" "$retain_ema_decay" "$stage_schedule" "$stage1_mode" "$stage2_mode" "$stage1_gamma" "$stage2_gamma" "$switch_frac" "$transition_frac" "$train_batch_size" "$grad_accum" "$seed" "$run_eval" "$max_steps"
done
