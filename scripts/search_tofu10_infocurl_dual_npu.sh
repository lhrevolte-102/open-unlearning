#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/private/envs/open-unlearning-npu-venv/bin/activate
unset PYTHONPATH

BASELINE_RUN="tofu_Llama-3.2-3B-Instruct_forget10_NPO_effbs32_8x4_full_s0"

run_one() {
  local rt_dev="$1"
  local run_tag="$2"
  local mode="$3"
  local gamma="$4"
  local k_steps="$5"

  echo "============================================================"
  echo "Launching ${run_tag} on ASCEND_RT_VISIBLE_DEVICES=${rt_dev}"
  echo "mode=${mode} gamma=${gamma} K=${k_steps}"
  echo "============================================================"

  (
    unset ASCEND_VISIBLE_DEVICES
    export ASCEND_RT_VISIBLE_DEVICES="${rt_dev}"
    export RUN_TAG="${run_tag}"
    export TRAINER_NAME=InfoCURL_NPO
    export MODE="${mode}"
    export SCORE_GAMMA="${gamma}"
    export K_STEPS="${k_steps}"
    export SCORE_SUBPOOL=64
    export TRAIN_BATCH_SIZE=8
    export GRAD_ACCUM=4
    export SEED=0
    export RUN_EVAL=1
    bash scripts/tofu10_infocurl_npo_single.sh
  )
}

worker0() {
  run_one 0 tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p20_k20_effbs32_8x4_s0 hard 0.20 20
  run_one 0 tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p25_k30_effbs32_8x4_s0 hard 0.25 30
}

worker1() {
  run_one 1 tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_easy_g0p75_k20_effbs32_8x4_s0 easy 0.75 20
  run_one 1 tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_easy_g1p00_k20_effbs32_8x4_s0 easy 1.00 20
}

worker0 &
PID0=$!
worker1 &
PID1=$!

wait "$PID0"
wait "$PID1"

python scripts/summarize_tofu10_runs.py
python scripts/analyze_infocurl_search.py
