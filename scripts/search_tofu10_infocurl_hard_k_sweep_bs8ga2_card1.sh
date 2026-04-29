#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/private/envs/open-unlearning-npu-venv/bin/activate
unset PYTHONPATH

# Card-1 sweep: isolate the effect of K while keeping
# mode=hard, gamma=0.14, effective batch size = 8 * 2 = 16.
#
# Run on card 1:
#   bash scripts/search_tofu10_infocurl_hard_k_sweep_bs8ga2_card1.sh
#
# Recommended to run in parallel with:
#   bash scripts/search_tofu10_infocurl_hard_gamma_sweep_bs8ga2_card0.sh

run_one() {
  local run_tag="$1"
  local k_steps="$2"

  echo "============================================================"
  echo "Launching ${run_tag}"
  echo "  mode=hard gamma=0.14 K=${k_steps} bs=8 ga=2"
  echo "============================================================"

  (
    unset ASCEND_VISIBLE_DEVICES
    export ASCEND_RT_VISIBLE_DEVICES=1
    export RUN_TAG="${run_tag}"
    export TRAINER_NAME=InfoCURL_NPO
    export MODE=hard
    export SCORE_GAMMA=0.14
    export K_STEPS="${k_steps}"
    export SCORE_SUBPOOL=64
    export TRAIN_BATCH_SIZE=8
    export GRAD_ACCUM=2
    export SEED=0
    export RUN_EVAL=1
    bash scripts/tofu10_infocurl_npo_single.sh
  )
}

run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p14_k10_bs8ga2_s0 10
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p14_k15_bs8ga2_s0 15
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p14_k20_bs8ga2_s0 20
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p14_k25_bs8ga2_s0 25
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p14_k30_bs8ga2_s0 30
