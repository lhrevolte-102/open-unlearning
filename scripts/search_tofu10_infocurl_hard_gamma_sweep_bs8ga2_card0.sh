#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
source /opt/conda/private/envs/open-unlearning-npu-venv/bin/activate
unset PYTHONPATH

# Card-0 sweep: isolate the effect of gamma while keeping
# mode=hard, K=20, effective batch size = 8 * 2 = 16.
#
# Run on card 0:
#   bash scripts/search_tofu10_infocurl_hard_gamma_sweep_bs8ga2_card0.sh
#
# Recommended to run in parallel with:
#   bash scripts/search_tofu10_infocurl_hard_k_sweep_bs8ga2_card1.sh

run_one() {
  local run_tag="$1"
  local gamma="$2"

  echo "============================================================"
  echo "Launching ${run_tag}"
  echo "  mode=hard gamma=${gamma} K=20 bs=8 ga=2"
  echo "============================================================"

  (
    unset ASCEND_VISIBLE_DEVICES
    export ASCEND_RT_VISIBLE_DEVICES=0
    export RUN_TAG="${run_tag}"
    export TRAINER_NAME=InfoCURL_NPO
    export MODE=hard
    export SCORE_GAMMA="${gamma}"
    export K_STEPS=20
    export SCORE_SUBPOOL=64
    export TRAIN_BATCH_SIZE=8
    export GRAD_ACCUM=2
    export SEED=0
    export RUN_EVAL=1
    bash scripts/tofu10_infocurl_npo_single.sh
  )
}

run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p08_k20_bs8ga2_s0 0.08
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p10_k20_bs8ga2_s0 0.10
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p12_k20_bs8ga2_s0 0.12
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p16_k20_bs8ga2_s0 0.16
run_one tofu_Llama-3.2-3B-Instruct_forget10_InfoCURL_NPO_hard_g0p20_k20_bs8ga2_s0 0.20
