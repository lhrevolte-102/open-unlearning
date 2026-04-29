#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

source /opt/conda/private/envs/open-unlearning-npu-venv/bin/activate
unset PYTHONPATH

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ -z "${ASCEND_VISIBLE_DEVICES:-}" && -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  export ASCEND_VISIBLE_DEVICES=14
fi

MODEL="${MODEL:-Llama-3.2-3B-Instruct}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
SEED="${SEED:-0}"
TRAINER_NAME="${TRAINER_NAME:-InfoCURL_NPO}"
MODE="${MODE:-easy}"
SCORE_GAMMA="${SCORE_GAMMA:-0.5}"
LAM="${LAM:-0.0}"
SCORE_SUBPOOL="${SCORE_SUBPOOL:-64}"
K_STEPS="${K_STEPS:-10}"
PARAM_SCOPE="${PARAM_SCOPE:-last_layer_lm_head}"
RETAIN_BATCH_SIZE="${RETAIN_BATCH_SIZE:-8}"
RETAIN_EMA_DECAY="${RETAIN_EMA_DECAY:-0.9}"
STAGE_SCHEDULE="${STAGE_SCHEDULE:-}"
STAGE1_MODE="${STAGE1_MODE:-easy}"
STAGE2_MODE="${STAGE2_MODE:-hard}"
STAGE1_GAMMA="${STAGE1_GAMMA:-1.0}"
STAGE2_GAMMA="${STAGE2_GAMMA:-0.10}"
SWITCH_FRAC="${SWITCH_FRAC:-0.35}"
TRANSITION_FRAC="${TRANSITION_FRAC:-0.30}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_TAG="${RUN_TAG:-}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
RUN_GD_SWEEP="${RUN_GD_SWEEP:-0}"
CARD0="${CARD0:-0}"
CARD1="${CARD1:-1}"
SIMNPO_DELTA="${SIMNPO_DELTA:-0.0}"
SIMNPO_BETA="${SIMNPO_BETA:-4.5}"
SIMNPO_GAMMA="${SIMNPO_GAMMA:-0.125}"
SIMNPO_ALPHA="${SIMNPO_ALPHA:-1.0}"
GD_GAMMA="${GD_GAMMA:-1.0}"
GD_ALPHA="${GD_ALPHA:-1.0}"
GD_RETAIN_LOSS_TYPE="${GD_RETAIN_LOSS_TYPE:-NLL}"

RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}"

run_gd_sweep_one() {
  local device="$1"
  local trainer_name="$2"
  local tag_suffix="$3"
  local method_gamma="$4"
  local method_alpha="$5"
  local score_gamma="${6:-0.14}"
  local k_steps="${7:-20}"

  local run_tag
  if [[ "${trainer_name}" == "GradDiff" ]]; then
    run_tag="tofu_${MODEL}_${FORGET_SPLIT}_GradDiff_${tag_suffix}_bs8ga2_s${SEED}"
  else
    run_tag="tofu_${MODEL}_${FORGET_SPLIT}_InfoCURL_GradDiff_${tag_suffix}_bs8ga2_s${SEED}"
  fi
  local summary_path="saves/unlearn/${run_tag}/evals/TOFU_SUMMARY.json"
  if [[ -f "${summary_path}" ]]; then
    echo "SKIP existing ${run_tag}"
    return 0
  fi

  echo "============================================================"
  echo "Device ${device} | ${run_tag}"
  echo "${trainer_name} GD_GAMMA=${method_gamma} GD_ALPHA=${method_alpha} SCORE_GAMMA=${score_gamma} K=${k_steps}"
  echo "============================================================"

  (
    unset ASCEND_VISIBLE_DEVICES
    export ASCEND_RT_VISIBLE_DEVICES="${device}"
    export RUN_GD_SWEEP=0
    export TRAINER_NAME="${trainer_name}"
    export RUN_TAG="${run_tag}"
    export TRAIN_BATCH_SIZE=8
    export GRAD_ACCUM=2
    export SEED="${SEED}"
    export RUN_EVAL=1
    export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}"
    export GD_GAMMA="${method_gamma}"
    export GD_ALPHA="${method_alpha}"
    export GD_RETAIN_LOSS_TYPE=NLL
    if [[ "${trainer_name}" == "InfoCURL_GradDiff" ]]; then
      export MODE=hard
      export SCORE_GAMMA="${score_gamma}"
      export K_STEPS="${k_steps}"
      export SCORE_SUBPOOL=64
      export PARAM_SCOPE=last_layer_lm_head
      export RETAIN_BATCH_SIZE=8
      export RETAIN_EMA_DECAY=0.9
    fi
    bash scripts/tofu10_infocurl_npo_single.sh
  )
}

run_gd_sweep_card0() {
  run_gd_sweep_one "${CARD0}" GradDiff g0p25_a1p00 0.25 1.00
  run_gd_sweep_one "${CARD0}" GradDiff g0p50_a1p00 0.50 1.00
  run_gd_sweep_one "${CARD0}" GradDiff g1p00_a1p00 1.00 1.00
  run_gd_sweep_one "${CARD0}" GradDiff g1p50_a1p00 1.50 1.00
  run_gd_sweep_one "${CARD0}" GradDiff g2p00_a1p00 2.00 1.00
  run_gd_sweep_one "${CARD0}" GradDiff g1p00_a2p00 1.00 2.00
}

run_gd_sweep_card1() {
  run_gd_sweep_one "${CARD1}" InfoCURL_GradDiff mg0p50_a1p00_hsg0p14_k20 0.50 1.00 0.14 20
  run_gd_sweep_one "${CARD1}" InfoCURL_GradDiff mg1p00_a1p00_hsg0p14_k20 1.00 1.00 0.14 20
  run_gd_sweep_one "${CARD1}" InfoCURL_GradDiff mg1p50_a1p00_hsg0p14_k20 1.50 1.00 0.14 20
  run_gd_sweep_one "${CARD1}" InfoCURL_GradDiff mg2p00_a1p00_hsg0p14_k20 2.00 1.00 0.14 20
  run_gd_sweep_one "${CARD1}" InfoCURL_GradDiff mg1p00_a1p00_hsg0p10_k20 1.00 1.00 0.10 20
  run_gd_sweep_one "${CARD1}" InfoCURL_GradDiff mg1p00_a1p00_hsg0p20_k20 1.00 1.00 0.20 20
}

if [[ "${RUN_GD_SWEEP}" == "1" ]]; then
  run_gd_sweep_card0 &
  pid0=$!
  run_gd_sweep_card1 &
  pid1=$!
  echo "Launched GD sweep: card0 pid=${pid0}, card1 pid=${pid1}"
  wait "${pid0}"
  wait "${pid1}"
  python3 scripts/summarize_tofu10_runs.py
  exit 0
fi

resolve_model_path() {
  if [[ -n "${MODEL_PATH:-}" ]]; then
    echo "${MODEL_PATH}"
    return 0
  fi

  local cache_root="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  local cache_dir="${cache_root}/models--open-unlearning--tofu_${MODEL}_full/snapshots"
  if [[ -d "${cache_dir}" ]]; then
    local first_snapshot
    first_snapshot="$(find "${cache_dir}" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
    if [[ -n "${first_snapshot}" ]]; then
      echo "${first_snapshot}"
      return 0
    fi
  fi

  echo "open-unlearning/tofu_${MODEL}_full"
}

resolve_tokenizer_path() {
  if [[ -n "${TOKENIZER_PATH:-}" ]]; then
    echo "${TOKENIZER_PATH}"
    return 0
  fi

  local cache_root="${HF_HOME:-$HOME/.cache/huggingface}/hub"
  local cache_dir="${cache_root}/models--meta-llama--${MODEL}/snapshots"
  if [[ -d "${cache_dir}" ]]; then
    local first_snapshot
    first_snapshot="$(find "${cache_dir}" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1)"
    if [[ -n "${first_snapshot}" ]]; then
      echo "${first_snapshot}"
      return 0
    fi
  fi

  echo "meta-llama/${MODEL}"
}

MODEL_PATH="$(resolve_model_path)"
TOKENIZER_PATH="$(resolve_tokenizer_path)"

if [[ "${TRAINER_NAME}" == "NPO" || "${TRAINER_NAME}" == "SimNPO" || "${TRAINER_NAME}" == "GradDiff" ]]; then
  RUN_NAME="${RUN_TAG:-tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER_NAME}_single_s${SEED}}"
else
  RUN_NAME="${RUN_TAG:-tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER_NAME}_${MODE}_g${SCORE_GAMMA}_k${K_STEPS}_s${SEED}}"
fi

echo "================================================================"
echo "Run name    : ${RUN_NAME}"
echo "Trainer     : ${TRAINER_NAME}"
echo "ASCEND_VISIBLE_DEVICES   : ${ASCEND_VISIBLE_DEVICES:-<unset>}"
echo "ASCEND_RT_VISIBLE_DEVICES: ${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
echo "Forget split: ${FORGET_SPLIT}"
echo "Seed        : ${SEED}"
echo "================================================================"

TRAIN_CMD=(
  python src/train.py
  --config-name=unlearn.yaml
  experiment=unlearn/tofu/default.yaml
  model="${MODEL}"
  task_name="${RUN_NAME}"
  forget_split="${FORGET_SPLIT}"
  retain_split="${RETAIN_SPLIT}"
  holdout_split="${HOLDOUT_SPLIT}"
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}"
  model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}"
  model.model_args.attn_implementation=eager
  retain_logs_path="${RETAIN_LOGS_PATH}"
  trainer.args.seed="${SEED}"
  trainer.args.optim=adamw_torch
  trainer.args.per_device_train_batch_size="${TRAIN_BATCH_SIZE}"
  trainer.args.gradient_accumulation_steps="${GRAD_ACCUM}"
  trainer.args.report_to=none
  trainer.args.logging_steps=1
  trainer.args.do_eval=false
  trainer.args.eval_on_start=false
  trainer.args.eval_strategy=no
  trainer.args.save_strategy=no
)

if [[ -n "${MAX_STEPS}" ]]; then
  TRAIN_CMD+=("+trainer.args.max_steps=${MAX_STEPS}")
fi

if [[ "${TRAINER_NAME}" == "NPO" ]]; then
  TRAIN_CMD+=(
    trainer=NPO
    trainer.args.gradient_checkpointing=false
    trainer.args.ddp_find_unused_parameters=None
  )
elif [[ "${TRAINER_NAME}" == "SimNPO" ]]; then
  TRAIN_CMD+=(
    trainer=SimNPO
    trainer.args.gradient_checkpointing=false
    trainer.args.ddp_find_unused_parameters=None
    trainer.method_args.delta="${SIMNPO_DELTA}"
    trainer.method_args.beta="${SIMNPO_BETA}"
    trainer.method_args.gamma="${SIMNPO_GAMMA}"
    trainer.method_args.alpha="${SIMNPO_ALPHA}"
  )
elif [[ "${TRAINER_NAME}" == "GradDiff" ]]; then
  TRAIN_CMD+=(
    trainer=GradDiff
    trainer.args.gradient_checkpointing=false
    trainer.args.ddp_find_unused_parameters=None
    trainer.method_args.gamma="${GD_GAMMA}"
    trainer.method_args.alpha="${GD_ALPHA}"
    trainer.method_args.retain_loss_type="${GD_RETAIN_LOSS_TYPE}"
  )
else
  if [[ "${TRAINER_NAME}" == "InfoCURL_GradDiff" ]]; then
    TRAIN_CMD+=(
      trainer=GradDiff
      collator=DataCollatorForSupervisedDatasetwithIndex
      trainer.handler=InfoCURL_GradDiff
      trainer.args.gradient_checkpointing=false
      trainer.args.ddp_find_unused_parameters=None
      trainer.method_args.gamma="${GD_GAMMA}"
      trainer.method_args.alpha="${GD_ALPHA}"
      trainer.method_args.retain_loss_type="${GD_RETAIN_LOSS_TYPE}"
      +trainer.method_args.sampler.mode="${MODE}"
      +trainer.method_args.sampler.gamma="${SCORE_GAMMA}"
      +trainer.method_args.sampler.lam="${LAM}"
      +trainer.method_args.sampler.K="${K_STEPS}"
      +trainer.method_args.sampler.score_subpool="${SCORE_SUBPOOL}"
      +trainer.method_args.sampler.param_scope="${PARAM_SCOPE}"
      +trainer.method_args.sampler.retain_batch_size="${RETAIN_BATCH_SIZE}"
      +trainer.method_args.sampler.retain_ema_decay="${RETAIN_EMA_DECAY}"
    )
  else
    TRAIN_CMD+=(
      trainer=InfoCURL_NPO
      collator=DataCollatorForSupervisedDatasetwithIndex
      trainer.method_args.sampler.mode="${MODE}"
      trainer.method_args.sampler.gamma="${SCORE_GAMMA}"
      trainer.method_args.sampler.lam="${LAM}"
      trainer.method_args.sampler.K="${K_STEPS}"
      trainer.method_args.sampler.score_subpool="${SCORE_SUBPOOL}"
      trainer.method_args.sampler.param_scope="${PARAM_SCOPE}"
      trainer.method_args.sampler.retain_batch_size="${RETAIN_BATCH_SIZE}"
      trainer.method_args.sampler.retain_ema_decay="${RETAIN_EMA_DECAY}"
    )
  fi
  if [[ "${TRAINER_NAME}" == "InfoCURL_SimNPO" ]]; then
    TRAIN_CMD+=(
      trainer.handler=InfoCURL_SimNPO
      +trainer.method_args.delta="${SIMNPO_DELTA}"
      trainer.method_args.beta="${SIMNPO_BETA}"
      trainer.method_args.gamma="${SIMNPO_GAMMA}"
      trainer.method_args.alpha="${SIMNPO_ALPHA}"
      trainer.method_args.retain_loss_type=NLL
    )
  elif [[ "${TRAINER_NAME}" != "InfoCURL_NPO" && "${TRAINER_NAME}" != "InfoCURL_GradDiff" ]]; then
    echo "Unsupported TRAINER_NAME=${TRAINER_NAME}" >&2
    exit 2
  fi
  if [[ -n "${STAGE_SCHEDULE}" ]]; then
    TRAIN_CMD+=(
      +trainer.method_args.sampler.schedule="${STAGE_SCHEDULE}"
      +trainer.method_args.sampler.stage1_mode="${STAGE1_MODE}"
      +trainer.method_args.sampler.stage2_mode="${STAGE2_MODE}"
      +trainer.method_args.sampler.stage1_gamma="${STAGE1_GAMMA}"
      +trainer.method_args.sampler.stage2_gamma="${STAGE2_GAMMA}"
      +trainer.method_args.sampler.switch_frac="${SWITCH_FRAC}"
      +trainer.method_args.sampler.transition_frac="${TRANSITION_FRAC}"
    )
  fi
fi

"${TRAIN_CMD[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default.yaml \
    model="${MODEL}" \
    task_name="${RUN_NAME}_eval" \
    model.model_args.pretrained_model_name_or_path="saves/unlearn/${RUN_NAME}" \
    model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/${RUN_NAME}" \
    model.model_args.device_map=npu:0 \
    model.model_args.attn_implementation=eager \
    forget_split="${FORGET_SPLIT}" \
    holdout_split="${HOLDOUT_SPLIT}" \
    retain_logs_path="${RETAIN_LOGS_PATH}" \
    paths.output_dir="saves/unlearn/${RUN_NAME}/evals" \
    eval.tofu.batch_size="${EVAL_BATCH_SIZE}"
fi
