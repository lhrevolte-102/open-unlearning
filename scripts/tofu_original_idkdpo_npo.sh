#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source .venv/bin/activate

########################################
# Experiment config
########################################
MODEL="${MODEL:-Llama-3.2-3B-Instruct}"
FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-10}"
BETA="${BETA:-0.1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
TRAIN_LOGGING_STEPS="${TRAIN_LOGGING_STEPS:-1}"

GPU_ID="${GPU_ID:-0}"
RESUME="${RESUME:-true}"

log() {
    echo "[Original-TOFU] $*"
}

is_truthy() {
    local value="${1:-}"
    value="${value,,}"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "y" || "$value" == "on" ]]
}

model_dir_complete() {
    local model_dir="$1"
    [[ -d "$model_dir" ]] || return 1
    [[ -s "${model_dir}/config.json" ]] || return 1
    if compgen -G "${model_dir}/model-*.safetensors" > /dev/null; then
        return 0
    fi
    if [[ -s "${model_dir}/model.safetensors" || -s "${model_dir}/model.safetensors.index.json" ]]; then
        return 0
    fi
    return 1
}

training_output_complete() {
    local output_dir="$1"
    [[ -s "${output_dir}/trainer_state.json" ]] || return 1
    model_dir_complete "$output_dir"
}

latest_checkpoint_in_dir() {
    local output_dir="$1"
    [[ -d "$output_dir" ]] || return 0
    find "$output_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

tofu_eval_has_full_metrics() {
    local eval_file="$1"
    [[ -s "$eval_file" ]] || return 1
    python -c 'import json, sys
required = {
    "exact_memorization",
    "mia_gradnorm",
    "mia_loss",
    "mia_min_k",
    "mia_min_k_plus_plus",
    "mia_reference",
    "mia_zlib",
}
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    metrics = json.load(handle)
raise SystemExit(0 if required.issubset(metrics) else 1)' "$eval_file"
}

run_unlearn() {
    local method_label="$1"
    local experiment="$2"
    local trainer_name="$3"
    local task_name="$4"
    local output_dir="saves/unlearn/${task_name}"
    local extra_args=()
    local latest_checkpoint=""

    if is_truthy "$RESUME" && training_output_complete "$output_dir"; then
        log "Skipping ${method_label} unlearn; found completed model output at ${output_dir}."
        return
    fi

    if is_truthy "$RESUME"; then
        latest_checkpoint=$(latest_checkpoint_in_dir "$output_dir")
        if [[ -n "$latest_checkpoint" ]]; then
            log "Resuming ${method_label} unlearn from in-progress checkpoint ${latest_checkpoint}."
            extra_args+=("resume_from_checkpoint=${latest_checkpoint}")
        fi
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train.py --config-name=unlearn.yaml \
        experiment=${experiment} \
        trainer=${trainer_name} \
        task_name=${task_name} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        retain_logs_path=${RETAIN_LOGS_PATH} \
        trainer.method_args.beta=${BETA} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.num_train_epochs=${TOTAL_EPOCHS} \
        trainer.args.logging_steps=${TRAIN_LOGGING_STEPS} \
        +trainer.args.logging_first_step=true \
        trainer.args.gradient_checkpointing=true \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.eval_strategy=no \
        trainer.args.save_strategy=epoch \
        +trainer.args.save_total_limit=1 \
        trainer.args.save_only_model=false \
        +trainer.args.ignore_data_skip=true \
        "${extra_args[@]}" \
        paths.output_dir=${output_dir}
}

run_eval() {
    local method_label="$1"
    local task_name="$2"
    local output_dir="saves/unlearn/${task_name}"
    local eval_dir="${output_dir}/evals"

    if is_truthy "$RESUME" && tofu_eval_has_full_metrics "${eval_dir}/TOFU_EVAL.json"; then
        log "Skipping ${method_label} eval; found existing full-metric eval logs at ${eval_dir}/TOFU_EVAL.json."
        return
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/eval.py \
        experiment=eval/tofu/full \
        forget_split=${FORGET_SPLIT} \
        model=${MODEL} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${output_dir} \
        model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        paths.output_dir=${eval_dir} \
        retain_logs_path=${RETAIN_LOGS_PATH} \
        reference_model_path=${RETAIN_MODEL_PATH}
}

########################################
# Derived paths
########################################
BASE_MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
RETAIN_MODEL_PATH="open-unlearning/tofu_${MODEL}_${RETAIN_SPLIT}"
RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"

IDKDPO_TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_idkdpo_original"
NPO_TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_npo_original"

if is_truthy "$RESUME"; then
    log "Resume mode enabled. Completed unlearning runs and full-metric evals will be skipped."
else
    log "Resume mode disabled. Existing outputs will be reused only when the underlying command does so implicitly."
fi

run_unlearn "IdkDPO" "unlearn/tofu/idk.yaml" "DPO" "${IDKDPO_TASK_NAME}"
run_unlearn "NPO" "unlearn/tofu/default.yaml" "NPO" "${NPO_TASK_NAME}"

run_eval "IdkDPO" "${IDKDPO_TASK_NAME}"
run_eval "NPO" "${NPO_TASK_NAME}"

log "Finished original TOFU runs."
log "IdkDPO output: saves/unlearn/${IDKDPO_TASK_NAME}"
log "NPO output: saves/unlearn/${NPO_TASK_NAME}"
