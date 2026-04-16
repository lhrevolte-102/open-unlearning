#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT_DIR"
source .venv/bin/activate

########################################
# Experiment config
########################################
MODEL="Llama-3.2-3B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"

TOTAL_EPOCHS=10
STAGE_PERCENTILES='[0.3,0.6,1.0]'
STAGE_EPOCH_RATIOS='[0.3,0.3,0.4]'
STAGE_LOGGING_STEPS="${STAGE_LOGGING_STEPS:-1}"
INTRA_STAGE_ORDER="${INTRA_STAGE_ORDER:-random}"
BETA=0.1
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
NUM_REFERENCE_REPEATS="${NUM_REFERENCE_REPEATS:-3}"
REPEAT_SPLIT_SEED="${REPEAT_SPLIT_SEED:-0}"

GPU_ID="0"
RESUME="${RESUME:-true}"

log() {
    echo "[Selective-NPO] $*"
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

selective_output_dir() {
    local mode="$1"
    local task_name="$2"
    echo "saves/${mode}/${task_name}"
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

########################################
# Derived paths
########################################
BASE_MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
RETAIN_MODEL_PATH="open-unlearning/tofu_${MODEL}_${RETAIN_SPLIT}"
TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_selective_npo_random_repeated_halving"
REFERENCE_TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_references_npo_random_repeated_halving"
REFERENCE_TASK_NAME="${REFERENCE_TASK_PREFIX}"
PREPARE_TASK_NAME="${TASK_PREFIX}_prepare"
RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
REFERENCE_DIR=$(selective_output_dir "selective_reference" "${REFERENCE_TASK_NAME}")
REFERENCE_SPLITS_DIR="${REFERENCE_DIR}/reference_splits"
REFERENCE_MODELS_DIR="${REFERENCE_DIR}/models"
REFERENCE_MANIFEST_PATH="${REFERENCE_DIR}/reference_models.json"
PREPARE_DIR=$(selective_output_dir "selective_prepare" "${PREPARE_TASK_NAME}")
DIFFICULTY_PATH="${PREPARE_DIR}/difficulty.json"

mkdir -p "$REFERENCE_SPLITS_DIR" "$REFERENCE_MODELS_DIR" "$(dirname "$DIFFICULTY_PATH")"

if is_truthy "$RESUME"; then
    log "Resume mode enabled. Completed reference split training, difficulty scoring, stage training, and evals will be skipped."
else
    log "Resume mode disabled. Existing outputs will be reused only when the underlying command does so implicitly."
fi

python src/selective_reference.py \
    experiment=selective/tofu/npo \
    task_name=${REFERENCE_TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    reference_splits_output_dir=${REFERENCE_SPLITS_DIR} \
    reference_splits_summary_path=${REFERENCE_SPLITS_DIR}/reference_splits.json \
    checkpoint_root_dir=${REFERENCE_MODELS_DIR} \
    reference_manifest_output_path=${REFERENCE_MANIFEST_PATH} \
    num_repeats=${NUM_REFERENCE_REPEATS} \
    repeat_split_seed=${REPEAT_SPLIT_SEED} \
    validate_checkpoint_paths=false

for TRAIN_MANIFEST in "${REFERENCE_SPLITS_DIR}"/split*_train.json; do
    if [[ ! -f "${TRAIN_MANIFEST}" ]]; then
        echo "No reference split manifests were found under ${REFERENCE_SPLITS_DIR}."
        exit 1
    fi
    SPLIT_NAME=$(python -c "import json; print(json.load(open('${TRAIN_MANIFEST}', 'r', encoding='utf-8'))['split_name'])")
    SPLIT_TASK_NAME=${REFERENCE_TASK_PREFIX}_${SPLIT_NAME}
    SPLIT_OUTPUT_DIR=${REFERENCE_MODELS_DIR}/${SPLIT_NAME}
    if is_truthy "$RESUME" && training_output_complete "$SPLIT_OUTPUT_DIR"; then
        log "Skipping reference ${SPLIT_NAME}; found completed model output at ${SPLIT_OUTPUT_DIR}."
        continue
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/selective_npo \
        trainer=NPO \
        task_name=${SPLIT_TASK_NAME} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        retain_logs_path=${RETAIN_LOGS_PATH} \
        selective_manifest_path=${TRAIN_MANIFEST} \
        trainer.method_args.beta=${BETA} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.num_train_epochs=${TOTAL_EPOCHS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.do_eval=false \
        trainer.args.eval_on_start=false \
        trainer.args.eval_strategy=no \
        trainer.args.save_strategy=no \
        trainer.args.save_only_model=true \
        paths.output_dir=${SPLIT_OUTPUT_DIR}
done

python src/selective_reference.py \
    experiment=selective/tofu/npo \
    task_name=${REFERENCE_TASK_NAME} \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    reference_splits_output_dir=${REFERENCE_SPLITS_DIR} \
    reference_splits_summary_path=${REFERENCE_SPLITS_DIR}/reference_splits.json \
    checkpoint_root_dir=${REFERENCE_MODELS_DIR} \
    reference_manifest_output_path=${REFERENCE_MANIFEST_PATH} \
    num_repeats=${NUM_REFERENCE_REPEATS} \
    repeat_split_seed=${REPEAT_SPLIT_SEED} \
    validate_checkpoint_paths=true

if is_truthy "$RESUME" && [[ -s "$DIFFICULTY_PATH" ]]; then
    log "Skipping difficulty preparation; found existing score file at ${DIFFICULTY_PATH}."
else
    python src/selective_prepare.py \
        experiment=selective/tofu/npo \
        task_name=${PREPARE_TASK_NAME} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        reference_manifest_path=${REFERENCE_MANIFEST_PATH} \
        beta=${BETA}
fi

run_selective_order() {
    local intra_stage_order="$1"
    local stage_task_prefix="${TASK_PREFIX}_${intra_stage_order}"
    local stage_task_name="${stage_task_prefix}_stages"
    local stage_dir
    local prev_output_dir=""
    local final_task_name=""
    local final_output_dir=""

    stage_dir="$(selective_output_dir "selective_stage" "${stage_task_name}")/stages"

    mkdir -p "$stage_dir"

    python src/selective_stage.py \
        task_name=${stage_task_name} \
        difficulty_path=${DIFFICULTY_PATH} \
        intra_stage_order=${intra_stage_order} \
        stage_percentiles=${STAGE_PERCENTILES} \
        stage_epoch_ratios=${STAGE_EPOCH_RATIOS}

    for STAGE_MANIFEST in "${stage_dir}"/stage[0-9]*.json; do
        STAGE_NAME=$(python -c "import json; print(json.load(open('${STAGE_MANIFEST}', 'r', encoding='utf-8'))['stage_name'])")
        EPOCH_RATIO=$(python -c "import json; print(json.load(open('${STAGE_MANIFEST}', 'r', encoding='utf-8'))['epoch_ratio'])")
        STAGE_EPOCHS=$(python -c "total_epochs=float('${TOTAL_EPOCHS}'); epoch_ratio=float('${EPOCH_RATIO}'); print(max(epoch_ratio * total_epochs, 1.0))")
        STAGE_TASK_NAME=${stage_task_prefix}_${STAGE_NAME}
        FINAL_TASK_NAME=${STAGE_TASK_NAME}
        final_output_dir="saves/unlearn/${FINAL_TASK_NAME}"
        STAGE_OUTPUT_DIR="saves/unlearn/${STAGE_TASK_NAME}"
        if is_truthy "$RESUME" && training_output_complete "$STAGE_OUTPUT_DIR"; then
            log "Skipping ${STAGE_NAME} (${intra_stage_order}); found completed training output at ${STAGE_OUTPUT_DIR}."
            prev_output_dir=${STAGE_OUTPUT_DIR}
            continue
        fi

        EXTRA_ARGS=()
        LATEST_CHECKPOINT=""
        STAGE_MODEL_PATH="${BASE_MODEL_PATH}"
        if is_truthy "$RESUME"; then
            LATEST_CHECKPOINT=$(latest_checkpoint_in_dir "$STAGE_OUTPUT_DIR")
            if [[ -n "$LATEST_CHECKPOINT" ]]; then
                log "Resuming ${STAGE_NAME} (${intra_stage_order}) from in-progress checkpoint ${LATEST_CHECKPOINT}."
                EXTRA_ARGS+=("resume_from_checkpoint=${LATEST_CHECKPOINT}")
            fi
        fi

        if [[ ${#EXTRA_ARGS[@]} -eq 0 && -n "$prev_output_dir" ]]; then
            LATEST_CHECKPOINT=$(latest_checkpoint_in_dir "$prev_output_dir")
            if [[ -z "$LATEST_CHECKPOINT" ]]; then
                echo "No checkpoint found under ${prev_output_dir} for ${STAGE_NAME} resume."
                exit 1
            fi
            log "Initializing ${STAGE_NAME} (${intra_stage_order}) from previous stage model ${LATEST_CHECKPOINT}."
            STAGE_MODEL_PATH="${LATEST_CHECKPOINT}"
        fi

        CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train.py --config-name=unlearn.yaml \
            experiment=unlearn/tofu/selective_npo \
            trainer=NPO \
            task_name=${STAGE_TASK_NAME} \
            model=${MODEL} \
            forget_split=${FORGET_SPLIT} \
            retain_split=${RETAIN_SPLIT} \
            model.model_args.pretrained_model_name_or_path=${STAGE_MODEL_PATH} \
            model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
            retain_logs_path=${RETAIN_LOGS_PATH} \
            intra_stage_order=${intra_stage_order} \
            selective_manifest_path=${STAGE_MANIFEST} \
            trainer.method_args.beta=${BETA} \
            trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
            trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
            trainer.args.num_train_epochs=${STAGE_EPOCHS} \
            trainer.args.logging_steps=${STAGE_LOGGING_STEPS} \
            +trainer.args.logging_first_step=true \
            trainer.args.gradient_checkpointing=true \
            trainer.args.do_eval=false \
            trainer.args.eval_on_start=false \
            trainer.args.eval_strategy=no \
            trainer.args.save_strategy=epoch \
            +trainer.args.save_total_limit=1 \
            trainer.args.save_only_model=false \
            +trainer.args.ignore_data_skip=true \
            "${EXTRA_ARGS[@]}" \
            paths.output_dir=${STAGE_OUTPUT_DIR}

        prev_output_dir=${STAGE_OUTPUT_DIR}
    done

    if [[ -z "$FINAL_TASK_NAME" ]]; then
        echo "No stage manifests were found under ${stage_dir}."
        exit 1
    fi

    FINAL_EVAL_DIR="${final_output_dir}/evals"
    if is_truthy "$RESUME" && tofu_eval_has_full_metrics "${FINAL_EVAL_DIR}/TOFU_EVAL.json"; then
        log "Skipping final eval for ${intra_stage_order}; found existing full-metric eval logs at ${FINAL_EVAL_DIR}/TOFU_EVAL.json."
    else
        CUDA_VISIBLE_DEVICES=${GPU_ID} python src/eval.py \
            experiment=eval/tofu/full \
            forget_split=${FORGET_SPLIT} \
            model=${MODEL} \
            task_name=${FINAL_TASK_NAME} \
            model.model_args.pretrained_model_name_or_path=${final_output_dir} \
            model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
            paths.output_dir=${FINAL_EVAL_DIR} \
            retain_logs_path=${RETAIN_LOGS_PATH} \
            reference_model_path=${RETAIN_MODEL_PATH}
    fi

    echo "Selective-NPO (${intra_stage_order}) training completed. Final stage output: ${prev_output_dir}"
}

run_selective_order "${INTRA_STAGE_ORDER}"
