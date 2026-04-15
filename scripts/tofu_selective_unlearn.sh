#!/bin/bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source .venv/bin/activate

########################################
# Experiment config
########################################
METHOD_VARIANT="npo"  # Options: npo, idkdpo
MODEL="Llama-3.2-3B-Instruct"
FORGET_SPLIT="forget10"
RETAIN_SPLIT="retain90"

TOTAL_EPOCHS=10
STAGE_PERCENTILES='[0.3,0.6,1.0]'
STAGE_EPOCH_RATIOS='[0.3,0.3,0.4]'
BETA=0.1
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4

GPU_ID="0"

########################################
# Derived paths
########################################
BASE_MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_selective_${METHOD_VARIANT}"
REFERENCE_TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_references_${METHOD_VARIANT}"
RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
REFERENCE_MANIFEST_PATH="saves/selective_refs/${REFERENCE_TASK_PREFIX}/reference_models.json"
SELECTIVE_DIR="saves/selective/${TASK_PREFIX}"
DIFFICULTY_PATH="${SELECTIVE_DIR}/difficulty/difficulty.json"
STAGE_DIR="${SELECTIVE_DIR}/stages"

if [[ -z "$REFERENCE_MANIFEST_PATH" ]]; then
    echo "REFERENCE_MANIFEST_PATH must point to a reference_models.json manifest."
    exit 1
fi

if [[ ! -f "$REFERENCE_MANIFEST_PATH" ]]; then
    echo "Reference manifest not found: ${REFERENCE_MANIFEST_PATH}"
    exit 1
fi

case "$METHOD_VARIANT" in
    npo)
        PREP_EXPERIMENT=selective/tofu/npo
        TRAIN_EXPERIMENT=unlearn/tofu/selective_npo
        TRAINER=NPO
        ;;
    idkdpo)
        PREP_EXPERIMENT=selective/tofu/idkdpo
        TRAIN_EXPERIMENT=unlearn/tofu/selective_idk
        TRAINER=DPO
        ;;
    *)
        echo "Unsupported method variant: $METHOD_VARIANT"
        echo "Expected one of: npo, idkdpo"
        exit 1
        ;;
esac

mkdir -p "$(dirname "$DIFFICULTY_PATH")" "$STAGE_DIR"

python src/selective_prepare.py \
    experiment=${PREP_EXPERIMENT} \
    task_name=${TASK_PREFIX}_prepare \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    reference_manifest_path=${REFERENCE_MANIFEST_PATH} \
    beta=${BETA} \
    score_output_path=${DIFFICULTY_PATH}

python src/selective_stage.py \
    task_name=${TASK_PREFIX}_stages \
    difficulty_path=${DIFFICULTY_PATH} \
    output_dir=${STAGE_DIR} \
    stage_percentiles=${STAGE_PERCENTILES} \
    stage_epoch_ratios=${STAGE_EPOCH_RATIOS}

PREV_OUTPUT_DIR=""
for STAGE_MANIFEST in "${STAGE_DIR}"/stage*.json; do
    STAGE_NAME=$(python -c "import json; print(json.load(open('${STAGE_MANIFEST}', 'r', encoding='utf-8'))['stage_name'])")
    EPOCH_RATIO=$(python -c "import json; print(json.load(open('${STAGE_MANIFEST}', 'r', encoding='utf-8'))['epoch_ratio'])")
    STAGE_EPOCHS=$(python -c "total_epochs=float('${TOTAL_EPOCHS}'); epoch_ratio=float('${EPOCH_RATIO}'); print(max(epoch_ratio * total_epochs, 1.0))")
    STAGE_TASK_NAME=${TASK_PREFIX}_${STAGE_NAME}

    EXTRA_ARGS=()
    if [[ -n "$PREV_OUTPUT_DIR" ]]; then
        LATEST_CHECKPOINT=$(find "$PREV_OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)
        if [[ -z "$LATEST_CHECKPOINT" ]]; then
            echo "No checkpoint found under ${PREV_OUTPUT_DIR} for ${STAGE_NAME} resume."
            exit 1
        fi
        EXTRA_ARGS+=("resume_from_checkpoint=${LATEST_CHECKPOINT}")
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train.py --config-name=unlearn.yaml \
        experiment=${TRAIN_EXPERIMENT} \
        trainer=${TRAINER} \
        task_name=${STAGE_TASK_NAME} \
        model=${MODEL} \
        forget_split=${FORGET_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
        retain_logs_path=${RETAIN_LOGS_PATH} \
        selective_manifest_path=${STAGE_MANIFEST} \
        trainer.method_args.beta=${BETA} \
        trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
        trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
        trainer.args.num_train_epochs=${STAGE_EPOCHS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.do_eval=false \
        trainer.args.save_strategy=epoch \
        trainer.args.save_total_limit=1 \
        trainer.args.save_only_model=false \
        trainer.args.ignore_data_skip=true \
        "${EXTRA_ARGS[@]}"

    PREV_OUTPUT_DIR=saves/unlearn/${STAGE_TASK_NAME}
done

echo "Selective ${METHOD_VARIANT} training completed. Final stage output: ${PREV_OUTPUT_DIR}"
