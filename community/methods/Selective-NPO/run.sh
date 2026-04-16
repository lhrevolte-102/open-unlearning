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
BETA=0.1
PER_DEVICE_TRAIN_BATCH_SIZE=16
GRADIENT_ACCUMULATION_STEPS=4
NUM_FOLDS=4
FOLD_ASSIGNMENT_SEED=0

GPU_ID="0"

########################################
# Derived paths
########################################
BASE_MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_selective_npo"
REFERENCE_TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_references_npo"
RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
REFERENCE_DIR="saves/selective_refs/${REFERENCE_TASK_PREFIX}"
FOLDS_DIR="${REFERENCE_DIR}/folds"
REFERENCE_MODELS_DIR="${REFERENCE_DIR}/models"
REFERENCE_MANIFEST_PATH="saves/selective_refs/${REFERENCE_TASK_PREFIX}/reference_models.json"
SELECTIVE_DIR="saves/selective/${TASK_PREFIX}"
DIFFICULTY_PATH="${SELECTIVE_DIR}/difficulty/difficulty.json"
STAGE_DIR="${SELECTIVE_DIR}/stages"

mkdir -p "$FOLDS_DIR" "$REFERENCE_MODELS_DIR" "$(dirname "$DIFFICULTY_PATH")" "$STAGE_DIR"

python src/selective_reference.py \
    experiment=selective/tofu/npo \
    task_name=${REFERENCE_TASK_PREFIX}_folds \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    num_folds=${NUM_FOLDS} \
    fold_assignment_seed=${FOLD_ASSIGNMENT_SEED} \
    folds_output_dir=${FOLDS_DIR} \
    folds_summary_path=${FOLDS_DIR}/folds.json \
    checkpoint_root_dir=${REFERENCE_MODELS_DIR} \
    reference_manifest_output_path=${REFERENCE_MANIFEST_PATH} \
    validate_checkpoint_paths=false

for (( fold_id=0; fold_id<NUM_FOLDS; fold_id++ )); do
    TRAIN_MANIFEST=${FOLDS_DIR}/fold${fold_id}_train.json
    FOLD_TASK_NAME=${REFERENCE_TASK_PREFIX}_fold${fold_id}
    FOLD_OUTPUT_DIR=${REFERENCE_MODELS_DIR}/fold${fold_id}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train.py --config-name=unlearn.yaml \
        experiment=unlearn/tofu/selective_npo \
        trainer=NPO \
        task_name=${FOLD_TASK_NAME} \
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
        paths.output_dir=${FOLD_OUTPUT_DIR}
done

python src/selective_reference.py \
    experiment=selective/tofu/npo \
    task_name=${REFERENCE_TASK_PREFIX}_manifest \
    model=${MODEL} \
    forget_split=${FORGET_SPLIT} \
    retain_split=${RETAIN_SPLIT} \
    model.model_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
    num_folds=${NUM_FOLDS} \
    fold_assignment_seed=${FOLD_ASSIGNMENT_SEED} \
    folds_output_dir=${FOLDS_DIR} \
    folds_summary_path=${FOLDS_DIR}/folds.json \
    checkpoint_root_dir=${REFERENCE_MODELS_DIR} \
    reference_manifest_output_path=${REFERENCE_MANIFEST_PATH} \
    validate_checkpoint_paths=true

python src/selective_prepare.py \
    experiment=selective/tofu/npo \
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
FINAL_TASK_NAME=""
for STAGE_MANIFEST in "${STAGE_DIR}"/stage*.json; do
    STAGE_NAME=$(python -c "import json; print(json.load(open('${STAGE_MANIFEST}', 'r', encoding='utf-8'))['stage_name'])")
    EPOCH_RATIO=$(python -c "import json; print(json.load(open('${STAGE_MANIFEST}', 'r', encoding='utf-8'))['epoch_ratio'])")
    STAGE_EPOCHS=$(python -c "total_epochs=float('${TOTAL_EPOCHS}'); epoch_ratio=float('${EPOCH_RATIO}'); print(max(epoch_ratio * total_epochs, 1.0))")
    STAGE_TASK_NAME=${TASK_PREFIX}_${STAGE_NAME}
    FINAL_TASK_NAME=${STAGE_TASK_NAME}

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
        experiment=unlearn/tofu/selective_npo \
        trainer=NPO \
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
        trainer.args.eval_on_start=false \
        trainer.args.eval_strategy=no \
        trainer.args.save_strategy=epoch \
        trainer.args.save_total_limit=1 \
        trainer.args.save_only_model=false \
        trainer.args.ignore_data_skip=true \
        "${EXTRA_ARGS[@]}"

    PREV_OUTPUT_DIR=saves/unlearn/${STAGE_TASK_NAME}
done

CUDA_VISIBLE_DEVICES=${GPU_ID} python src/eval.py \
    experiment=eval/tofu/default.yaml \
    forget_split=${FORGET_SPLIT} \
    model=${MODEL} \
    task_name=${FINAL_TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=saves/unlearn/${FINAL_TASK_NAME} \
    paths.output_dir=saves/unlearn/${FINAL_TASK_NAME}/evals \
    retain_logs_path=${RETAIN_LOGS_PATH}

echo "Selective-NPO training completed. Final stage output: ${PREV_OUTPUT_DIR}"
