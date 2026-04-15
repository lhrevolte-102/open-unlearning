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
REFERENCE_NUM_EPOCHS=${TOTAL_EPOCHS}
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_FOLDS=4
FOLD_ASSIGNMENT_SEED=0
BETA=0.1

GPU_ID="0"

########################################
# Derived paths
########################################
BASE_MODEL_PATH="open-unlearning/tofu_${MODEL}_full"
TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_references_${METHOD_VARIANT}"
RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
REFERENCE_DIR="saves/selective_refs/${TASK_PREFIX}"
FOLDS_DIR="${REFERENCE_DIR}/folds"
REFERENCE_MODELS_DIR="${REFERENCE_DIR}/models"
REFERENCE_MANIFEST_PATH="${REFERENCE_DIR}/reference_models.json"

case "$METHOD_VARIANT" in
    npo)
        REFERENCE_EXPERIMENT=selective/tofu/npo
        TRAIN_EXPERIMENT=unlearn/tofu/selective_npo
        TRAINER=NPO
        ;;
    idkdpo)
        REFERENCE_EXPERIMENT=selective/tofu/idkdpo
        TRAIN_EXPERIMENT=unlearn/tofu/selective_idk
        TRAINER=DPO
        ;;
    *)
        echo "Unsupported method variant: $METHOD_VARIANT"
        echo "Expected one of: npo, idkdpo"
        exit 1
        ;;
esac

mkdir -p "$FOLDS_DIR" "$REFERENCE_MODELS_DIR"

python src/selective_reference.py \
    experiment=${REFERENCE_EXPERIMENT} \
    task_name=${TASK_PREFIX}_folds \
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
    FOLD_TASK_NAME=${TASK_PREFIX}_fold${fold_id}
    FOLD_OUTPUT_DIR=${REFERENCE_MODELS_DIR}/fold${fold_id}

    CUDA_VISIBLE_DEVICES=${GPU_ID} python src/train.py --config-name=unlearn.yaml \
        experiment=${TRAIN_EXPERIMENT} \
        trainer=${TRAINER} \
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
        trainer.args.num_train_epochs=${REFERENCE_NUM_EPOCHS} \
        trainer.args.gradient_checkpointing=true \
        trainer.args.do_eval=false \
        trainer.args.save_strategy=no \
        trainer.args.save_only_model=true \
        paths.output_dir=${FOLD_OUTPUT_DIR}
done

python src/selective_reference.py \
    experiment=${REFERENCE_EXPERIMENT} \
    task_name=${TASK_PREFIX}_manifest \
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

echo "Selective ${METHOD_VARIANT} references completed. Manifest: ${REFERENCE_MANIFEST_PATH}"
