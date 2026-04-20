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
INTRA_STAGE_ORDER="${INTRA_STAGE_ORDER:-}"
if [[ -n "$INTRA_STAGE_ORDER" ]]; then
    INTRA_STAGE_ORDERS="${INTRA_STAGE_ORDERS:-$INTRA_STAGE_ORDER}"
else
    INTRA_STAGE_ORDERS="${INTRA_STAGE_ORDERS:-random strict}"
fi
MAX_STAGE_ID="${MAX_STAGE_ID:-2}"
METHODS="${METHODS:-npo dpo}"

GPU_ID="${GPU_ID:-0}"
RESUME="${RESUME:-true}"

log() {
    echo "[Selective-NPO-Eval] $*"
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

tofu_eval_has_full_metrics() {
    local eval_file="$1"
    [[ -s "$eval_file" ]] || return 1
    python -c 'import json, sys
required = {
    "forget_Truth_Ratio",
    "forget_quality",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "model_utility",
    "privleak",
    "extraction_strength",
    "exact_memorization",
    "mia_min_k_plus_plus",
    "mia_min_k",
    "mia_loss",
    "mia_zlib",
    "mia_gradnorm",
    "mia_reference",
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
RETAIN_LOGS_PATH="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
TASK_PREFIX="tofu_${MODEL}_${FORGET_SPLIT}_selective_npo"

method_name_upper() {
    local method="$1"
    case "$method" in
        npo)
            echo "NPO"
            ;;
        dpo)
            echo "DPO"
            ;;
        *)
            echo "$method"
            ;;
    esac
}

method_task_prefix() {
    local method="$1"
    case "$method" in
        npo)
            echo "tofu_${MODEL}_${FORGET_SPLIT}_selective_npo"
            ;;
        dpo)
            echo "tofu_${MODEL}_${FORGET_SPLIT}_selective_dpo"
            ;;
        *)
            echo "Unsupported method '${method}'. Allowed values in METHODS: npo dpo"
            exit 1
            ;;
    esac
}

if is_truthy "$RESUME"; then
    log "Resume mode enabled. Existing full-metric eval logs for stage<=${MAX_STAGE_ID} will be skipped."
else
    log "Resume mode disabled. Target stage evals will be recomputed."
fi

found_target_stage=0
evaluated_count=0
skipped_count=0
found_any_combo=0

for INTRA_STAGE_ORDER in ${INTRA_STAGE_ORDERS}; do
    for METHOD in ${METHODS}; do
        METHOD="${METHOD,,}"
        TASK_PREFIX="$(method_task_prefix "${METHOD}")"
        STAGE_TASK_PREFIX="${TASK_PREFIX}_${INTRA_STAGE_ORDER}"
        STAGE_MANIFEST_DIR="saves/selective_stage/${STAGE_TASK_PREFIX}_stages/stages"

        if [[ ! -d "$STAGE_MANIFEST_DIR" ]]; then
            log "Skipping $(method_name_upper "${METHOD}") ${INTRA_STAGE_ORDER}; stage manifest directory not found: ${STAGE_MANIFEST_DIR}."
            continue
        fi

        found_any_combo=1
        found_target_stage_for_combo=0

        while IFS= read -r STAGE_MANIFEST; do
            STAGE_ID=$(python -c 'import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(int(payload["stage_id"]))' "$STAGE_MANIFEST")

            if (( STAGE_ID > MAX_STAGE_ID )); then
                continue
            fi

            found_target_stage=1
            found_target_stage_for_combo=1

            STAGE_NAME=$(python -c 'import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    payload = json.load(handle)
print(payload["stage_name"])' "$STAGE_MANIFEST")

            TASK_NAME="${STAGE_TASK_PREFIX}_${STAGE_NAME}"
            OUTPUT_DIR="saves/unlearn/${TASK_NAME}"
            EVAL_DIR="${OUTPUT_DIR}/evals"

            if ! model_dir_complete "$OUTPUT_DIR"; then
                log "Skipping $(method_name_upper "${METHOD}") ${INTRA_STAGE_ORDER} ${STAGE_NAME}; trained model output is incomplete or missing at ${OUTPUT_DIR}."
                skipped_count=$((skipped_count + 1))
                continue
            fi

            if is_truthy "$RESUME" && tofu_eval_has_full_metrics "${EVAL_DIR}/TOFU_EVAL.json"; then
                log "Skipping $(method_name_upper "${METHOD}") ${INTRA_STAGE_ORDER} ${STAGE_NAME}; found existing full-metric eval logs at ${EVAL_DIR}/TOFU_EVAL.json."
                skipped_count=$((skipped_count + 1))
                continue
            fi

            log "Evaluating $(method_name_upper "${METHOD}") ${INTRA_STAGE_ORDER} ${STAGE_NAME} from ${OUTPUT_DIR}."
            CUDA_VISIBLE_DEVICES=${GPU_ID} python src/eval.py \
                experiment=eval/tofu/full \
                forget_split=${FORGET_SPLIT} \
                model=${MODEL} \
                task_name=${TASK_NAME} \
                model.model_args.pretrained_model_name_or_path=${OUTPUT_DIR} \
                model.tokenizer_args.pretrained_model_name_or_path=${BASE_MODEL_PATH} \
                paths.output_dir=${EVAL_DIR} \
                retain_logs_path=${RETAIN_LOGS_PATH} \
                reference_model_path=${RETAIN_MODEL_PATH}

            evaluated_count=$((evaluated_count + 1))
        done < <(find "$STAGE_MANIFEST_DIR" -maxdepth 1 -type f -name 'stage[0-9]*.json' | sort -V)

        if [[ "$found_target_stage_for_combo" -eq 0 ]]; then
            log "No stage manifests with stage_id <= ${MAX_STAGE_ID} were found for $(method_name_upper "${METHOD}") ${INTRA_STAGE_ORDER} under ${STAGE_MANIFEST_DIR}."
        fi
    done
done

if [[ "$found_any_combo" -eq 0 ]]; then
    echo "No valid stage manifest directories were found. Check METHODS/MODEL/FORGET_SPLIT/INTRA_STAGE_ORDERS and run selective training first."
    exit 1
fi

if [[ "$found_target_stage" -eq 0 ]]; then
    echo "No stage manifests with stage_id <= ${MAX_STAGE_ID} were found across METHODS='${METHODS}'."
    exit 1
fi

log "Done. evaluated=${evaluated_count}, skipped=${skipped_count}, methods='${METHODS}', intra_stage_orders='${INTRA_STAGE_ORDERS}', max_stage_id=${MAX_STAGE_ID}."
