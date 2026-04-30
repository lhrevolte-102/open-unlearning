#!/bin/bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

backend="${BACKEND:-cuda}"  # cuda | ascend
case "${backend}" in
    ascend)
        default_env_activate="/opt/conda/private/envs/open-unlearning-npu-venv/bin/activate"
        eval_device_map="npu:0"
        ;;
    cuda)
        default_env_activate="activate unlearning"
        eval_device_map="cuda:0"
        ;;
    *)
        echo "Unknown BACKEND=${backend}. Use cuda or ascend." >&2
        exit 2
        ;;
esac

env_activate="${ENV_ACTIVATE:-${default_env_activate}}"
if [[ "${backend}" == "cuda" && -z "${ENV_ACTIVATE:-}" ]]; then
    source activate unlearning
elif [[ -f "${env_activate}" ]]; then
    source "${env_activate}"
elif [[ -n "${env_activate}" ]]; then
    echo "WARN: env activate script not found: ${env_activate}" >&2
fi
unset PYTHONPATH

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [[ "${backend}" == "ascend" ]]; then
    if [[ -z "${ASCEND_VISIBLE_DEVICES:-}" && -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
        export ASCEND_RT_VISIBLE_DEVICES="${RT_DEVICE:-0}"
    fi
else
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
    fi
fi

model="${MODEL:-Llama-3.2-3B-Instruct}"
forget_split="${FORGET_SPLIT:-forget10}"
holdout_split="${HOLDOUT_SPLIT:-holdout10}"
retain_split="${RETAIN_SPLIT:-retain90}"
seed="${SEED:-0}"
run_eval="${RUN_EVAL:-1}"
dry_run="${DRY_RUN:-0}"
eval_batch_size="${EVAL_BATCH_SIZE:-32}"
max_steps="${MAX_STEPS:-}"
backend_tag="${BACKEND_TAG:-${backend}}"

train_batch_size="${TRAIN_BATCH_SIZE:-8}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"

ado_eta="${ADO_ETA:-1.0}"
ado_beta="${ADO_BETA:-0.9}"
ado_refresh_epochs="${ADO_REFRESH_EPOCHS:-1}"
ado_gain_floor="${ADO_GAIN_FLOOR:-0.0}"
ado_gain_clip="${ADO_GAIN_CLIP:-null}"
ado_prob_floor="${ADO_PROB_FLOOR:-0.0}"
ado_uniform_mix="${ADO_UNIFORM_MIX:-0.05}"
ado_log_every="${ADO_LOG_EVERY:-10}"

retain_logs_path="${RETAIN_LOGS_PATH:-saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json}"

fmt_num() {
    python - "$1" <<'PY'
import sys
print(f"{float(sys.argv[1]):.2f}".replace(".", "p"))
PY
}

resolve_model_path() {
    if [[ -n "${MODEL_PATH:-}" ]]; then
        echo "${MODEL_PATH}"
    else
        echo "open-unlearning/tofu_${model}_full"
    fi
}

resolve_tokenizer_path() {
    if [[ -n "${TOKENIZER_PATH:-}" ]]; then
        echo "${TOKENIZER_PATH}"
    else
        resolve_model_path
    fi
}

task_name="${TASK_NAME:-tofu_${model}_${forget_split}_ADO_NPO_eta$(fmt_num "${ado_eta}")_ema$(fmt_num "${ado_beta}")_mix$(fmt_num "${ado_uniform_mix}")_${backend_tag}_bs${train_batch_size}ga${gradient_accumulation_steps}_s${seed}}"
summary_path="saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json"

if [[ -f "${summary_path}" && "${FORCE:-0}" != "1" ]]; then
    echo "SKIP existing ${task_name}"
    exit 0
fi

model_path="$(resolve_model_path)"
tokenizer_path="$(resolve_tokenizer_path)"

echo "================================================================"
echo "Task       : ${task_name}"
echo "Trainer    : ADO_NPO"
echo "Model      : ${model_path}"
echo "Split      : forget=${forget_split} holdout=${holdout_split} retain=${retain_split}"
echo "ADO        : eta=${ado_eta} beta=${ado_beta} refresh_epochs=${ado_refresh_epochs} uniform_mix=${ado_uniform_mix}"
echo "Batch      : per_device=${train_batch_size} grad_accum=${gradient_accumulation_steps}"
echo "Backend    : ${backend}"
if [[ "${backend}" == "ascend" ]]; then
    echo "Device     : ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
else
    echo "Device     : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi
echo "================================================================"

train_cmd=(
    python src/train.py
    --config-name=unlearn.yaml
    experiment=unlearn/tofu/default.yaml
    trainer=ADO_NPO
    model="${model}"
    task_name="${task_name}"
    forget_split="${forget_split}"
    retain_split="${retain_split}"
    holdout_split="${holdout_split}"
    model.model_args.pretrained_model_name_or_path="${model_path}"
    model.tokenizer_args.pretrained_model_name_or_path="${tokenizer_path}"
    model.model_args.attn_implementation=eager
    retain_logs_path="${retain_logs_path}"
    trainer.args.seed="${seed}"
    trainer.args.optim=adamw_torch
    trainer.args.per_device_train_batch_size="${train_batch_size}"
    trainer.args.gradient_accumulation_steps="${gradient_accumulation_steps}"
    trainer.args.dataloader_num_workers=0
    trainer.args.report_to=none
    trainer.args.logging_steps=1
    trainer.args.do_eval=false
    trainer.args.eval_on_start=false
    trainer.args.eval_strategy=no
    trainer.args.save_strategy=no
    trainer.args.gradient_checkpointing=false
    trainer.args.ddp_find_unused_parameters=None
    trainer.method_args.sampler.eta="${ado_eta}"
    trainer.method_args.sampler.beta="${ado_beta}"
    trainer.method_args.sampler.refresh_epochs="${ado_refresh_epochs}"
    trainer.method_args.sampler.gain_floor="${ado_gain_floor}"
    trainer.method_args.sampler.gain_clip="${ado_gain_clip}"
    trainer.method_args.sampler.prob_floor="${ado_prob_floor}"
    trainer.method_args.sampler.uniform_mix="${ado_uniform_mix}"
    trainer.method_args.sampler.log_every="${ado_log_every}"
)

if [[ -n "${max_steps}" ]]; then
    train_cmd+=("+trainer.args.max_steps=${max_steps}")
fi

if [[ "${dry_run}" == "1" ]]; then
    printf 'DRY_RUN train command:'
    printf ' %q' "${train_cmd[@]}"
    printf '\n'
    exit 0
fi

"${train_cmd[@]}"

if [[ "${run_eval}" == "1" ]]; then
    python src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/tofu/default.yaml \
        model="${model}" \
        task_name="${task_name}_eval" \
        model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
        model.tokenizer_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
        model.model_args.device_map="${eval_device_map}" \
        model.model_args.attn_implementation=eager \
        forget_split="${forget_split}" \
        holdout_split="${holdout_split}" \
        retain_logs_path="${retain_logs_path}" \
        paths.output_dir="saves/unlearn/${task_name}/evals" \
        eval.tofu.batch_size="${eval_batch_size}"
fi
