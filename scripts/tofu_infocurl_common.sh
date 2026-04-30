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
        echo "Unknown BACKEND=${backend}. Use ascend or cuda." >&2
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

score_subpool="${SCORE_SUBPOOL:-64}"
param_scope="${PARAM_SCOPE:-last_layer_lm_head}"
retain_batch_size="${RETAIN_BATCH_SIZE:-8}"
retain_ema_decay="${RETAIN_EMA_DECAY:-0.9}"
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

task_name_for() {
    local family="$1"
    local trainer="$2"
    local mode="$3"
    local gamma="$4"
    local k_steps="$5"
    local lam="$6"
    local schedule="$7"
    local stage2_gamma="$8"
    local switch_frac="$9"
    local transition_frac="${10}"
    local bs="${11}"
    local ga="${12}"

    if [[ "${trainer}" == "NPO" ]]; then
        echo "tofu_${model}_${forget_split}_NPO_${family}_${backend_tag}_bs${bs}ga${ga}_s${seed}"
        return
    fi

    local name="tofu_${model}_${forget_split}_${trainer}_${family}_mode${mode}_g$(fmt_num "${gamma}")_k${k_steps}"
    if [[ -n "${lam}" && "${lam}" != "0" && "${lam}" != "0.0" && "${lam}" != "0.00" ]]; then
        name+="_lam$(fmt_num "${lam}")"
    fi
    name+="_sp${score_subpool}"
    if [[ -n "${schedule}" ]]; then
        name+="_sched${schedule}_g2$(fmt_num "${stage2_gamma}")_sw$(fmt_num "${switch_frac}")"
        if [[ -n "${transition_frac}" ]]; then
            name+="_tr$(fmt_num "${transition_frac}")"
        fi
    fi
    name+="_${backend_tag}_bs${bs}ga${ga}_s${seed}"
    echo "${name}"
}

run_experiment() {
    local family="$1"
    local trainer="$2"
    local mode="${3:-}"
    local gamma="${4:-}"
    local k_steps="${5:-}"
    local lam="${6:-0.0}"
    local schedule="${7:-}"
    local stage2_gamma="${8:-}"
    local switch_frac="${9:-}"
    local transition_frac="${10:-}"
    local bs="${11:-8}"
    local ga="${12:-2}"

    local task_name
    task_name="$(task_name_for "${family}" "${trainer}" "${mode}" "${gamma}" "${k_steps}" "${lam}" "${schedule}" "${stage2_gamma}" "${switch_frac}" "${transition_frac}" "${bs}" "${ga}")"

    local summary_path="saves/unlearn/${task_name}/evals/TOFU_SUMMARY.json"
    if [[ -f "${summary_path}" ]]; then
        echo "SKIP existing ${task_name}"
        return
    fi

    local model_path tokenizer_path
    model_path="$(resolve_model_path)"
    tokenizer_path="$(resolve_tokenizer_path)"

    echo "================================================================"
    echo "Task    : ${task_name}"
    echo "Trainer : ${trainer}"
    echo "Setting : family=${family} mode=${mode} gamma=${gamma} K=${k_steps} lam=${lam}"
    echo "Backend : ${backend}"
    if [[ "${backend}" == "ascend" ]]; then
        echo "Device  : ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
    else
        echo "Device  : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    fi
    echo "================================================================"

    train_cmd=(
        python src/train.py
        --config-name=unlearn.yaml
        experiment=unlearn/tofu/default.yaml
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
        trainer.args.per_device_train_batch_size="${bs}"
        trainer.args.gradient_accumulation_steps="${ga}"
        trainer.args.report_to=none
        trainer.args.logging_steps=1
        trainer.args.do_eval=false
        trainer.args.eval_on_start=false
        trainer.args.eval_strategy=no
        trainer.args.save_strategy=no
        trainer.args.gradient_checkpointing=false
        trainer.args.ddp_find_unused_parameters=None
    )

    if [[ -n "${max_steps}" ]]; then
        train_cmd+=("+trainer.args.max_steps=${max_steps}")
    fi

    if [[ "${trainer}" == "NPO" ]]; then
        train_cmd+=(trainer=NPO)
    else
        train_cmd+=(
            trainer=InfoCURL_NPO
            collator=DataCollatorForSupervisedDatasetwithIndex
            trainer.method_args.sampler.mode="${mode}"
            trainer.method_args.sampler.gamma="${gamma}"
            trainer.method_args.sampler.lam="${lam}"
            trainer.method_args.sampler.K="${k_steps}"
            trainer.method_args.sampler.score_subpool="${score_subpool}"
            trainer.method_args.sampler.param_scope="${param_scope}"
            trainer.method_args.sampler.retain_batch_size="${retain_batch_size}"
            trainer.method_args.sampler.retain_ema_decay="${retain_ema_decay}"
        )
        if [[ -n "${schedule}" ]]; then
            train_cmd+=(
                +trainer.method_args.sampler.schedule="${schedule}"
                +trainer.method_args.sampler.stage1_mode=easy
                +trainer.method_args.sampler.stage2_mode=hard
                +trainer.method_args.sampler.stage1_gamma=1.0
                +trainer.method_args.sampler.stage2_gamma="${stage2_gamma}"
                +trainer.method_args.sampler.switch_frac="${switch_frac}"
                +trainer.method_args.sampler.transition_frac="${transition_frac:-0.30}"
            )
        fi
    fi

    if [[ "${dry_run}" == "1" ]]; then
        printf 'DRY_RUN train command:'
        printf ' %q' "${train_cmd[@]}"
        printf '\n'
        return
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
}

finish_infocurl_runs() {
    if [[ "${dry_run}" != "1" ]]; then
        python scripts/summarize_tofu10_runs.py
    fi
}
