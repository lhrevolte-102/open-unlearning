#!/bin/bash

set -euo pipefail

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
echo "Master Port: $MASTER_PORT"
echo "ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>}"

models=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-3B-Instruct"
    "Llama-3.1-8B-Instruct"
)
trainers_experiments=(
    "GradAscent unlearn/tofu/default.yaml"
    "GradDiff unlearn/tofu/default.yaml"
    "NPO unlearn/tofu/default.yaml"
    "DPO unlearn/tofu/idk.yaml"
)
splits=(
    "forget01 holdout01 retain99"
    "forget05 holdout05 retain95"
    "forget10 holdout10 retain90"
)

per_device_train_batch_size=4
gradient_accumulation_steps=4
attn_implementation=eager
optim=adamw_torch

for split in "${splits[@]}"; do
    forget_split=$(echo "$split" | cut -d' ' -f1)
    holdout_split=$(echo "$split" | cut -d' ' -f2)
    retain_split=$(echo "$split" | cut -d' ' -f3)

    for model in "${models[@]}"; do
        for trainer_experiment in "${trainers_experiments[@]}"; do
            trainer=$(echo "$trainer_experiment" | cut -d' ' -f1)
            experiment=$(echo "$trainer_experiment" | cut -d' ' -f2)

            task_name=tofu_${model}_${forget_split}_${trainer}_ascend
            model_path=open-unlearning/tofu_${model}_full
            echo "${task_name}: Unlearning ${model_path} using ${trainer}"

            accelerate launch --config_file configs/accelerate/ascend_multi_npu.yaml --main_process_port "$MASTER_PORT" \
            src/train.py --config-name=unlearn.yaml \
            experiment="${experiment}" \
            trainer="${trainer}" \
            task_name="${task_name}" \
            model="${model}" \
            forget_split="${forget_split}" \
            retain_split="${retain_split}" \
            model.model_args.pretrained_model_name_or_path="${model_path}" \
            model.model_args.attn_implementation="${attn_implementation}" \
            retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json" \
            trainer.args.optim="${optim}" \
            trainer.args.per_device_train_batch_size="$per_device_train_batch_size" \
            trainer.args.gradient_accumulation_steps="$gradient_accumulation_steps" \
            trainer.args.ddp_find_unused_parameters=true \
            trainer.args.gradient_checkpointing=false

            python src/eval.py \
            experiment=eval/tofu/default.yaml \
            forget_split="${forget_split}" \
            holdout_split="${holdout_split}" \
            model="${model}" \
            task_name="${task_name}" \
            model.model_args.attn_implementation="${attn_implementation}" \
            model.model_args.pretrained_model_name_or_path="saves/unlearn/${task_name}" \
            paths.output_dir="saves/unlearn/${task_name}/evals" \
            retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
        done
    done
done
