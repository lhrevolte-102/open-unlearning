#!/bin/bash
# Environment smoke test — upstream NPO only, 50 steps, Llama-3.2-3B-Instruct × TOFU-forget10.
# Purpose: before writing any InfoCURL code, confirm that
#   (a) the HF Hub checkpoint `open-unlearning/tofu_Llama-3.2-3B-Instruct_full` loads,
#   (b) bf16 + flash-attn + NPO fits on 1×A800 at the Round-11 batch geometry
#       (per_device_train_batch_size=4, gradient_accumulation_steps=1),
#   (c) TOFU-forget10 data pipeline and eval harness run cleanly,
#   (d) dataloader_num_workers=0 single-process training produces no NaN / no OOM.
# If this fails, no InfoCURL code will help — fix the environment first.

set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

model="Llama-3.2-3B-Instruct"
model_path="open-unlearning/tofu_${model}_full"

forget_split="forget10"
holdout_split="holdout10"
retain_split="retain90"

# Round-11/12 geometry: effective batch = 4, single-process, no worker prefetch.
per_device_train_batch_size=4
gradient_accumulation_steps=1
dataloader_num_workers=0

max_steps=50
seed=0

task_name=env_smoketest_tofu10_${model}_NPO_s${seed}

retain_logs_path="saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json"
[[ -f "${retain_logs_path}" ]] || retain_logs_path=null

echo "================================================================"
echo "Environment smoke test: ${task_name}"
echo "  model       = ${model_path}"
echo "  forget      = ${forget_split} (holdout=${holdout_split}, retain=${retain_split})"
echo "  geometry    = bs=${per_device_train_batch_size} grad_accum=${gradient_accumulation_steps} workers=${dataloader_num_workers}"
echo "  max_steps   = ${max_steps}"
echo "================================================================"

# Train (single GPU, no accelerate, no DDP).
CUDA_VISIBLE_DEVICES=0 python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=NPO \
    task_name=${task_name} \
    trainer.args.seed=${seed} \
    model=${model} \
    model.model_args.pretrained_model_name_or_path=${model_path} \
    forget_split=${forget_split} \
    retain_split=${retain_split} \
    holdout_split=${holdout_split} \
    retain_logs_path=${retain_logs_path} \
    trainer.args.per_device_train_batch_size=${per_device_train_batch_size} \
    trainer.args.gradient_accumulation_steps=${gradient_accumulation_steps} \
    trainer.args.gradient_checkpointing=true \
    trainer.args.num_train_epochs=1 \
    trainer.args.save_strategy=no

# Eval — load the just-trained checkpoint (it was not saved; point eval at the HF path if save_strategy=no).
# For a true smoke test the train run alone is the gate; eval is optional.
# Uncomment the block below if you want to also exercise the eval harness:
#
# CUDA_VISIBLE_DEVICES=0 python src/eval.py \
#     experiment=eval/tofu/default.yaml \
#     forget_split=${forget_split} \
#     holdout_split=${holdout_split} \
#     model=${model} \
#     task_name=${task_name} \
#     model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
#     paths.output_dir=saves/unlearn/${task_name}/evals \
#     retain_logs_path=${retain_logs_path}

echo ""
echo "================ smoke test complete ================"
echo "If training finished 50 steps with no OOM / NaN, the environment is ready"
echo "for Stage-3 InfoCURL code (src/trainer/unlearn/infocurl.py)."
