#!/bin/bash

# Question: does retain-aware conflict penalty improve utility under hard ranking?
# Sweep lambda while keeping mode=hard and K=20.

source "$(dirname "$0")/tofu_infocurl_common.sh"

gammas=(0.08 0.10 0.12 0.14)
lams=(0.02 0.05 0.10)
train_batch_size=8
gradient_accumulation_steps=2
score_subpool=64

for gamma in "${gammas[@]}"; do
    for lam in "${lams[@]}"; do
        run_experiment hard_retain InfoCURL_NPO hard "${gamma}" 20 "${lam}" "" "" "" "" "${train_batch_size}" "${gradient_accumulation_steps}"
    done
done

finish_infocurl_runs
