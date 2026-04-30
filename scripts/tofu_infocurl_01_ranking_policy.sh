#!/bin/bash

# Question: which fixed ranking policy works best?
# Sweep easy/hard, InfoCURL score gamma, K re-rank interval, and score subpool.

source "$(dirname "$0")/tofu_infocurl_common.sh"

modes=(easy hard)
gammas=(0.08 0.10 0.12 0.14 0.16)
k_steps_list=(10 20 30)
score_subpools=(32 64)
train_batch_size=8
gradient_accumulation_steps=2

for mode in "${modes[@]}"; do
    for gamma in "${gammas[@]}"; do
        for k_steps in "${k_steps_list[@]}"; do
            for score_subpool in "${score_subpools[@]}"; do
                run_experiment fixed InfoCURL_NPO "${mode}" "${gamma}" "${k_steps}" 0.0 "" "" "" "" "${train_batch_size}" "${gradient_accumulation_steps}"
            done
        done
    done
done

finish_infocurl_runs
