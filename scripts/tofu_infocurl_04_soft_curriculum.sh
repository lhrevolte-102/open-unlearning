#!/bin/bash

# Question: does a gradual easy-to-hard curriculum help?
# Start easy, blend into hard over transition_frac, then stay hard.

source "$(dirname "$0")/tofu_infocurl_common.sh"

stage2_gammas=(0.08 0.10 0.12 0.14)
switch_fracs=(0.20 0.30)
transition_fracs=(0.20 0.30 0.40)
train_batch_size=8
gradient_accumulation_steps=2
score_subpool=64

for stage2_gamma in "${stage2_gammas[@]}"; do
    for switch_frac in "${switch_fracs[@]}"; do
        for transition_frac in "${transition_fracs[@]}"; do
            run_experiment soft_e2h InfoCURL_NPO easy 1.00 20 0.0 soft_easy_to_hard "${stage2_gamma}" "${switch_frac}" "${transition_frac}" "${train_batch_size}" "${gradient_accumulation_steps}"
        done
    done
done

finish_infocurl_runs
