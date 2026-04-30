#!/bin/bash

# Question: does retain-aware penalty help on top of soft easy-to-hard curriculum?
# Sweep lambda, switch point, transition length, and final hard-stage gamma.

source "$(dirname "$0")/tofu_infocurl_common.sh"

stage2_gammas=(0.10 0.14)
lams=(0.02 0.05)
switch_fracs=(0.20 0.30)
transition_fracs=(0.30 0.40)
train_batch_size=8
gradient_accumulation_steps=2
score_subpool=64

for stage2_gamma in "${stage2_gammas[@]}"; do
    for lam in "${lams[@]}"; do
        for switch_frac in "${switch_fracs[@]}"; do
            for transition_frac in "${transition_fracs[@]}"; do
                run_experiment soft_retain InfoCURL_NPO easy 1.00 20 "${lam}" soft_easy_to_hard "${stage2_gamma}" "${switch_frac}" "${transition_frac}" "${train_batch_size}" "${gradient_accumulation_steps}"
            done
        done
    done
done

finish_infocurl_runs
