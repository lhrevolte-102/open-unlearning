#!/bin/bash

# Reference point: plain NPO with the same TOFU/model/batch geometry.

source "$(dirname "$0")/tofu_infocurl_common.sh"

train_batch_size=8
gradient_accumulation_steps=2

run_experiment baseline NPO "" "" "" "" "" "" "" "" "${train_batch_size}" "${gradient_accumulation_steps}"

finish_infocurl_runs
