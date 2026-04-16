#!/bin/bash
#SBATCH -J selective_npo
#SBATCH -p i64m1tga40u
#SBATCH -o selective_npo_%j.out
#SBATCH -e selective_npo_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/yliu814/workspace/open-unlearning-sdpo

set -euo pipefail

echo "Job started at $(date)"
bash community/methods/Selective-NPO/run.sh
echo "Job ended at $(date)"
