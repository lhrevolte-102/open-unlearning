#!/bin/bash
#SBATCH -J selective_npo
#SBATCH -p i64m1tga40u
#SBATCH -o selective_npo_%j.out
#SBATCH -e selective_npo_%j.err
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -D /hpc2hdd/home/yliu814/workspace/open-unlearning-sdpo

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
source .venv/bin/activate

echo "Job started at $(date)"
python community/methods/Selective-NPO/run.py
echo "Job ended at $(date)"
