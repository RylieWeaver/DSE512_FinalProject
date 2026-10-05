#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"

python3 "${REPO_DIR}/genome_regression/train_genome_regression.py" \
    --data_dir "${REPO_DIR}/genome_regression/data" \
    --ckpt_dir "${REPO_DIR}/genome_regression/checkpoints" \
    --log_dir "${REPO_DIR}/genome_regression/log" \
    --batch_size 1 \
    --dim 256 \
    --num_filters 16 \
    --num_heads 8 \
    --num_transformer_layers 4 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --amp
