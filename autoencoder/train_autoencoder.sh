#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"

python3 "${REPO_DIR}/autoencoder/train_autoencoder.py" \
    --data_dir "${REPO_DIR}/dse/data/reference/Microbial" \
    --ckpt_dir "${REPO_DIR}/autoencoder/checkpoints" \
    --log_dir "${REPO_DIR}/autoencoder/log" \
    --chunk_size 2048 \
    --batch_size 8 \
    --latent_dim 768 \
    --num_layers 4 \
    --learning_rate 3e-4 \
    --steps 10000 \
    --amp
