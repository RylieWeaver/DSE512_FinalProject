#!/bin/bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"

AUTOENCODER_CHECKPOINT="${1:-${REPO_DIR}/autoencoder/checkpoints/best}"

python3 "${REPO_DIR}/genome_regression/build_embeddings.py" \
    --reference_dir "${REPO_DIR}/dse/data/reference/Microbial" \
    --metadata_dir "${REPO_DIR}/dse/data/ribosomal" \
    --autoencoder_checkpoint "${AUTOENCODER_CHECKPOINT}" \
    --output_dir "${REPO_DIR}/genome_regression/data" \
    --chunk_size 2048 \
    --inference_batch_size 32 \
    --amp
