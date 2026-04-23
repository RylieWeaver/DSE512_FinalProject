#!/bin/bash

set -euo pipefail

# Path to executable script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/frontier_scaling_laws.sh"

# Export Env Vars
export ENV_DIR="/lustre/orion/lrn036/world-shared/rylieweaver/Environments/DSE512/dse"
export REPO_DIR="/ccs/home/rylieweaver/Scratch/DSE512_FinalProject"
export DATA_DIR="/lustre/orion/lrn036/proj-shared/rylieweaver/Datasets/Microbial/reference"
export CKPT_ROOT="/lustre/orion/lrn036/proj-shared/rylieweaver/checkpoints/Microbial/Pretrain"
export LOG_ROOT="${REPO_DIR}/experiments/log"

# Create mesh
## Mesh Dim 1
MODEL_DIMS=(64 128 256 512)
## Mesh Dim 2
CONTEXT_LENS=(64 128 256 512)
BATCH_SIZES=(1 1 1 1)
BATCHES_PER_STEP=(1 1 1 1)
SP_SIZES=(1 2 4 8)
DP_SIZES=(256 128 64 32)

# Check Mesh
if [[ ${#CONTEXT_LENS[@]} -ne ${#BATCHES_PER_STEP[@]} ]]; then
    echo "CONTEXT_LENS and BATCHES_PER_STEP must have the same length."
    exit 1
fi
if [[ ${#CONTEXT_LENS[@]} -ne ${#BATCH_SIZES[@]} ]]; then
    echo "CONTEXT_LENS and BATCH_SIZES must have the same length."
    exit 1
fi
if [[ ${#CONTEXT_LENS[@]} -ne ${#SP_SIZES[@]} ]]; then
    echo "CONTEXT_LENS and SP_SIZES must have the same length."
    exit 1
fi
if [[ ${#CONTEXT_LENS[@]} -ne ${#DP_SIZES[@]} ]]; then
    echo "CONTEXT_LENS and DP_SIZES must have the same length."
    exit 1
fi

# Submit jobs
for model_dim in "${MODEL_DIMS[@]}"; do
    for i in "${!CONTEXT_LENS[@]}"; do
        context_len="${CONTEXT_LENS[$i]}"
        batch_size="${BATCH_SIZES[$i]}"
        batches_per_step="${BATCHES_PER_STEP[$i]}"
        sp_size="${SP_SIZES[$i]}"
        dp_size="${DP_SIZES[$i]}"
        run_name="md${model_dim}_ctx${context_len}_bs${batch_size}_bps${batches_per_step}_sp${sp_size}_dp${dp_size}"

        sbatch \
            --job-name="${run_name}" \
            --output="${LOG_ROOT}/${run_name}-%j.o" \
            --error="${LOG_ROOT}/${run_name}-%j.e" \
            "${WORKER_SCRIPT}" "${model_dim}" "${context_len}" "${batch_size}" "${batches_per_step}" "${sp_size}" "${dp_size}"
    done
done
