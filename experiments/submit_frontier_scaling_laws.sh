#!/bin/bash

set -euo pipefail

# Path to executable script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/frontier_scaling_laws.sh"
LOG_ROOT="${SCRIPT_DIR}/slurm_logs"
mkdir -p "${LOG_ROOT}"

# Create mesh
MODEL_DIMS=(128 256 512 1024 2048)
CONTEXT_LENS=(128 256 512 1024 2048)
BATCHES_PER_STEP=(16 8 4 2 1)

# Check Mesh
if [[ ${#CONTEXT_LENS[@]} -ne ${#BATCHES_PER_STEP[@]} ]]; then
    echo "CONTEXT_LENS and BATCHES_PER_STEP must have the same length."
    exit 1
fi

# Submit jobs
for model_dim in "${MODEL_DIMS[@]}"; do
    for i in "${!CONTEXT_LENS[@]}"; do
        context_len="${CONTEXT_LENS[$i]}"
        batches_per_step="${BATCHES_PER_STEP[$i]}"
        run_name="md${model_dim}_ctx${context_len}_bps${batches_per_step}"

        sbatch \
            --job-name="${run_name}" \
            --output="${LOG_ROOT}/${run_name}-%j.o" \
            --error="${LOG_ROOT}/${run_name}-%j.e" \
            "${WORKER_SCRIPT}" "${model_dim}" "${context_len}" "${batches_per_step}"
    done
done
