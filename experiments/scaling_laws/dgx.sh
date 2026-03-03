#!/usr/bin/env bash
set -euo pipefail


# Logging Setup
echo "Time: $(date)"

# Directory Setup (all relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../../:${PYTHONPATH:-}"
cd "$SCRIPT_DIR"
mkdir -p logs data

# Environment Setup
# source /lustre/isaac24/proj/UTK0448/DNA_LM_Group/Environments/<env_name>/bin/activate


# Make Experiment Mesh
DP_SIZE=8
SP_SIZE=1
MODEL_DIMS=(128 256 512 1024)
CONTEXT_LENS=(256 512 1024 2048)

# Run Experiments
for model_dim in "${MODEL_DIMS[@]}"; do
    for context_len in "${CONTEXT_LENS[@]}"; do
        # torchrun --standalone --nproc_per_node="$DP_SIZE" train_distributed.py \
        #     --model_dim "$model_dim" \
        #     --context_len "$context_len" \
        #     --data_parallel_size "$DP_SIZE" \
        #     --sequence_parallel_size "$SP_SIZE"
        python3 -c "print('model_dim=$model_dim context_len=$context_len dp=$DP_SIZE sp=$SP_SIZE')"  # Placeholder for the actual command (will print later)
    done
done
