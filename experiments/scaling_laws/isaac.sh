#!/usr/bin/env bash
#SBATCH -J ScalingExperiment
#SBATCH -A ISAAC-UTK0448
#SBATCH -N 1
#SBATCH -p short
#SBATCH -q short
#SBATCH -t 00:05:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

# World Configuration

SLURM_NNODES="${SLURM_NNODES:-1}"
NUM_GPUS_PER_NODE=1
GPUS_PER_TASK=1
WORLD_SIZE=$((SLURM_NNODES * NUM_GPUS_PER_NODE))
NTASKS=$((WORLD_SIZE / GPUS_PER_TASK))
export SLURM_NNODES NUM_GPUS_PER_NODE GPUS_PER_TASK WORLD_SIZE NTASKS

# Environment Set Up
# source  /lustre/isaac24/proj/UTK0448/DNA_LM_Group/Environments/<env_name>/bin/activate


MODEL_DIMS=(128 256 512 1024)
CONTEXT_LENS=(256 512 1024 2048)

for dim in "${MODEL_DIMS[@]}"; do
  for ctx in "${CONTEXT_LENS[@]}"; do
    echo "=== Running model_dim=$dim, context_len=$ctx ==="

    #srun -N ${SLURM_NNODES} --ntasks-per-node ${NGPUS_PER_NODE} -c 7 --gpus-per-task=$GPUS_PER_TASK --gpu-bind=closest \
	#python train_distributed.py \
      	#--model_dim "$dim" \
      	#--context_len "$ctx" \
      	#--data_parallel_size 2 \
      	#--sequence_parallel_size 2
  done
done
