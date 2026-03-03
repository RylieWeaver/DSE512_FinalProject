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

###### Helpful ######
# - squeue -u <userid>
# - scancel <job_id>
# - salloc -A ISAAC-UTK0448 -N 1 -n 1 -p short -q short -t 01:00:00
# - salloc -A ACF-UTK0011 -N 1 -n 1 -p campus-gpu -q campus-gpu -t 01:00:00 --gpus=1
# -- (--constraints gives cpu) (isaac-sinfo) (--gres=gpu:<type>:<number>) (--gres=gpu:v100s:1)
# - sbatch --array=<index_list> script.sh
# - singularity build container.sif container.def


# Setup
mkdir -p logs
mkdir -p data

# Log
echo "SLURM Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "SLURM Array Task Count: ${SLURM_ARRAY_TASK_COUNT:-N/A}"
echo "Hostname: $(hostname)"
echo "Time: $(date)"
