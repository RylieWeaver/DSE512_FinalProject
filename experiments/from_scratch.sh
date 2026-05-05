#!/bin/bash
#SBATCH -A LRN036
#SBATCH -J FromScratch
#SBATCH -o from_scratch_log/%j/slurm.o
#SBATCH -e from_scratch_log/%j/slurm.e
#SBATCH -t 02:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 32

# salloc -A LRN036 -t 02:00:00 -q debug -N 1

# Pipe erros and disable core dumps
set -euo pipefail
ulimit -c 0

# World Configuration
export SLURM_NNODES="${SLURM_NNODES}"
export NGPUS_PER_NODE=8
export WORLD_SIZE=$((SLURM_NNODES * NGPUS_PER_NODE))
GPUS_PER_TASK=1
NTASKS=$((WORLD_SIZE / GPUS_PER_TASK))
DP_SIZE=32
SP_SIZE=8

# Check world size = SP_SIZE * DP_SIZE
if [[ $((SP_SIZE * DP_SIZE)) -ne $WORLD_SIZE ]]; then
    echo "Error: SP_SIZE * DP_SIZE must equal WORLD_SIZE (${SP_SIZE} * ${DP_SIZE} != ${WORLD_SIZE})"
    exit 1
fi

# Base paths (default to environment)
export ENV_DIR="/lustre/orion/lrn036/world-shared/rylieweaver/Environments/DSE512/dse"
source "${ENV_DIR}/bin/activate"
export REPO_DIR="/ccs/home/rylieweaver/Scratch/DSE512_FinalProject"
export DATA_DIR="${REPO_DIR}/dse/data/ribosomal/"
export CKPT_DIR="/lustre/orion/lrn036/proj-shared/rylieweaver/checkpoints/Microbial/FromScratch/"
export LOG_DIR="${REPO_DIR}/experiments/from_scratch_log"

# Modules
module load PrgEnv-gnu/8.6.0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

# Activate Environment
source "${ENV_DIR}/bin/activate"
PYTHONPATH="${PYTHONPATH:-}:${REPO_DIR}"
export PYTHONPATH


# Distributed Env Vars
export MASTER_ADDR
MASTER_ADDR="$(hostname -i)"
export MASTER_PORT=3442
export NCCL_SOCKET_IFNAME=hsn0

# Apparently needed for triton to avoid parallel Create/Delete contention
export TRITON_CACHE_DIR="/tmp/triton-cache-${SLURM_JOB_ID}-${SLURM_PROCID}"
rm -r -f "${TRITON_CACHE_DIR}"
mkdir -p "${TRITON_CACHE_DIR}"

# Needed to bypass MIOpen disk I/O errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
rm -rf "${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# Proxies
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=http://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

# Other Env Vars
export OMP_NUM_THREADS=1
# export NCCL_DEBUG=INFO

# Hyperparameters
CONTEXT_LEN=80000
MODEL_DIM=1536
EPOCHS=100
BACKBONE_LEARNING_RATE=1e-7
HEAD_LEARNING_RATE=1e-5
WARMUP_STEPS=100
EMBEDDING_DROPOUT=0.05
# Attention dropout has to be zero for AMD Triton Flash-Attn kernels
ATTENTION_DROPOUT=0.00
RESIDUAL_DROPOUT=0.1
HEAD_DROPOUT=0.2
BACKBONE_WEIGHT_DECAY=0.0
HEAD_WEIGHT_DECAY=1e-3


# Run experiment
cd "${REPO_DIR}/experiments"
srun -N "${SLURM_NNODES}" --ntasks-per-node "${NGPUS_PER_NODE}" -c 7 --gpus-per-task="${GPUS_PER_TASK}" --gpu-bind=closest \
    python3 -W ignore -u train_doubling_distributed.py \
    --data_dir "${DATA_DIR}" \
    --ckpt_dir "${CKPT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --context_len "${CONTEXT_LEN}" \
    --model_dim "${MODEL_DIM}" \
    --epochs "${EPOCHS}" \
    --backbone_learning_rate "${BACKBONE_LEARNING_RATE}" \
    --head_learning_rate "${HEAD_LEARNING_RATE}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --embed_dropout "${EMBEDDING_DROPOUT}" \
    --attn_dropout "${ATTENTION_DROPOUT}" \
    --resid_dropout "${RESIDUAL_DROPOUT}" \
    --head_dropout "${HEAD_DROPOUT}" \
    --backbone_weight_decay "${BACKBONE_WEIGHT_DECAY}" \
    --head_weight_decay "${HEAD_WEIGHT_DECAY}" \
    --data_parallel_size "${DP_SIZE}" \
    --sequence_parallel_size "${SP_SIZE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
