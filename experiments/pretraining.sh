#!/bin/bash
#SBATCH -A LRN036
#SBATCH -J Pretraining
#SBATCH -o pretraining_log/%j/slurm.o
#SBATCH -e pretraining_log/%j/slurm.e
#SBATCH -t 02:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 32

# Pipe erros and disable core dumps
set -euo pipefail
ulimit -c 0

# Read args
MODEL_DIM=1536
CHUNK_SIZE=80000
CONTEXT_LEN=80000
BATCH_SIZE=1
BATCHES_PER_STEP=1
SP_SIZE=8
DP_SIZE=32

# World Configuration
export SLURM_NNODES="${SLURM_NNODES}"
export NGPUS_PER_NODE=8
export WORLD_SIZE=$((SLURM_NNODES * NGPUS_PER_NODE))
GPUS_PER_TASK=1
NTASKS=$((WORLD_SIZE / GPUS_PER_TASK))

# Check world size = SP_SIZE * DP_SIZE
if [[ $((SP_SIZE * DP_SIZE)) -ne $WORLD_SIZE ]]; then
    echo "Error: SP_SIZE * DP_SIZE must equal WORLD_SIZE (${SP_SIZE} * ${DP_SIZE} != ${WORLD_SIZE})"
    exit 1
fi

# Base paths (default to environment)
export ENV_DIR="/lustre/orion/lrn036/world-shared/rylieweaver/Environments/DSE512/dse"
export REPO_DIR="/ccs/home/rylieweaver/Scratch/DSE512_FinalProject"
export DATA_DIR="/lustre/orion/lrn036/proj-shared/rylieweaver/Datasets/Microbial/reference"
export CKPT_DIR="/lustre/orion/lrn036/proj-shared/rylieweaver/checkpoints/Microbial/Final_Pretrain"
export LOG_ROOT="${REPO_DIR}/experiments/pretraining_log"

# Run-specific paths
RUN_NAME="md${MODEL_DIM}_ctx${CONTEXT_LEN}_bs${BATCH_SIZE}_bps${BATCHES_PER_STEP}_sp${SP_SIZE}_dp${DP_SIZE}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

# Log experiment configuration
echo "Time: $(date)"
echo "RUN_NAME=${RUN_NAME}"
echo "MODEL_DIM=${MODEL_DIM}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "CONTEXT_LEN=${CONTEXT_LEN}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "BATCHES_PER_STEP=${BATCHES_PER_STEP}"
echo "SP_SIZE=${SP_SIZE}"
echo "DP_SIZE=${DP_SIZE}"

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

# Experiment mesh constants
LEARNING_RATE=1e-5
END_STEP=25000
WARMUP_STEPS=3000

# Run experiment
cd "${REPO_DIR}/experiments"
srun -N "${SLURM_NNODES}" --ntasks-per-node "${NGPUS_PER_NODE}" -c 7 --gpus-per-task="${GPUS_PER_TASK}" --gpu-bind=closest \
    python3 -W ignore -u train_mlm_distributed.py \
    --data_dir "${DATA_DIR}" \
    --ckpt_dir "${CKPT_DIR}" \
    --log_dir "${LOG_DIR}" \
    --chunk_size "${CHUNK_SIZE}" \
    --context_len "${CONTEXT_LEN}" \
    --model_dim "${MODEL_DIM}" \
    --learning_rate "${LEARNING_RATE}" \
    --end_step "${END_STEP}" \
    --batch_size "${BATCH_SIZE}" \
    --batches_per_step "${BATCHES_PER_STEP}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --data_parallel_size "${DP_SIZE}" \
    --sequence_parallel_size "${SP_SIZE}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    --resume_from "${CKPT_DIR}/step_7800" \
