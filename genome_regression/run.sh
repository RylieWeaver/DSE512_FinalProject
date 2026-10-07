#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/DGX01/Personal/r9w/Environments/DSE/dse/bin/python}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# Edit these settings here, or supply them as environment variables.
DEVICE="${DEVICE:-cuda:0}"
RUN_DIR="${RUN_DIR:-/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/genome_regression_seed42}"
REFERENCE_DIR="${REFERENCE_DIR:-/mnt/DGX01/Personal/r9w/Datasets}"
METADATA_DIR="${METADATA_DIR:-${REPO_DIR}/dse/data/ribosomal}"
EPOCHS="${EPOCHS:-200}"
AE_STEPS="${AE_STEPS:-10000}"
SEED="${SEED:-42}"
SMALL_AE_CHECKPOINT="${SMALL_AE_CHECKPOINT:-}"
MEDIUM_AE_CHECKPOINT="${MEDIUM_AE_CHECKPOINT:-}"
LARGE_AE_CHECKPOINT="${LARGE_AE_CHECKPOINT:-}"
LARGE_EMBEDDINGS="${LARGE_EMBEDDINGS:-}"
SMALL_EMBEDDINGS="${RUN_DIR}/scenario1/embeddings"
MEDIUM_EMBEDDINGS="${RUN_DIR}/scenario3/embeddings"

if (( $# )); then
    echo "run.sh takes no arguments; set DEVICE, RUN_DIR, or checkpoint variables instead." >&2
    exit 2
fi
AMP=()
if [[ "${DEVICE}" == cuda* ]]; then
    AMP=(--amp)
fi

# Train the autoencoders first. Supplied checkpoints skip their training calls.
if [[ -z "${SMALL_AE_CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" -u -m autoencoder.train_autoencoder \
        --data_dir "${REFERENCE_DIR}" \
        --ckpt_dir "${RUN_DIR}/scenario1/autoencoder" \
        --log_dir "${RUN_DIR}/scenario1/autoencoder/log" \
        --latent_dim 10 --hidden_dim 256 --dtype float64 \
        --chunk_size 2048 --num_layers 2 --batch_size 32 \
        --learning_rate 3e-5 --warmup_steps 1000 \
        --steps "${AE_STEPS}" --save_every "${AE_STEPS}" --no_save_best \
        --device "${DEVICE}" --seed "${SEED}"
    SMALL_AE_CHECKPOINT="${RUN_DIR}/scenario1/autoencoder/step_${AE_STEPS}"
fi

if [[ -z "${MEDIUM_AE_CHECKPOINT}" ]]; then
    "${PYTHON_BIN}" -u -m autoencoder.train_autoencoder \
        --data_dir "${REFERENCE_DIR}" \
        --ckpt_dir "${RUN_DIR}/scenario3/autoencoder" \
        --log_dir "${RUN_DIR}/scenario3/autoencoder/log" \
        --latent_dim 32 --hidden_dim 256 --dtype float64 \
        --chunk_size 2048 --num_layers 2 --batch_size 32 \
        --learning_rate 3e-5 --warmup_steps 1000 \
        --steps "${AE_STEPS}" --save_every "${AE_STEPS}" --no_save_best \
        --device "${DEVICE}" --seed "${SEED}"
    MEDIUM_AE_CHECKPOINT="${RUN_DIR}/scenario3/autoencoder/step_${AE_STEPS}"
fi

if [[ -z "${LARGE_AE_CHECKPOINT}" && -z "${LARGE_EMBEDDINGS}" ]]; then
    "${PYTHON_BIN}" -u -m autoencoder.train_autoencoder \
        --data_dir "${REFERENCE_DIR}" \
        --ckpt_dir "${RUN_DIR}/large_autoencoder" \
        --log_dir "${RUN_DIR}/large_autoencoder/log" \
        --latent_dim 8192 --expansion_factor 2.0 --dtype float32 \
        --chunk_size 2048 --num_layers 2 --batch_size 32 \
        --learning_rate 3e-5 --warmup_steps 1000 \
        --steps "${AE_STEPS}" --save_every "${AE_STEPS}" --no_save_best \
        --device "${DEVICE}" --seed "${SEED}" "${AMP[@]}"
    LARGE_AE_CHECKPOINT="${RUN_DIR}/large_autoencoder/step_${AE_STEPS}"
fi

# Generate each cache once, using the original standardized phenotype CSVs.
"${PYTHON_BIN}" -u -m genome_regression.build_embeddings \
    --reference_dir "${REFERENCE_DIR}" --metadata_dir "${METADATA_DIR}" \
    --autoencoder_checkpoint "${SMALL_AE_CHECKPOINT}" \
    --output_dir "${SMALL_EMBEDDINGS}" \
    --chunk_size 2048 --overlap 256 --storage_dtype float64 \
    --overwrite --device "${DEVICE}"

"${PYTHON_BIN}" -u -m genome_regression.build_embeddings \
    --reference_dir "${REFERENCE_DIR}" --metadata_dir "${METADATA_DIR}" \
    --autoencoder_checkpoint "${MEDIUM_AE_CHECKPOINT}" \
    --output_dir "${MEDIUM_EMBEDDINGS}" \
    --chunk_size 2048 --overlap 256 --storage_dtype float64 \
    --overwrite --device "${DEVICE}"

if [[ -z "${LARGE_EMBEDDINGS}" ]]; then
    LARGE_EMBEDDINGS="${RUN_DIR}/large_embeddings"
    "${PYTHON_BIN}" -u -m genome_regression.build_embeddings \
        --reference_dir "${REFERENCE_DIR}" --metadata_dir "${METADATA_DIR}" \
        --autoencoder_checkpoint "${LARGE_AE_CHECKPOINT}" \
        --output_dir "${LARGE_EMBEDDINGS}" \
        --chunk_size 2048 --overlap 256 --storage_dtype float16 \
        --overwrite --device "${DEVICE}" "${AMP[@]}"
fi

# 1. Small autoencoder + small MLP / transformer.
"${PYTHON_BIN}" -u -m genome_regression.train_embedding_regression \
    --data_dir "${SMALL_EMBEDDINGS}" \
    --ckpt_dir "${RUN_DIR}/scenario1/mlp" --log_dir "${RUN_DIR}/scenario1/mlp/log" \
    --model mlp --dim 2 \
    --dropout 0.0 --learning_rate 3e-3 --weight_decay 0.0 \
    --warmup_steps 100 --batches_per_step 8 \
    --epochs "${EPOCHS}" --device "${DEVICE}" --seed "${SEED}"

"${PYTHON_BIN}" -u -m genome_regression.train_embedding_regression \
    --data_dir "${SMALL_EMBEDDINGS}" \
    --ckpt_dir "${RUN_DIR}/scenario1/transformer" --log_dir "${RUN_DIR}/scenario1/transformer/log" \
    --model transformer --dim 2 --num_heads 1 --num_transformer_layers 1 --mlp_ratio 1.0 \
    --dropout 0.0 --learning_rate 3e-3 --weight_decay 0.0 \
    --warmup_steps 100 --batches_per_step 8 \
    --epochs "${EPOCHS}" --device "${DEVICE}" --seed "${SEED}"

# 2. Large autoencoder + small MLP / transformer, sharing the large cache.
"${PYTHON_BIN}" -u -m genome_regression.train_embedding_regression \
    --data_dir "${LARGE_EMBEDDINGS}" \
    --ckpt_dir "${RUN_DIR}/scenario2/mlp" --log_dir "${RUN_DIR}/scenario2/mlp/log" \
    --model mlp --dim 2 \
    --dropout 0.0 --learning_rate 3e-3 --weight_decay 0.0 \
    --warmup_steps 100 --batches_per_step 8 \
    --epochs "${EPOCHS}" --device "${DEVICE}" --seed "${SEED}"

"${PYTHON_BIN}" -u -m genome_regression.train_embedding_regression \
    --data_dir "${LARGE_EMBEDDINGS}" \
    --ckpt_dir "${RUN_DIR}/scenario2/transformer" --log_dir "${RUN_DIR}/scenario2/transformer/log" \
    --model transformer --dim 2 --num_heads 1 --num_transformer_layers 1 --mlp_ratio 1.0 \
    --dropout 0.0 --learning_rate 3e-3 --weight_decay 0.0 \
    --warmup_steps 100 --batches_per_step 8 \
    --epochs "${EPOCHS}" --device "${DEVICE}" --seed "${SEED}"

# 3. Medium autoencoder + medium MLP / transformer.
"${PYTHON_BIN}" -u -m genome_regression.train_embedding_regression \
    --data_dir "${MEDIUM_EMBEDDINGS}" \
    --ckpt_dir "${RUN_DIR}/scenario3/mlp" --log_dir "${RUN_DIR}/scenario3/mlp/log" \
    --model mlp --dim 16 \
    --dropout 0.0 --learning_rate 3e-4 --weight_decay 0.0 \
    --warmup_steps 100 --batches_per_step 8 \
    --epochs "${EPOCHS}" --device "${DEVICE}" --seed "${SEED}"

"${PYTHON_BIN}" -u -m genome_regression.train_embedding_regression \
    --data_dir "${MEDIUM_EMBEDDINGS}" \
    --ckpt_dir "${RUN_DIR}/scenario3/transformer" --log_dir "${RUN_DIR}/scenario3/transformer/log" \
    --model transformer --dim 16 --num_heads 1 --num_transformer_layers 1 --mlp_ratio 1.0 \
    --dropout 0.0 --learning_rate 3e-4 --weight_decay 0.0 \
    --warmup_steps 100 --batches_per_step 8 \
    --epochs "${EPOCHS}" --device "${DEVICE}" --seed "${SEED}"

# 4. Large autoencoder + original large transformer (plot_this.txt settings).
"${PYTHON_BIN}" -u -m genome_regression.train_embedding_regression \
    --data_dir "${LARGE_EMBEDDINGS}" \
    --ckpt_dir "${RUN_DIR}/scenario4" --log_dir "${RUN_DIR}/scenario4/log" \
    --model transformer --dim 768 --num_heads 8 --num_transformer_layers 6 --mlp_ratio 4.0 --dtype float32 \
    --dropout 0.0 --learning_rate 3e-5 --weight_decay 0.0 \
    --warmup_steps 100 --batches_per_step 8 \
    --epochs "${EPOCHS}" --device "${DEVICE}" --seed "${SEED}" "${AMP[@]}"
