# Running Experiments

These scripts are set up to run on Frontier through Slurm. Run commands from the `experiments` directory unless otherwise noted:

```bash
cd /ccs/home/rylieweaver/Scratch/DSE512_FinalProject/experiments
```

Before submitting, check that the paths in each script point to the environment, repository, data, checkpoint, and log locations you want to use.


## Frontier Scaling Laws Experiment

The scaling-law sweep is submitted with:

```bash
bash submit_frontier_scaling_laws.sh
```

This wrapper builds a mesh of model sizes and context lengths, then submits one Slurm job per configuration using `frontier_scaling_laws.sh`.

Edit these path variables in `submit_frontier_scaling_laws.sh` before launching if needed:

- `ENV_DIR`: Python environment to activate.
- `REPO_DIR`: repository path.
- `DATA_DIR`: FASTA/reference data directory.
- `CKPT_ROOT`: root directory for checkpoints.
- `LOG_ROOT`: root directory for the Slurm output and error files submitted by the wrapper.

The worker script currently sets its per-run training logs under `${REPO_DIR}/experiments/log`.

Edit these arrays in `submit_frontier_scaling_laws.sh` to change the sweep:

- `MODEL_DIMS`: model dimensions to sweep over.
- `CONTEXT_LENS`: context lengths to sweep over.
- `BATCH_SIZES`: per-GPU batch size for each context length.
- `BATCHES_PER_STEP`: gradient accumulation count for each context length.
- `SP_SIZES`: sequence-parallel size for each context length.
- `DP_SIZES`: data-parallel size for each context length.

The `CONTEXT_LENS`, `BATCH_SIZES`, `BATCHES_PER_STEP`, `SP_SIZES`, and `DP_SIZES` arrays must all have the same length. Each entry at the same index is paired together. For Frontier's current 32-node setup in `frontier_scaling_laws.sh`, `SP_SIZE * DP_SIZE` must equal `WORLD_SIZE`, where `WORLD_SIZE = SLURM_NNODES * 8`.

The worker script receives these values as positional arguments:

```bash
sbatch frontier_scaling_laws.sh <model_dim> <context_len> <batch_size> <batches_per_step> <sp_size> <dp_size>
```

Usually you should submit through `submit_frontier_scaling_laws.sh`, but the direct `sbatch` form is useful for one-off scaling-law runs (such as if one run crashed).

If changing the actual training behavior, edit these constants in `frontier_scaling_laws.sh`:

- Slurm resources: `#SBATCH -A`, `#SBATCH -t`, `#SBATCH -p`, `#SBATCH -N`.
- `LEARNING_RATE`
- `STEPS`
- `WARMUP_STEPS`
- Module versions and proxy settings if Frontier's software stack changes.


## Pretrain -> Finetune Experiment

### Pretrain

Submit the single pretraining job with:

```bash
sbatch pretraining.sh
```

The main run settings are hard-coded near the top of `pretraining.sh`:

- `MODEL_DIM`: transformer model dimension.
- `CHUNK_SIZE`: FASTA chunk size passed to `train_mlm_distributed.py`.
- `CONTEXT_LEN`: model context length.
- `BATCH_SIZE`: per-GPU batch size.
- `BATCHES_PER_STEP`: gradient accumulation count.
- `SP_SIZE`: sequence-parallel size.
- `DP_SIZE`: data-parallel size.

As with the scaling-law runs, `SP_SIZE * DP_SIZE` must equal the Slurm world size. With `#SBATCH -N 32` and `NGPUS_PER_NODE=8`, the world size is 256.

Check these paths before submitting:

- `ENV_DIR`
- `REPO_DIR`
- `DATA_DIR`
- `CKPT_DIR`
- `LOG_ROOT`

Change these training constants in `pretraining.sh` when adjusting the schedule:

- `LEARNING_RATE`
- `END_STEP`
- `WARMUP_STEPS`
- `--resume_from "${CKPT_DIR}/*"`: uncomment and update this if resuming from a checkpoint after a crash or scheduled-time limit.

The script launches `train_mlm_distributed.py`. The shell variables above are passed through as CLI args such as `--data_dir`, `--ckpt_dir`, `--log_dir`, `--chunk_size`, `--context_len`, `--model_dim`, `--learning_rate`, `--end_step`, `--batch_size`, `--batches_per_step`, `--warmup_steps`, `--data_parallel_size`, and `--sequence_parallel_size`. If checkpoint resume is enabled, it also passes `--resume_from`.

### Finetune

Submit the finetuning job with:

```bash
sbatch finetune.sh
```

Check these paths before submitting:

- `ENV_DIR`: Python environment to activate.
- `REPO_DIR`: repository path.
- `DATA_DIR`: finetuning dataset directory.
- `CKPT_DIR`: output checkpoint directory for the finetuned model.
- `FINETUNE_FROM`: pretrained checkpoint to initialize from.
- `LOG_DIR`: log directory.

Edit these parallelism settings in `finetune.sh` if the job size changes:

- `#SBATCH -N`
- `DP_SIZE`
- `SP_SIZE`

Again, `SP_SIZE * DP_SIZE` must equal `SLURM_NNODES * 8`.

You can edit these model and training hyperparameters in `finetune.sh`:

- `CONTEXT_LEN`
- `MODEL_DIM`
- `EPOCHS`
- `BACKBONE_LEARNING_RATE`
- `HEAD_LEARNING_RATE`
- `WARMUP_STEPS`
- `EMBEDDING_DROPOUT`
- `ATTENTION_DROPOUT`
- `RESIDUAL_DROPOUT`
- `HEAD_DROPOUT`
- `BACKBONE_WEIGHT_DECAY`
- `HEAD_WEIGHT_DECAY`

The script launches `train_doubling_distributed.py`. The shell variables above are passed through as CLI args such as `--data_dir`, `--ckpt_dir`, `--log_dir`, `--context_len`, `--model_dim`, `--epochs`, `--backbone_learning_rate`, `--head_learning_rate`, `--warmup_steps`, `--embed_dropout`, `--attn_dropout`, `--resid_dropout`, `--head_dropout`, `--backbone_weight_decay`, `--head_weight_decay`, `--finetune_from`, `--data_parallel_size`, and `--sequence_parallel_size`.

Note that `ATTENTION_DROPOUT` is currently set to `0.00` because the AMD Triton Flash-Attn kernels on Frontier require attention dropout to be zero.
