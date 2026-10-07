# Genome Regression

`run.sh` trains the autoencoders, builds their embedding caches, then runs:

| Run | Autoencoder bottleneck | Regressors | Regression learning rate | Regression precision |
|---|---|---|---|---|
| 1 | 10 (hidden width 256) | Width-2 MLP and transformer | `3e-3` | FP64 |
| 2 | 8192 | Width-2 MLP and transformer | `3e-3` | FP64 |
| 3 | 32 (hidden width 256) | Width-16 MLP and transformer | `3e-4` | FP64 |
| 4 | Same 8192 cache | Original width-768, 6-layer, 8-head transformer | `3e-5` | FP32 with CUDA BF16 AMP |

The small and medium autoencoders and their stored embeddings use FP64. Runs 2 and 4 share
an FP32 large autoencoder (with BF16 AMP on CUDA) and an FP16 embedding cache.
Run 2 converts those cached embeddings to FP64 for regression.
All autoencoders keep a learning rate of `3e-5`.

All runs use the existing training scripts. Defaults: seed 42, 10,000 autoencoder
steps, and 200 regression epochs without early stopping. Train/validation/test
are evaluated each epoch. Temperature and log doubling time use the original
normalized CSVs; embeddings are not normalized.

Run from the repository root:

```bash
DEVICE=cuda:0 bash genome_regression/run.sh
```

Edit settings at the top of `run.sh`, or set environment variables such as
`RUN_DIR`, `EPOCHS`, and `AE_STEPS`. To run one model, copy its command from the script.

To reuse trained autoencoders:

```bash
SMALL_AE_CHECKPOINT=/path/to/ae10/best \
MEDIUM_AE_CHECKPOINT=/path/to/ae32/step_10000 \
LARGE_AE_CHECKPOINT=/path/to/ae8192/step_10000 \
bash genome_regression/run.sh
```

Alternatively, set `LARGE_EMBEDDINGS=/path/to/cache` to skip both large-autoencoder
training and embedding generation.

Outputs go under `RUN_DIR` (default
`/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/genome_regression_seed42`), in
`scenario1/{mlp,transformer}`, `scenario2/{mlp,transformer}`,
`scenario3/{mlp,transformer}`, and `scenario4`.
Each model saves checkpoints, `history.csv`, and separate loss/R² PNG/PDF plots.
Use a new `RUN_DIR` to keep earlier results.

To regenerate plots:

```bash
python -m genome_regression.plot \
  --history /path/to/model/history.csv --name ae10_mlp
```
