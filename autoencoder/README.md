# DNA Autoencoder Experiment

This is a simple MNIST-style DNA autoencoder. It one-hot encodes and flattens a
chunk, compresses it into exactly one latent vector, and decodes that vector
directly into one flat `chunk_size * 6` logit vector. There are no transformers,
pooling operations, or encoder-to-decoder skip connections. When `num_layers`
is greater than one, the additional hidden processing uses residual MLP blocks.
BatchNorm is confined to the residual branches; the identity paths, latent
vector, and final output are not normalized. Inference uses BatchNorm's running
EMA statistics.

Training minimizes raw MSE against the one-hot input vector. No softmax is
applied before MSE. Reconstruction accuracy is measured with `argmax`, and
padded positions are excluded from both loss and accuracy.

```text
[B, chunk_size * 6]
→ Linear(chunk_size * 6, expansion_factor * latent_dim)
→ GELU
→ pre-BatchNorm residual blocks × (num_layers - 1)
→ Linear(expansion_factor * latent_dim, latent_dim)
→ GELU
→ Linear(latent_dim, expansion_factor * latent_dim)
→ GELU
→ pre-BatchNorm residual blocks × (num_layers - 1)
→ Linear(expansion_factor * latent_dim, chunk_size * 6)
```

## Train

```bash
python3 autoencoder/train_autoencoder.py \
    --data_dir /mnt/DGX01/Personal/r9w/Datasets \
    --chunk_size 2048 \
    --batch_size 16 \
    --latent_dim 768 \
    --expansion_factor 1.0 \
    --num_layers 4 \
    --steps 10000 \
    --no_save_best \
    --amp
```

The chunk size is part of the architecture and must match during inference.
Use a fresh checkpoint directory. `--save_every` controls periodic checkpoints.
