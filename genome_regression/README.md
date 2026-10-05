# Genome Chunk Regression

This pipeline freezes a trained DNA autoencoder, embeds overlapping chunks from
every reference genome, and trains a hierarchical model to predict standardized
`log_dob_h` from the resulting genome representation and standardized growth
temperature.

## 1. Build the embedding dataset

Download the reference FASTAs and train the autoencoder first. Then run:

```bash
bash genome_regression/build_embeddings.sh autoencoder/checkpoints/best
```

The default chunk overlap is `chunk_size // 8` (256 bases for a 2048-base
chunk). Override it directly when invoking Python:

```bash
python3 genome_regression/build_embeddings.py \
    --autoencoder_checkpoint autoencoder/checkpoints/best \
    --chunk_size 2048 \
    --overlap 256
```

Each FASTA record is chunked independently, so a chunk never crosses a record
boundary. The record name, per-record integer ID, chunk start, chunk length, and
within-record position are retained. The downloaded FASTA alone does not contain
reliable record-type metadata, so "chromosome" here means a FASTA sequence
record; filtering plasmids or scaffolds requires adding NCBI sequence-report
metadata.

The autoencoder produces one `[latent_dim]` vector for each input chunk. The
builder stores that vector directly without any additional pooling. Data is
stored as one ragged `.pt` record per assembly plus a `manifest.json`; it is
padded only when a batch is collated.

Existing records are reused only when the checkpoint, chunk size, overlap,
stride, and latent dimension match. Use `--overwrite` after changing any of
those settings.

## 2. Train regression

```bash
bash genome_regression/train_genome_regression.sh
```

The model applies chromosome context, genome context, learned absolute chunk
position, and a learned temperature projection. It then uses K learned filters
to attention-pool all valid chunks into K unordered tokens and processes those
tokens with ordinary transformer encoder layers. Padding logits are set to
negative infinity before softmax.

The organism index is retained as structural metadata but does not select an
embedding or any other organism-specific parameter. Each collated batch row is
one genome, so chromosome and whole-genome aggregation are defined by the row's
masks and chromosome IDs.
