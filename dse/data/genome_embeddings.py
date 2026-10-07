import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class GenomeEmbeddingDataset(Dataset):
    """Ragged per-genome records created by the autoencoder inference pipeline."""

    def __init__(self, data_dir, split, dtype=torch.float64):
        self.data_dir = Path(data_dir)
        self.split = split
        self.dtype = dtype
        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing genome embedding manifest: {manifest_path}")
        with manifest_path.open("r") as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("format_version") != 1:
            raise ValueError(
                f"Unsupported genome embedding format version: "
                f"{self.manifest.get('format_version')}"
            )
        if split not in self.manifest["splits"]:
            raise KeyError(f"Split '{split}' is not present in {manifest_path}")
        self.records = self.manifest["splits"][split]
        if not self.records:
            raise ValueError(f"Split '{split}' contains no genome records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        metadata = self.records[index]
        path = self.data_dir / metadata["path"]
        record = torch.load(path, map_location="cpu", weights_only=True)
        num_chunks, input_dim = record["chunk_embeddings"].shape
        if input_dim != self.manifest["input_dim"]:
            raise ValueError(
                f"Embedding dimension {input_dim} in {path} does not match manifest "
                f"dimension {self.manifest['input_dim']}"
            )
        for key in ("chromosome_ids", "position_ids", "chunk_starts", "chunk_lengths"):
            if record[key].numel() != num_chunks:
                raise ValueError(
                    f"{key} has {record[key].numel()} entries but {path} has "
                    f"{num_chunks} chunk embeddings"
                )
        return {
            "chunk_embeddings": record["chunk_embeddings"].to(self.dtype),
            "chromosome_ids": record["chromosome_ids"].long(),
            "position_ids": record["position_ids"].long(),
            "chunk_starts": record["chunk_starts"].long(),
            "chunk_lengths": record["chunk_lengths"].long(),
            "organism_index": int(record["organism_index"]),
            "temperature": float(record["temperature"]),
            "labels": record["labels"].to(self.dtype),
            "assembly_id": record["assembly_id"],
            "chromosome_names": record["chromosome_names"],
        }


class GenomeEmbeddingCollator:
    """Pad ragged genomes and retain an explicit valid-chunk mask."""

    def __init__(self, max_chunks=None):
        self.max_chunks = max_chunks

    def __call__(self, batch):
        batch_size = len(batch)
        input_dim = batch[0]["chunk_embeddings"].size(-1)
        num_chunks = max(item["chunk_embeddings"].size(0) for item in batch)
        if self.max_chunks is not None and num_chunks > self.max_chunks:
            raise ValueError(
                f"Batch requires {num_chunks} chunks but max_chunks={self.max_chunks}; "
                "refusing to silently truncate a genome"
            )

        dtype = batch[0]["chunk_embeddings"].dtype
        embeddings = torch.zeros(batch_size, num_chunks, input_dim, dtype=dtype)
        chromosome_ids = torch.full((batch_size, num_chunks), -1, dtype=torch.long)
        position_ids = torch.zeros(batch_size, num_chunks, dtype=torch.long)
        chunk_mask = torch.zeros(batch_size, num_chunks, dtype=torch.bool)

        for row, item in enumerate(batch):
            length = min(item["chunk_embeddings"].size(0), num_chunks)
            embeddings[row, :length] = item["chunk_embeddings"][:length]
            chromosome_ids[row, :length] = item["chromosome_ids"][:length]
            position_ids[row, :length] = item["position_ids"][:length]
            chunk_mask[row, :length] = True

        inputs = {
            "chunk_embeddings": embeddings,
            "chromosome_ids": chromosome_ids,
            "position_ids": position_ids,
            "chunk_mask": chunk_mask,
            "organism_indices": torch.tensor(
                [item["organism_index"] for item in batch], dtype=torch.long
            ),
            "temperatures": torch.tensor(
                [item["temperature"] for item in batch], dtype=dtype
            ),
        }
        labels = torch.stack([item["labels"] for item in batch])
        return inputs, labels
