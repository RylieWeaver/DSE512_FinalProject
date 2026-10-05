import argparse
import csv
import hashlib
import json
import warnings
from pathlib import Path

import pysam
import torch
from tqdm import tqdm

from dse.data import BPTokenizer
from dse.distributed import resolve_device
from dse.model import AutoEncoderConfig, DNAAutoEncoder


SPLITS = ("train", "val", "test")


def parse_args():
    repo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Create ragged genome chunk-embedding datasets with a trained autoencoder."
    )
    parser.add_argument("--reference_dir", type=Path, default=repo_dir / "dse/data/reference/Microbial")
    parser.add_argument("--metadata_dir", type=Path, default=repo_dir / "dse/data/ribosomal")
    parser.add_argument("--autoencoder_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--metadata_pattern", default="iso_rib_temp_mod_{split}_std_norm.csv")
    parser.add_argument("--assembly_col", default="assembly_id")
    parser.add_argument("--organism_col", default="assembly_id")
    parser.add_argument("--temperature_col", default="growth_tmp")
    parser.add_argument("--target_cols", nargs="+", default=["log_dob_h"])
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlap in bases; defaults to chunk_size // 8.",
    )
    parser.add_argument("--inference_batch_size", type=int, default=32)
    parser.add_argument("--storage_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping missing FASTAs")
    return parser.parse_args()


def read_rows(path, required_columns):
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Missing columns {sorted(missing)} in {path}")
        return list(reader)


def load_autoencoder(checkpoint_dir, device):
    checkpoint_dir = checkpoint_dir.resolve()
    with (checkpoint_dir / "model_config.json").open("r") as handle:
        model_dict = json.load(handle)
    if model_dict.get("architecture_version") != 14:
        raise ValueError(
            "The supplied checkpoint uses an older autoencoder architecture. "
            "Retrain the dense, no-pooling autoencoder before generating embeddings."
        )
    config = AutoEncoderConfig(**model_dict)
    model = DNAAutoEncoder(config).to(device)
    state_dict = torch.load(
        checkpoint_dir / "model.pt", weights_only=True, map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def file_sha256(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def chunk_starts(sequence_length, chunk_size, stride):
    """Cover a sequence without crossing its FASTA-record boundary."""
    if sequence_length <= chunk_size:
        return [0]
    starts = list(range(0, sequence_length - chunk_size + 1, stride))
    final_start = sequence_length - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def encode_genome(fasta_path, model, tokenizer, chunk_size, stride, batch_size, device, amp):
    token_batches = []
    chromosome_ids = []
    position_ids = []
    starts = []
    lengths = []
    chromosome_names = []

    fasta = pysam.FastaFile(str(fasta_path))
    try:
        for chromosome_name in fasta.references:
            sequence_length = fasta.get_reference_length(chromosome_name)
            if sequence_length == 0:
                continue
            chromosome_index = len(chromosome_names)
            chromosome_names.append(chromosome_name)
            for position_index, start in enumerate(
                chunk_starts(sequence_length, chunk_size, stride)
            ):
                sequence = str(
                    fasta.fetch(chromosome_name, start, min(start + chunk_size, sequence_length))
                ).upper()
                token_ids = torch.full(
                    (chunk_size,), tokenizer.tok2id[tokenizer.PAD], dtype=torch.long
                )
                encoded = torch.tensor(tokenizer.encode(sequence), dtype=torch.long)
                token_ids[: encoded.numel()] = encoded
                token_batches.append(token_ids)
                chromosome_ids.append(chromosome_index)
                position_ids.append(position_index)
                starts.append(start)
                lengths.append(encoded.numel())
    finally:
        fasta.close()

    if not token_batches:
        raise ValueError(f"No non-empty FASTA records found in {fasta_path}")

    embeddings = []
    with torch.inference_mode():
        for offset in tqdm(
            range(0, len(token_batches), batch_size),
            desc=f"Embedding {fasta_path.stem}",
            leave=False,
        ):
            batch = torch.stack(token_batches[offset : offset + batch_size]).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=amp,
            ):
                embedding = model.encode(batch)  # [B, latent_dim]
            embeddings.append(embedding.float().cpu())

    return {
        "chunk_embeddings": torch.cat(embeddings, dim=0),
        "chromosome_ids": torch.tensor(chromosome_ids, dtype=torch.long),
        "position_ids": torch.tensor(position_ids, dtype=torch.long),
        "chunk_starts": torch.tensor(starts, dtype=torch.long),
        "chunk_lengths": torch.tensor(lengths, dtype=torch.long),
        "chromosome_names": chromosome_names,
    }


def main():
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    overlap = args.chunk_size // 8 if args.overlap is None else args.overlap
    if not 0 <= overlap < args.chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    stride = args.chunk_size - overlap

    args.reference_dir = args.reference_dir.resolve()
    args.metadata_dir = args.metadata_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model = load_autoencoder(args.autoencoder_checkpoint, device)
    if args.chunk_size != model.cfg.chunk_size:
        raise ValueError(
            f"Embedding chunk_size={args.chunk_size} does not match autoencoder "
            f"chunk_size={model.cfg.chunk_size}"
        )
    autoencoder_sha256 = file_sha256(args.autoencoder_checkpoint.resolve() / "model.pt")
    tokenizer = BPTokenizer()
    storage_dtype = getattr(torch, args.storage_dtype)

    required_columns = {
        args.assembly_col,
        args.organism_col,
        args.temperature_col,
        *args.target_cols,
    }
    rows_by_split = {
        split: read_rows(
            args.metadata_dir / args.metadata_pattern.format(split=split), required_columns
        )
        for split in SPLITS
    }
    organism_values = sorted(
        {row[args.organism_col] for rows in rows_by_split.values() for row in rows}
    )
    organism_to_index = {value: index for index, value in enumerate(organism_values)}

    manifest = {
        "format_version": 1,
        "storage": "ragged-per-genome-pt",
        "autoencoder_checkpoint": str(args.autoencoder_checkpoint.resolve()),
        "autoencoder_model_sha256": autoencoder_sha256,
        "autoencoder_model_config": model.cfg.to_dict(),
        "input_dim": model.cfg.latent_dim,
        "latent_pooling": "none-single-vector-bottleneck",
        "storage_dtype": args.storage_dtype,
        "chunk_size": args.chunk_size,
        "overlap": overlap,
        "stride": stride,
        "temperature_col": args.temperature_col,
        "target_cols": args.target_cols,
        "organism_col": args.organism_col,
        "organism_to_index": organism_to_index,
        "num_organisms": len(organism_to_index),
        "max_position_id": 0,
        "max_num_chunks": 0,
        "splits": {split: [] for split in SPLITS},
    }

    for split in SPLITS:
        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for row in tqdm(rows_by_split[split], desc=f"Building {split}"):
            assembly_id = row[args.assembly_col]
            fasta_path = args.reference_dir / split / f"{assembly_id}.fa"
            if not fasta_path.exists():
                message = f"Missing reference FASTA for {assembly_id}: {fasta_path}"
                if args.strict:
                    raise FileNotFoundError(message)
                warnings.warn(message)
                continue

            relative_path = Path(split) / f"{assembly_id}.pt"
            output_path = args.output_dir / relative_path
            if output_path.exists() and not args.overwrite:
                record = torch.load(output_path, map_location="cpu", weights_only=True)
            else:
                record = encode_genome(
                    fasta_path,
                    model,
                    tokenizer,
                    args.chunk_size,
                    stride,
                    args.inference_batch_size,
                    device,
                    args.amp,
                )
                record.update(
                    {
                        "format_version": 1,
                        "assembly_id": assembly_id,
                        "organism_index": organism_to_index[row[args.organism_col]],
                        "temperature": float(row[args.temperature_col]),
                        "labels": torch.tensor(
                            [float(row[column]) for column in args.target_cols],
                            dtype=torch.float32,
                        ),
                        "chunk_size": args.chunk_size,
                        "overlap": overlap,
                        "stride": stride,
                        "autoencoder_checkpoint": str(
                            args.autoencoder_checkpoint.resolve()
                        ),
                        "autoencoder_model_sha256": autoencoder_sha256,
                        "storage_dtype": args.storage_dtype,
                    }
                )
                record["chunk_embeddings"] = record["chunk_embeddings"].to(storage_dtype)
                torch.save(record, output_path)

            expected_record_settings = {
                "chunk_size": args.chunk_size,
                "overlap": overlap,
                "stride": stride,
                "autoencoder_checkpoint": str(args.autoencoder_checkpoint.resolve()),
                "autoencoder_model_sha256": autoencoder_sha256,
                "storage_dtype": args.storage_dtype,
                "organism_index": organism_to_index[row[args.organism_col]],
                "temperature": float(row[args.temperature_col]),
            }
            mismatches = {
                key: (record.get(key), expected)
                for key, expected in expected_record_settings.items()
                if record.get(key) != expected
            }
            if record["chunk_embeddings"].size(1) != model.cfg.latent_dim:
                mismatches["input_dim"] = (
                    record["chunk_embeddings"].size(1),
                    model.cfg.latent_dim,
                )
            expected_labels = torch.tensor(
                [float(row[column]) for column in args.target_cols], dtype=torch.float32
            )
            if not torch.equal(record["labels"].float(), expected_labels):
                mismatches["labels"] = (record["labels"].tolist(), expected_labels.tolist())
            if mismatches:
                raise ValueError(
                    f"Existing record {output_path} was built with different settings: "
                    f"{mismatches}. Re-run with --overwrite."
                )

            num_chunks = int(record["chunk_embeddings"].size(0))
            max_position = int(record["position_ids"].max().item())
            manifest["max_num_chunks"] = max(manifest["max_num_chunks"], num_chunks)
            manifest["max_position_id"] = max(manifest["max_position_id"], max_position)
            manifest["splits"][split].append(
                {
                    "assembly_id": assembly_id,
                    "path": str(relative_path),
                    "num_chunks": num_chunks,
                    "num_chromosomes": len(record["chromosome_names"]),
                    "max_position_id": max_position,
                }
            )

    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote genome embedding dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
