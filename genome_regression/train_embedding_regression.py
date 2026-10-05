import argparse
import json
import math
from pathlib import Path

import torch

from dse.data import GenomeEmbeddingCollator, GenomeEmbeddingDataset
from dse.distributed import rank0_print, resolve_device
from dse.model import EmbeddingSequenceRegressionTransformer, TransformerConfig
from dse.train import EmbeddingRegressionTrainer, SequenceRegressionTrainerConfig
from dse.utils import set_all_random_seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a standard transformer directly over stored chunk embeddings."
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        default=Path(__file__).parent / "embedding_transformer_checkpoints",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path(__file__).parent / "embedding_transformer_log",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_transformer_layers", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batches_per_step", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--resume_from", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batches_per_step < 1:
        raise ValueError("--batches_per_step must be positive")
    if args.batch_size != 1:
        raise ValueError("Direct embedding-transformer training currently requires --batch_size 1")
    args.data_dir = args.data_dir.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    set_all_random_seeds(args.seed)
    device = resolve_device(args.device)

    with (args.data_dir / "manifest.json").open("r") as handle:
        manifest = json.load(handle)
    datasets = {
        split: GenomeEmbeddingDataset(args.data_dir, split)
        for split in ("train", "val", "test")
    }
    collator = GenomeEmbeddingCollator()
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=split == "train",
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    if args.resume_from is None:
        model_cfg = TransformerConfig(
            vocab_size=1,
            input_dim=manifest["input_dim"],
            max_seq_len=manifest["max_num_chunks"],
            dim=args.dim,
            num_heads=args.num_heads,
            num_layers=args.num_transformer_layers,
            mlp_ratio=args.mlp_ratio,
            use_flash_attn=True,
            output_dim=len(manifest["target_cols"]),
            embed_dropout=args.dropout,
            attn_dropout=args.dropout,
            resid_dropout=args.dropout,
            head_dropout=args.dropout,
        )
        model = EmbeddingSequenceRegressionTransformer(model_cfg)
        trainer_cfg = SequenceRegressionTrainerConfig(
            log_every=1,
            eval_every=args.eval_every,
            batches_per_step=args.batches_per_step,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=max(
                args.epochs * math.ceil(len(loaders["train"]) / args.batches_per_step)
                - args.warmup_steps,
                1,
            ),
            decay_type="cosine",
            weight_decay=args.weight_decay,
            checkpoint_dir=args.ckpt_dir,
            log_dir=args.log_dir,
            save_every=args.save_every,
            amp_dtype="bfloat16",
            amp_enabled=args.amp,
        )
        trainer = EmbeddingRegressionTrainer(trainer_cfg, model, device=device)
        trainer._init_optimizer()
    else:
        trainer = EmbeddingRegressionTrainer.load_checkpoint(
            args.resume_from.resolve(), device=device
        )
        if trainer.model.cfg.input_dim != manifest["input_dim"]:
            raise ValueError("Checkpoint input_dim does not match the embedding manifest")
        if trainer.model.cfg.max_seq_len < manifest["max_num_chunks"]:
            raise ValueError("Checkpoint context length is shorter than this embedding dataset")

    trainer.set_loaders(loaders["train"], loaders["val"], loaders["test"])
    rank0_print(f"Device: {device}")
    rank0_print(f"Flash attention enabled: {trainer.model.backbone.use_flash_attn}")
    rank0_print(f"Number of model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    trainer.train(epochs=args.epochs)


if __name__ == "__main__":
    main()
