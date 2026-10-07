import argparse
from pathlib import Path

import torch

from dse.data import AutoEncoderCollator, BPTokenizer, FASTADataset
from dse.distributed import rank0_print, resolve_device
from dse.model import AutoEncoderConfig, DNAAutoEncoder
from dse.train import AutoEncoderTrainer, AutoEncoderTrainerConfig
from dse.utils import set_all_random_seeds


def parse_args(argv=None):
    repo_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Train a flattened single-vector DNA autoencoder.")
    parser.add_argument("--data_dir", type=Path, default=repo_dir / "dse/data/reference/Microbial")
    parser.add_argument("--ckpt_dir", type=Path, default=Path(__file__).parent / "checkpoints")
    parser.add_argument("--log_dir", type=Path, default=Path(__file__).parent / "log")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--expansion_factor", type=float, default=4.0, help="Hidden width as a multiple of latent_dim")
    parser.add_argument("--num_layers", type=int, default=1, help="One projection layer plus num_layers-1 residual blocks per side")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--batches_per_step", type=int, default=1)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument(
        "--no_save_best",
        action="store_true",
        help="Disable rewriting the best checkpoint after validation improvements",
    )
    parser.add_argument("--resume_from", type=Path, default=None, help="Checkpoint directory, e.g. checkpoints/step_1000")
    parser.add_argument("--device", type=str, default=None, help="For example: cpu, cuda, or cuda:0")
    parser.add_argument("--amp", action="store_true", help="Enable bfloat16 autocast")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def require_data(data_dir):
    missing = [split for split in ("train", "val", "test") if not any((data_dir / split).glob("*.fa"))]
    if missing:
        raise SystemExit(
            f"No FASTA files found for {missing} under {data_dir}. "
            "Run `bash commands.txt` from the repository root, or pass --data_dir."
        )


def run(args):
    if args.dtype == "float64" and args.amp:
        raise ValueError("float64 training must not use AMP")
    if args.batches_per_step < 1:
        raise ValueError("--batches_per_step must be positive")
    args.data_dir = args.data_dir.resolve()
    args.ckpt_dir = args.ckpt_dir.resolve()
    args.log_dir = args.log_dir.resolve()
    require_data(args.data_dir)
    set_all_random_seeds(args.seed)
    device = resolve_device(args.device)

    tokenizer = BPTokenizer()
    datasets = {
        split: FASTADataset(
            fasta_dir=args.data_dir / split,
            chunk_size=args.chunk_size,
            tokenizer=tokenizer,
            base_seed=args.seed + split_index,
        )
        for split_index, split in enumerate(("train", "val", "test"))
    }
    collator = AutoEncoderCollator(tokenizer=tokenizer, max_pad_length=args.chunk_size)
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    if args.resume_from is None:
        model_cfg = AutoEncoderConfig(
            vocab_size=tokenizer.out_vocab_size,
            chunk_size=args.chunk_size,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            dtype=args.dtype,
            expansion_factor=args.expansion_factor,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_token_id=tokenizer.tok2id[tokenizer.PAD],
        )
        model = DNAAutoEncoder(model_cfg)
        trainer_cfg = AutoEncoderTrainerConfig(
            log_every=args.log_every,
            eval_every=args.eval_every,
            eval_batches=args.eval_batches,
            batches_per_step=args.batches_per_step,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=max(args.steps - args.warmup_steps, 1),
            decay_type="cosine",
            checkpoint_dir=args.ckpt_dir,
            log_dir=args.log_dir,
            save_every=args.save_every,
            save_best=not args.no_save_best,
            amp_dtype="bfloat16",
            amp_enabled=args.amp,
        )
        trainer = AutoEncoderTrainer(trainer_cfg, model, device=device)
        trainer._init_optimizer()
    else:
        trainer = AutoEncoderTrainer.load_checkpoint(args.resume_from.resolve(), device=device)
        if trainer.model.cfg.dtype != args.dtype:
            raise ValueError(f"Checkpoint dtype={trainer.model.cfg.dtype}, but --dtype={args.dtype}")
        if args.hidden_dim is not None and trainer.model.hidden_dim != args.hidden_dim:
            raise ValueError("Checkpoint hidden width differs from --hidden_dim")
        if trainer.model.cfg.chunk_size != args.chunk_size:
            raise ValueError(
                f"Checkpoint chunk_size={trainer.model.cfg.chunk_size}, but "
                f"--chunk_size={args.chunk_size}"
            )

    trainer.set_loaders(loaders["train"], loaders["val"], loaders["test"])
    rank0_print(f"Device: {device}")
    rank0_print(f"Single-vector latent shape: [batch, {trainer.model.cfg.latent_dim}]")
    rank0_print(f"Number of model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    trainer.train(steps=args.steps)


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
