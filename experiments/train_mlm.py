# General
import argparse
from pathlib import Path

# Torch
import torch

# DSE 512
from dse.train import TrainerConfig, MLMTrainer
from dse.model import TransformerConfig, MLMTransformer
from dse.data import FASTADataset, MLMCollator, BPTokenizer
from dse.distributed import resolve_device, rank0_print



# Commands:
# - python train_mlm.py --context_len 2048 --model_dim 1024


if __name__ == "__main__":
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_len", type=int, default=2048, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of linear warmup steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume training from checkpoint")
    args = parser.parse_args()
    context_len = args.context_len
    model_dim = args.model_dim
    steps = args.steps
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    resume_from = args.resume_from
    ckpt_dir = Path("/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/scratch").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = resolve_device()

    # Get datasets and loaders
    data_dir = Path("/mnt/DGX01/Personal/r9w/Datasets/Microbial")
    tokenizer = BPTokenizer()
    train_dataset = FASTADataset(fasta_dir=(data_dir / "train"), chunk_size=context_len, tokenizer=tokenizer)
    val_dataset = FASTADataset(fasta_dir=(data_dir / "val"), chunk_size=context_len, tokenizer=tokenizer)
    test_dataset = FASTADataset(fasta_dir=(data_dir / "test"), chunk_size=context_len, tokenizer=tokenizer)
    collator = MLMCollator(tokenizer=tokenizer, min_pad_length=context_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=collator)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collator)

    # Train from scratch if no resume step is provided
    if not resume_from:
        ## Define the model
        model_cfg = TransformerConfig(
            vocab_size=tokenizer.out_vocab_size,
            max_seq_len=context_len,
            dim=model_dim,
            num_heads=8,
            num_layers=24,
            use_flash_attn=True,
        )
        model = MLMTransformer(model_cfg).to(device)
        ## Trainer configuration
        trainer_cfg = TrainerConfig(
            log_every=1,
            eval_every=100,
            eval_batches=10,
            batches_per_step=1,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            checkpoint_dir=ckpt_dir,
            save_every=1000,
            amp_dtype="bfloat16",
            amp_enabled=True
        )
        trainer = MLMTrainer(config=trainer_cfg, model=model, device=device)
        trainer._init_optimizer()
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = ckpt_dir / resume_from
        trainer = MLMTrainer.load_checkpoint(ckpt_dir, device)
        if context_len > trainer.model.cfg.max_seq_len:
            trainer.model._update_context_len(new_context_len=context_len)

    # Train the model
    trainer.set_loaders(train_loader, val_loader, test_loader)
    rank0_print(f"Number of Model Parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    trainer.train(steps=steps)
