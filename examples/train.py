# General
import argparse
from pathlib import Path

# Torch
import torch

# DSE 512
from dse.train import MLMTrainerConfig, MLMTrainer
from dse.model import TransformerConfig, MLMTransformer
from dse.data import DNADataset, MLMCollator, create_random_dna_string
from dse.distributed import resolve_device
from dse.utils import set_all_random_seeds



# Commands:
# - python train.py --context_len 2048 --model_dim 1024


# Notes:
# - We forego a lot of things for simplicity here, including train/val/test splits, pathing, and
#   model hyperparameters.



if __name__ == "__main__":
    # Setup (just reading args here for flexibility in calling the script)
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=str(Path(__file__).parent.resolve()), help="Base directory for data and checkpoints")
    parser.add_argument("--context_len", type=int, default=2048, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=1024, help="Model dimension")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of linear warmup steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume training from checkpoint")
    args = parser.parse_args()
    base_dir = Path(args.base_dir).resolve()
    context_len = args.context_len
    model_dim = args.model_dim
    steps = args.steps
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    resume_from = args.resume_from
    set_all_random_seeds(42)

    # Device setup
    device = resolve_device()

    # Get dataset and loader
    data_path = base_dir / "dna.txt"
    """
    We must have n_bases >= context_len to ensure that we don't have to deal with padding, which would bloat the 
    code and detract from the learning purpose of this example. Additionally, if n_bases is too high, the user 
    may not see meaningful improvement over training, which detracts from seeing the training actually improve.
    Thus, we set n_bases to be just slightly larger than context_len (1% increase).
    """
    create_random_dna_string(data_path, n_bases=int(1.01 * context_len), seed=42)
    # NOTE: Dataset can be inspected with print(DNADataset.dna_string)
    dataset = DNADataset(path=data_path, chunk_size=context_len, base_seed=42)
    collator = MLMCollator(min_pad_length=context_len)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collator)

    # Train from scratch if no resume step is provided
    if not resume_from:
        ## Define the model
        model_cfg = TransformerConfig(
            vocab_size=dataset.tokenizer.out_vocab_size,
            max_seq_len=context_len,
            dim=model_dim,
            num_heads=8,
            num_layers=6,
            use_flash_attn=True,
        )
        model = MLMTransformer(model_cfg).to(device)
        ## Trainer configuration
        trainer_cfg = MLMTrainerConfig(
            log_every=1,
            eval_every=100,
            eval_batches=10,
            batches_per_step=1,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            checkpoint_dir=f"{base_dir}/checkpoints",
            save_every=1000,
            amp_dtype="bfloat16",
            amp_enabled=True
        )
        trainer = MLMTrainer(config=trainer_cfg, model=model, device=device)
        trainer._init_optimizer()
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = f"{base_dir}/checkpoints/{resume_from}"
        trainer = MLMTrainer.load_checkpoint(ckpt_dir, device)
        if context_len > trainer.model.cfg.max_seq_len:
            trainer.model._update_context_len(new_context_len=context_len)

    # Train the model
    trainer.set_loaders(train_loader, val_loader, test_loader)
    trainer.train(steps=steps)
