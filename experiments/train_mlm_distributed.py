# General
import argparse
from pathlib import Path

# Torch
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# DSE 512
from dse.train import TrainerConfig, MLMTrainer
from dse.model import TransformerConfig, MLMTransformer
from dse.data import FASTADataset, MLMCollator, BPTokenizer
from dse.distributed import init_parallel_state, resolve_device, rank0_print



# Commands:
# - torchrun --standalone --nproc_per_node=4 train_mlm_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=4 train_mlm_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'train_mlm_distributed.py'


if __name__ == "__main__":
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/DGX01/Personal/r9w/Datasets/Microbial", help="Directory for data")
    parser.add_argument("--ckpt_dir", type=str, default="/mnt/DGX01/Personal/r9w/Checkpoints/Microbial", help="Directory for checkpoints")
    parser.add_argument("--context_len", type=int, default=8192, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=1536, help="Model dimension")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of linear warmup steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume training from checkpoint")
    parser.add_argument("--data_parallel_size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--sequence_parallel_size", type=int, default=2, help="Sequence parallel size")
    parser.add_argument("--master_addr", type=str, default=None, help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, default=None, help="Master port for distributed training")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    context_len = args.context_len
    model_dim = args.model_dim
    steps = args.steps
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    resume_from = args.resume_from
    dp_size = args.data_parallel_size
    sp_size = args.sequence_parallel_size
    master_addr = args.master_addr
    master_port = args.master_port

    # Distributed setup
    parallel_state = init_parallel_state(
        master_addr=master_addr,
        master_port=master_port,
        dp_size=dp_size,
        sp_size=sp_size,
    )
    device = resolve_device(parallel_state=parallel_state)

    # Get datasets and loaders
    tokenizer = BPTokenizer()
    train_dataset = FASTADataset(fasta_dir=(data_dir / "train"), chunk_size=context_len, tokenizer=tokenizer, parallel_state=parallel_state)
    val_dataset = FASTADataset(fasta_dir=(data_dir / "val"), chunk_size=context_len, tokenizer=tokenizer, parallel_state=parallel_state)
    test_dataset = FASTADataset(fasta_dir=(data_dir / "test"), chunk_size=context_len, tokenizer=tokenizer, parallel_state=parallel_state)
    collator = MLMCollator(tokenizer=tokenizer, parallel_state=parallel_state)
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
        model = MLMTransformer(model_cfg, parallel_state).to(device)
        ## Trainer configuration
        trainer_cfg = TrainerConfig(
            log_every=1,
            eval_every=100,
            eval_batches=10,
            batches_per_step=32,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            checkpoint_dir=ckpt_dir,
            save_every=1000,
            amp_dtype="bfloat16",
            amp_enabled=True,
        )
        trainer = MLMTrainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer._init_optimizer()
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = ckpt_dir / resume_from
        trainer = MLMTrainer.load_checkpoint(ckpt_dir, device, parallel_state=parallel_state)
        if context_len > trainer.model.cfg.max_seq_len:
            trainer.model._update_context_len(new_context_len=context_len)

    # Train the model
    trainer.set_loaders(train_loader, val_loader, test_loader)
    rank0_print(f"Number of Model Parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    trainer.model = DDP(trainer.model, process_group=parallel_state.world_group, device_ids=[parallel_state.local_rank])
    trainer.train(steps=steps)
