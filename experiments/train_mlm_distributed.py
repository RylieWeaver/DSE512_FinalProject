# General
import json
import argparse
from pathlib import Path

# Torch
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# DSE 512
from dse.train import MLMTrainerConfig, MLMTrainer
from dse.model import TransformerConfig, MLMTransformer
from dse.data import FASTADataset, MLMCollator, BPTokenizer
from dse.distributed import init_parallel_state, resolve_device, rank0_print



# Commands:
# - torchrun --standalone --nproc_per_node=4 train_mlm_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2
# - nohup torchrun --standalone --nproc_per_node=4 train_mlm_distributed.py  --data_parallel_size 2 --sequence_parallel_size 2 > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'train_mlm_distributed.py'


if __name__ == "__main__":
    # Setup
    ## Passed Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/DGX01/Personal/r9w/Datasets/Microbial/reference", help="Directory for data")
    parser.add_argument("--ckpt_dir", type=str, default="/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/scratch/mlm", help="Directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default=Path(__file__).parent / "log", help="Directory to save logs and checkpoints to")
    parser.add_argument("--chunk_size", type=int, default=2048, help="Chunk size for FASTA dataset (should be <= context length)")
    parser.add_argument("--context_len", type=int, default=2048, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=1536, help="Model dimension")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--end_step", type=int, default=None, help="Step to end training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--batches_per_step", type=int, default=1, help="Number of batches to accumulate gradients for before stepping optimizer")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of linear warmup steps")
    parser.add_argument("--embed_dropout", type=float, default=0.0, help="Embedding dropout")
    parser.add_argument("--attn_dropout", type=float, default=0.0, help="Attention dropout")
    parser.add_argument("--resid_dropout", type=float, default=0.0, help="Residual dropout")
    parser.add_argument("--resume_only_weights_from", type=str, default=None, help="Directory to resume only weights from checkpoint (usually for context length update)")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume training from checkpoint")
    parser.add_argument("--data_parallel_size", type=int, default=2, help="Data parallel size")
    parser.add_argument("--sequence_parallel_size", type=int, default=2, help="Sequence parallel size")
    parser.add_argument("--master_addr", type=str, default=None, help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, default=None, help="Master port for distributed training")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    chunk_size = args.chunk_size
    context_len = args.context_len
    model_dim = args.model_dim
    steps = args.steps
    end_step = args.end_step
    batch_size = args.batch_size
    batches_per_step = args.batches_per_step
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    embed_dropout = args.embed_dropout
    attn_dropout = args.attn_dropout
    resid_dropout = args.resid_dropout
    resume_only_weights_from = args.resume_only_weights_from
    resume_from = args.resume_from
    dp_size = args.data_parallel_size
    sp_size = args.sequence_parallel_size
    master_addr = args.master_addr
    master_port = args.master_port
    ## Anything that I will keep constant
    log_every = 40
    eval_every = 40
    eval_batches = 40
    save_every = 40
    num_heads = 8
    num_layers = 24
    decay_type = "cosine"
    decay_steps = (end_step - warmup_steps) if end_step else (steps - warmup_steps)
    amp_dtype = "bfloat16"
    amp_enabled = True

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
    train_dataset = FASTADataset(fasta_dir=(data_dir / "train"), chunk_size=chunk_size, tokenizer=tokenizer, parallel_state=parallel_state)
    val_dataset = FASTADataset(fasta_dir=(data_dir / "val"), chunk_size=chunk_size, tokenizer=tokenizer, parallel_state=parallel_state)
    test_dataset = FASTADataset(fasta_dir=(data_dir / "test"), chunk_size=chunk_size, tokenizer=tokenizer, parallel_state=parallel_state)
    collator = MLMCollator(tokenizer=tokenizer, parallel_state=parallel_state)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, collate_fn=collator, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator, num_workers=4)

    # Train from scratch if no resume step is provided
    if not resume_only_weights_from and not resume_from:
        ## Define the model
        model_cfg = TransformerConfig(
            vocab_size=tokenizer.out_vocab_size,
            max_seq_len=context_len,
            dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_flash_attn=True,
            embed_dropout=embed_dropout,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )
        model = MLMTransformer(model_cfg, parallel_state).to(device)
        ## Trainer configuration
        trainer_cfg = MLMTrainerConfig(
            log_every=log_every,
            eval_every=eval_every,
            eval_batches=eval_batches,
            batches_per_step=batches_per_step,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            decay_type=decay_type,
            decay_steps=decay_steps,
            checkpoint_dir=ckpt_dir,
            save_every=save_every,
            log_dir=log_dir,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        trainer = MLMTrainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer._init_optimizer()
    # Elif resuming only weights (usually for context-length update)
    elif resume_only_weights_from:
        resume_only_weights_from = Path(resume_only_weights_from).resolve()
        ## Model
        load_dir = Path(resume_only_weights_from).resolve()
        model_dict_path = load_dir / "model_config.json"
        with model_dict_path.open("r") as f:
            model_dict = json.load(f)
        model_cfg = TransformerConfig(**model_dict)
        model_cfg.embed_dropout = embed_dropout
        model_cfg.attn_dropout = attn_dropout
        model_cfg.resid_dropout = resid_dropout
        model = MLMTransformer(model_cfg, parallel_state).to(device)
        state_dict = torch.load(load_dir / "model.pt", weights_only=True, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        ## Update context length if needed here!
        if context_len > model.cfg.max_seq_len:
            rank0_print(f"Updating model context length from {model.cfg.max_seq_len} to {context_len}")
            model._update_context_len(new_context_len=context_len)
        trainer_cfg = MLMTrainerConfig(
            log_every=log_every,
            eval_every=eval_every,
            eval_batches=eval_batches,
            batches_per_step=batches_per_step,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            decay_type=decay_type,
            decay_steps=decay_steps,
            checkpoint_dir=ckpt_dir,
            save_every=save_every,
            log_dir=log_dir,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled,
        )
        trainer = MLMTrainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer._init_optimizer()
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = ckpt_dir / resume_from
        trainer = MLMTrainer.load_checkpoint(ckpt_dir, device, parallel_state=parallel_state)
        # Update logging args in case they were changed
        trainer.cfg.save_every = save_every
        trainer.save_every = save_every
        trainer.cfg.log_every = log_every
        trainer.log_every = log_every
        trainer.cfg.eval_every = eval_every
        trainer.eval_every = eval_every
        trainer.cfg.eval_batches = eval_batches
        trainer.eval_batches = eval_batches

    # Train the model
    trainer.set_loaders(train_loader, val_loader, test_loader)
    rank0_print(f"Number of Model Parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    trainer.model = DDP(trainer.model, process_group=parallel_state.world_group, device_ids=[parallel_state.local_rank])
    trainer.train(steps=steps, end_step=end_step)
