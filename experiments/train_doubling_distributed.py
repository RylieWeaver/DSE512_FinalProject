# General
import json
import argparse
from pathlib import Path

# Torch
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# DSE 512
from dse.train import SequenceRegressionTrainerConfig, SequenceRegressionTrainer
from dse.model import TransformerConfig, SequenceRegressionTransformer
from dse.data import DoublingTimeDataset, SequenceRegressionCollator, BPTokenizer
from dse.distributed import resolve_device, rank0_print, init_parallel_state



# Commands:
# - python train_doubling_distributed.py --context_len 80000 --model_dim 1536
# - torchrun --standalone --nproc_per_node=4 train_doubling_distributed.py  --data_parallel_size 1 --sequence_parallel_size 4 --finetune_from /mnt/DGX01/Personal/r9w/Checkpoints/Microbial/frontier
# - nohup torchrun --standalone --nproc_per_node=4 train_doubling_distributed.py  --data_parallel_size 1 --sequence_parallel_size 4 --finetune_from /mnt/DGX01/Personal/r9w/Checkpoints/Microbial/frontier > output.txt 2>&1 &
# - pkill -u "$(whoami)" -f 'train_doubling_distributed.py'
# - NOTE: The model can take hundreds of epochs to converge, even with few parameters and context length.


if __name__ == "__main__":
    # Setup
    ## Read args
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/DGX01/Personal/r9w/Datasets/Microbial/doubling", help="Directory for data")
    parser.add_argument("--ckpt_dir", type=str, default="/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/scratch/doubling", help="Directory for checkpoints")
    parser.add_argument("--context_len", type=int, default=80000, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=1536, help="Model dimension")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of linear warmup steps")
    parser.add_argument("--embed_dropout", type=float, default=0.05, help="Embedding dropout")
    parser.add_argument("--attn_dropout", type=float, default=0.05, help="Attention dropout")
    parser.add_argument("--resid_dropout", type=float, default=0.1, help="Residual dropout")
    parser.add_argument("--head_dropout", type=float, default=0.2, help="Head dropout")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--finetune_from", type=str, default=None, help="Directory to finetune from checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume training from checkpoint")
    parser.add_argument("--data_parallel_size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--sequence_parallel_size", type=int, default=4, help="Sequence parallel size")
    parser.add_argument("--master_addr", type=str, default=None, help="Master address for distributed training")
    parser.add_argument("--master_port", type=str, default=None, help="Master port for distributed training")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()
    context_len = args.context_len
    model_dim = args.model_dim
    epochs = args.epochs
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    embed_dropout = args.embed_dropout
    attn_dropout = args.attn_dropout
    resid_dropout = args.resid_dropout
    head_dropout = args.head_dropout
    weight_decay = args.weight_decay
    finetune_from = args.finetune_from
    resume_from = args.resume_from
    dp_size = args.data_parallel_size
    sp_size = args.sequence_parallel_size
    master_addr = args.master_addr
    master_port = args.master_port
    ## Anything that I will keep constant
    log_every = 1
    eval_every = 1
    save_every = 1
    num_heads = 8
    num_layers = 24
    output_dim = 1
    batches_per_step = 16
    decay_type = "cosine"
    decay_steps = 9 * warmup_steps
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
    train_dataset = DoublingTimeDataset(df_path=(data_dir / "iso_rib_temp_mod_train.csv"), tokenizer=tokenizer, parallel_state=parallel_state)
    val_dataset = DoublingTimeDataset(df_path=(data_dir / "iso_rib_temp_mod_val.csv"), tokenizer=tokenizer, parallel_state=parallel_state)
    test_dataset = DoublingTimeDataset(df_path=(data_dir / "iso_rib_temp_mod_test.csv"), tokenizer=tokenizer, parallel_state=parallel_state)
    collator = SequenceRegressionCollator(tokenizer=tokenizer, max_pad_length=context_len, parallel_state=parallel_state)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=parallel_state.dp_size, rank=parallel_state.dp_rank, shuffle=True, drop_last=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=parallel_state.dp_size, rank=parallel_state.dp_rank, shuffle=False, drop_last=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=parallel_state.dp_size, rank=parallel_state.dp_rank, shuffle=False, drop_last=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=collator, num_workers=4, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collator, num_workers=4, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collator, num_workers=4, sampler=test_sampler)

    # Train from scratch
    if not finetune_from and not resume_from:
        ## Define the model
        model_cfg = TransformerConfig(
            vocab_size=tokenizer.out_vocab_size,
            max_seq_len=context_len,
            dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_flash_attn=True,
            output_dim=output_dim,
            embed_dropout=embed_dropout,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            head_dropout=head_dropout,
        )
        model = SequenceRegressionTransformer(model_cfg, parallel_state=parallel_state).to(device)
        ## Trainer configuration
        trainer_cfg = SequenceRegressionTrainerConfig(
            log_every=log_every,
            eval_every=eval_every,
            batches_per_step=batches_per_step,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            decay_type=decay_type,
            decay_steps=decay_steps,
            weight_decay=weight_decay,
            checkpoint_dir=ckpt_dir,
            save_every=save_every,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled
        )
        trainer = SequenceRegressionTrainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer._init_optimizer()
    # Finetune from checkpoint
    elif finetune_from:
        finetune_from = Path(finetune_from).resolve()
        ## Model
        load_dir = Path(finetune_from).resolve()
        model_dict_path = load_dir / "model_config.json"
        with model_dict_path.open("r") as f:
            model_dict = json.load(f)
        model_cfg = TransformerConfig(**model_dict)
        model_cfg.embed_dropout = embed_dropout
        model_cfg.attn_dropout = attn_dropout
        model_cfg.resid_dropout = resid_dropout
        model_cfg.head_dropout = head_dropout
        model_cfg.output_dim = output_dim
        model = SequenceRegressionTransformer(model_cfg, parallel_state=parallel_state).to(device)
        state_dict = torch.load(load_dir / "model.pt", weights_only=True, map_location=device)
        backbone_state_dict = {
            k.removeprefix("backbone."): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        model.backbone.load_state_dict(backbone_state_dict, strict=True)
        trainer_cfg = SequenceRegressionTrainerConfig(
            log_every=log_every,
            eval_every=eval_every,
            batches_per_step=batches_per_step,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            decay_type=decay_type,
            decay_steps=decay_steps,
            weight_decay=weight_decay,
            checkpoint_dir=ckpt_dir,
            save_every=save_every,
            amp_dtype=amp_dtype,
            amp_enabled=amp_enabled
        )
        trainer = SequenceRegressionTrainer(config=trainer_cfg, model=model, device=device, parallel_state=parallel_state)
        trainer._init_optimizer()
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = ckpt_dir / resume_from
        trainer = SequenceRegressionTrainer.load_checkpoint(ckpt_dir, device, parallel_state=parallel_state)

    # Train the model
    trainer.set_loaders(train_loader, val_loader, test_loader)
    rank0_print(f"Number of Model Parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    trainer.model = DDP(trainer.model, process_group=parallel_state.world_group, device_ids=[parallel_state.local_rank])
    trainer.train(epochs=epochs)
