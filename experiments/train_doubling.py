# General
import json
import argparse
from pathlib import Path

# Torch
import torch

# DSE 512
from dse.train import SequenceRegressionTrainerConfig, SequenceRegressionTrainer
from dse.model import TransformerConfig, SequenceRegressionTransformer
from dse.data import DoublingTimeDataset, SequenceRegressionCollator, BPTokenizer
from dse.distributed import resolve_device, rank0_print



# Commands:
# - python train_regression.py --context_len 2048 --model_dim 256
# - finetune_from = Path("/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/long_train/mlm/step_300000").resolve()
# - NOTE: The model can take hundreds of epochs to converge, even with few parameters and context length.


if __name__ == "__main__":
    # Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_len", type=int, default=2048, help="Context length for model")
    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of linear warmup steps")
    parser.add_argument("--finetune_from", type=str, default=None, help="Directory to finetune from checkpoint")
    parser.add_argument("--resume_from", type=str, default=None, help="Directory to resume training from checkpoint")
    args = parser.parse_args()
    context_len = args.context_len
    model_dim = args.model_dim
    epochs = args.epochs
    learning_rate = args.learning_rate
    warmup_steps = args.warmup_steps
    finetune_from = args.finetune_from
    resume_from = args.resume_from
    ckpt_dir = Path("/mnt/DGX01/Personal/r9w/Checkpoints/Microbial/scratch/doubling").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = resolve_device()

    # Get datasets and loaders
    data_dir = Path("/mnt/DGX01/Personal/r9w/Datasets/Microbial/doubling")
    tokenizer = BPTokenizer()
    train_dataset = DoublingTimeDataset(df_path=(data_dir / "iso_rib_temp_mod_train.csv"), tokenizer=tokenizer)
    val_dataset = DoublingTimeDataset(df_path=(data_dir / "iso_rib_temp_mod_val.csv"), tokenizer=tokenizer)
    test_dataset = DoublingTimeDataset(df_path=(data_dir / "iso_rib_temp_mod_test.csv"), tokenizer=tokenizer)
    collator = SequenceRegressionCollator(tokenizer=tokenizer, max_pad_length=context_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=collator, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=collator)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collator)    

    # Train from scratch 
    if not finetune_from and not resume_from:
        ## Define the model
        model_cfg = TransformerConfig(
            vocab_size=tokenizer.out_vocab_size,
            max_seq_len=context_len,
            dim=model_dim,
            num_heads=8,
            num_layers=24,
            use_flash_attn=True,
            output_dim=1,
            embed_dropout=0.05,
            attn_dropout=0.05,
            resid_dropout=0.1,
            head_dropout=0.2,
        )
        model = SequenceRegressionTransformer(model_cfg).to(device)
        ## Trainer configuration
        trainer_cfg = SequenceRegressionTrainerConfig(
            log_every=1,
            eval_every=10,
            batches_per_step=16,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=1e-2,
            checkpoint_dir=ckpt_dir,
            save_every=100,
            amp_dtype="bfloat16",
            amp_enabled=True
        )
        trainer = SequenceRegressionTrainer(config=trainer_cfg, model=model, device=device)
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
        model_cfg.embed_dropout = 0.05
        model_cfg.attn_dropout = 0.05
        model_cfg.resid_dropout = 0.1
        model_cfg.head_dropout = 0.2
        model_cfg.output_dim = 1
        model = SequenceRegressionTransformer(model_cfg).to(device)
        state_dict = torch.load(load_dir / "model.pt", weights_only=True, map_location=device)
        backbone_state_dict = {
            k.removeprefix("backbone."): v
            for k, v in state_dict.items()
            if k.startswith("backbone.")
        }
        model.backbone.load_state_dict(backbone_state_dict, strict=True)
        trainer_cfg = SequenceRegressionTrainerConfig(
            log_every=1,
            eval_every=1,
            batches_per_step=16,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=1e-2,
            checkpoint_dir=ckpt_dir,
            save_every=100,
            amp_dtype="bfloat16",
            amp_enabled=True
        )
        trainer = SequenceRegressionTrainer(config=trainer_cfg, model=model, device=device)
        trainer._init_optimizer()
    # Otherwise, train from checkpoint
    else:
        ckpt_dir = ckpt_dir / resume_from
        trainer = SequenceRegressionTrainer.load_checkpoint(ckpt_dir, device)

    # Train the model
    trainer.set_loaders(train_loader, val_loader, test_loader)
    rank0_print(f"Number of Model Parameters: {sum(p.numel() for p in trainer.model.parameters())}")
    trainer.train(epochs=epochs)
