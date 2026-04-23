# General
import math

# Torch
import torch
from torch.optim.lr_scheduler import LambdaLR

# DSE 512
from dse.distributed import unwrap_model


def init_optimizer_and_scheduler(
    model,
    learning_rate,
    warmup_steps=0,
    decay_steps=None,
    decay_type="none",   # "none", "linear", "cosine"
    min_lr_scale=0.1,
    weight_decay=1e-4,
):
    model = unwrap_model(model)

    no_wd = set()
    if hasattr(model, "no_wd_params") and callable(model.no_wd_params):
        no_wd = set(model.no_wd_params())
    elif hasattr(model, "_no_wd_params") and callable(model._no_wd_params):
        no_wd = set(model._no_wd_params())

    if no_wd:
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if name in no_wd else decay).append(param)
        param_groups = []
        if decay:
            param_groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            param_groups.append({"params": no_decay, "weight_decay": 0.0})
    else:
        param_groups = model.parameters()

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)

    def lr_lambda(step: int):
        # Warmup
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        # No decay
        if decay_type == "none" or decay_steps is None or decay_steps <= 0:
            return 1.0

        # Progress through decay phase
        decay_step = min(max(step - warmup_steps, 0), decay_steps)
        progress = decay_step / decay_steps

        if decay_type == "linear":
            return min_lr_scale + (1.0 - min_lr_scale) * (1.0 - progress)

        if decay_type == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_scale + (1.0 - min_lr_scale) * cosine

        raise ValueError(f"Unknown decay_type: {decay_type}")

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler
