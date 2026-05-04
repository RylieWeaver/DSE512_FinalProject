# General
import math

# Torch
import torch
from torch.optim.lr_scheduler import LambdaLR

# DSE 512
from dse.distributed import unwrap_model


def init_optimizer_and_scheduler(
    model,
    learning_rate=None,
    warmup_steps=0,
    decay_steps=None,
    decay_type="none",   # "none", "linear", "cosine"
    min_lr_scale=0.1,
    weight_decay=1e-4,
    param_groups=None,
):
    model = unwrap_model(model)

    no_wd = set()
    if hasattr(model, "no_wd_params") and callable(model.no_wd_params):
        no_wd = set(model.no_wd_params())
    elif hasattr(model, "_no_wd_params") and callable(model._no_wd_params):
        no_wd = set(model._no_wd_params())

    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    param_name_by_id = {id(param): name for name, param in named_params}

    if param_groups is not None:
        optimizer_param_groups = []
        for group in param_groups:
            group_params = list(group["params"])
            decay, no_decay = [], []
            for param in group_params:
                name = param_name_by_id.get(id(param))
                if name in no_wd:
                    no_decay.append(param)
                else:
                    decay.append(param)

            decay_group = {k: v for k, v in group.items() if k != "params"}
            no_decay_group = {k: v for k, v in group.items() if k != "params"}
            no_decay_group["weight_decay"] = 0.0
            if decay:
                optimizer_param_groups.append({"params": decay, **decay_group})
            if no_decay:
                optimizer_param_groups.append({"params": no_decay, **no_decay_group})
    elif no_wd:
        decay, no_decay = [], []
        for name, param in named_params:
            (no_decay if name in no_wd else decay).append(param)
        optimizer_param_groups = []
        if decay:
            optimizer_param_groups.append({"params": decay, "weight_decay": weight_decay})
        if no_decay:
            optimizer_param_groups.append({"params": no_decay, "weight_decay": 0.0})
    else:
        optimizer_param_groups = [param for _, param in named_params]

    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=learning_rate, weight_decay=weight_decay)

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
