# General

# Torch
import torch
from torch.optim.lr_scheduler import LinearLR, LambdaLR

# DSE 512
from dse.distributed import unwrap_model



def init_optimizer_and_scheduler(model, learning_rate, warmup_steps=0, weight_decay=1e-4):
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
    if warmup_steps > 0:
        scheduler = LinearLR(
            optimizer,
            start_factor=(1 / warmup_steps),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
    else:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    return optimizer, scheduler
