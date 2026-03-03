# General
from typing import Union, Optional

# Torch
import torch

# DSE 512
from .groups import ParallelState


def resolve_device(device: Optional[Union[torch.device, str]] = None, parallel_state: Optional[ParallelState] = None) -> torch.device:
    # NOTE: Only allow CPU training if not doing any type of parallelism
    if isinstance(device, torch.device):
        return device
    elif isinstance(device, str):
        return torch.device(device)
    elif parallel_state is not None and parallel_state.device is not None:
        if isinstance(parallel_state.device, torch.device):
            return parallel_state.device
        elif isinstance(parallel_state.device, str):
            return torch.device(parallel_state.device)
    elif torch.cuda.is_available():
        if parallel_state is not None:
            local_rank = parallel_state.local_rank
            return torch.device(f"cuda:{local_rank}")
        else:
            return torch.device("cuda")
    else:
        return torch.device("cpu")
