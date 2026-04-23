# General
from typing import Union

# Torch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# DSE 512



def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_rank0(group: Union[dist.ProcessGroup, None] = None) -> bool:
    if not is_dist():
        return True
    if group is None:
        return dist.get_rank() == 0
    return dist.get_rank(group) == 0

def rank0_print(*args, **kwargs):
    if is_rank0():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

def rank0_write(path, message: str, mode: str = "a"):
    if is_rank0():
        with path.open(mode) as f:
            f.write(f"\n{message}")

def unwrap_model(model):
    if is_dist() and isinstance(model, DDP):
        return model.module
    return model

@torch.no_grad()
def reduce_scalar(x, device: torch.device, group: dist.ProcessGroup, dtype=torch.float32, average=True) -> float:
    if not is_dist():
        return float(x)
    
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=dtype, device=device).detach()
    
    dist.barrier()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    if average:
        x /= dist.get_world_size(group)
    return float(x.item())

def broadcast_tensor(x, group, device, src=None, dtype=torch.float32):
    # Default to broadcast from rank 0 of the group, unless specified
    if src is None:
        src = dist.get_global_rank(group, 0)
    is_src = dist.get_rank() == src

    # 1. Broadcast ndims
    ndim = torch.empty(1, dtype=torch.long, device=device)
    if is_src:
        ndim[0] = x.ndim
    dist.broadcast(ndim, src=src, group=group)
    ndim = int(ndim.item())

    # 2. Broadcast shape 
    shape = torch.empty(ndim, dtype=torch.long, device=device)
    if is_src:
        shape[:] = torch.tensor(x.shape, dtype=torch.long, device=device)
    dist.broadcast(shape, src=src, group=group)
    shape = tuple(shape.tolist())

    # 3. Broadcast tensor
    if not is_src:
        x = torch.empty(shape, dtype=dtype, device=device)
    dist.broadcast(x, src=src, group=group)
    return x
