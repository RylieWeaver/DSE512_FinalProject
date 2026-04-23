# General

# Torch
import torch
from torch.nn import functional as F

# DSE 512



def move_to(data, device):
    """
    This function is a recursive helper to move data to a given device.
    
    Mostly this just helps move data dicts, which are helpful to keep 
    track of which tensors represent what inputs.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to(v, device) for v in data)
    else:
        return data


def pad(x_list: list[torch.Tensor], pad_val, pad_length):
    x_padded = []
    pad_mask = []
    for x in x_list:
        L = x.size(0)
        # Pad
        if L < pad_length:
            x_padded.append(
                F.pad(x, (0, pad_length - L), value=pad_val)
            )
            pad_mask.append(torch.cat([torch.zeros(L, dtype=torch.long), torch.ones(pad_length - L, dtype=torch.long)]))
        # Truncate
        elif L > pad_length:
            x_padded.append(x[:pad_length])
            pad_mask.append(torch.zeros(pad_length, dtype=torch.long))
        # No change
        else:
            x_padded.append(x)
            pad_mask.append(torch.zeros(L, dtype=torch.long))
    
    x_padded = torch.stack(x_padded, dim=0)
    pad_mask = torch.stack(pad_mask, dim=0)
    pad_mask.requires_grad = False
    return x_padded, pad_mask