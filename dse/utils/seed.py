# General
import random
import numpy as np

# Torch
import torch

# DSE 512



def set_all_random_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
