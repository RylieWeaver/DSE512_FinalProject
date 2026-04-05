# General
import random

# Torch
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

# DSE 512
from dse.distributed import ParallelState, rank0_write, is_rank0



def create_random_dna_string(path, n_bases=1_000_000, seed=42):
    """
    This file should easily be small enough to load even with 1M bases.

    Make sure that n_bases is larger than the chunk_size you planso that you
    can sample large chunks without worrying about padding! (explained below)
    It's not the focus of this repo to deal with padding, so we keep it simple.
    """
    # Create
    rng = random.Random(seed)
    bases = ["A", "C", "G", "T"]
    seq = "".join(rng.choices(bases, k=n_bases))
    # Save/Return
    rank0_write(path, seq, mode="w")
    return seq


def bert_mlm_mask(input_ids, mask_token_id, dna_token_ids):
    """
    Apply standard BERT MLM corruption to a DNA token tensor.

    Hard-coded probabilities:
    - 15% of maskable tokens are selected for prediction
    - of selected tokens: 80% -> [MASK], 10% -> random DNA base, 10% -> keep
    """
    # Setup
    input_ids = input_ids.clone()
    labels = torch.full_like(input_ids, -100)
    dna_token_ids = torch.as_tensor(dna_token_ids, dtype=input_ids.dtype, device=input_ids.device)
    maskable = torch.isin(input_ids, dna_token_ids)
    if not maskable.any():
        return input_ids, labels

    # Select MLM positions and ensure at least one supervised token
    selected = (torch.rand(input_ids.shape, device=input_ids.device) < 0.15) & maskable
    if not selected.any():
        flat_selected = selected.reshape(-1)
        flat_maskable = maskable.reshape(-1)
        maskable_idx = torch.nonzero(flat_maskable, as_tuple=False).flatten()
        forced_idx = maskable_idx[torch.randint(maskable_idx.numel(), (1,), device=input_ids.device)]
        flat_selected[forced_idx] = True
        selected = flat_selected.view_as(selected)

    # Supervise selected original tokens
    labels[selected] = input_ids[selected]

    # 80% [MASK], 10% random DNA base, 10% unchanged
    corruption = torch.rand(input_ids.shape, device=input_ids.device)
    to_mask = selected & (corruption < 0.8)
    to_rand = selected & (corruption >= 0.8) & (corruption < 0.9)

    input_ids[to_mask] = mask_token_id
    if to_rand.any():
        rand_choices = dna_token_ids[
            torch.randint(dna_token_ids.numel(), input_ids.shape, device=input_ids.device)
        ]
        input_ids[to_rand] = rand_choices[to_rand]
    return input_ids, labels


class DNADataset(IterableDataset):
    def __init__(self, path, chunk_size=8192, seed=42, parallel_state=None):
        with open(path, "r", encoding="utf-8") as handle:
            self.dna_string = handle.read().strip().upper()
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4, "[MASK]": 5}
        self.dna_token_ids = tuple(self.dna_vocab[bp] for bp in ("A", "C", "G", "T"))
        self.mask_token_id = self.dna_vocab["[MASK]"]
        self.vocab_size = len(self.dna_vocab)
        self.chunk_size = chunk_size
        self.seed = seed
        self.parallel_state = parallel_state if parallel_state else ParallelState()
    
    def _encode(self, sequence):
        return [self.dna_vocab.get(ch, self.dna_vocab["N"]) for ch in sequence]

    def __iter__(self):
        """
        Note that the parametrization of start_idx having a maximum of len(seq) - chunk_size
        ensures that we always have enough sequence left to return a full chunk (i.e. we don't 
        need to worry about padding). The model will need to account for padding if future data
        includes padding. The sneaky error that could arise is if your chunk size is larger than
        your sequence length, but the default dataset sequence length is 100M which should be 
        big enough.
        """
        worker = torch.utils.data.get_worker_info()
        base_seed = self.seed + (worker.id if worker is not None else 0)
        rng = random.Random(base_seed)
        while True:
            if (self.parallel_state.sp_size == 1 or is_rank0(self.parallel_state.sp_group)):
                if self.chunk_size > len(self.dna_string):
                    raise ValueError(f"Chunk size {self.chunk_size} must be smaller than dataset sequence length {len(self.dna_string)}")
                start_idx = rng.randint(0, len(self.dna_string) - self.chunk_size)
                chunk = self.dna_string[start_idx : start_idx + self.chunk_size]
                chunk = torch.tensor(self._encode(chunk), dtype=torch.long)  # Str -> Tensor[int]
                token_ids, labels = bert_mlm_mask(
                    chunk,
                    mask_token_id=self.mask_token_id,
                    dna_token_ids=self.dna_token_ids,
                )
                yield token_ids, labels
            else:
                token_ids = torch.empty(self.chunk_size, dtype=torch.long)
                labels = torch.full((self.chunk_size,), -100, dtype=torch.long)
                yield token_ids, labels
