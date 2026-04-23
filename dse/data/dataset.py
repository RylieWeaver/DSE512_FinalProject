# General
import random
import warnings
from pathlib import Path
import pysam
from typing import Union
import pandas as pd

# Torch
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

# DSE 512
from dse.distributed import ParallelState, rank0_print, rank0_write, is_rank0
from .tokenizer import BPTokenizer
from .utils import move_to, pad


def bert_mlm_mask(input_ids, tokenizer=None, prob=0.15, mask_prob=0.0, rand_prob=0.0, keep_prob=0.0):
    """
    A lot of logic here is made to make sure we don't mask/predict 'N' tokens since they aren't informative.
    """
    # Setup
    tokenizer = tokenizer if tokenizer is not None else BPTokenizer()
    device = input_ids.device
    maskable_ints = torch.tensor(                                               # only mask [A,T,C,G]
        [tokenizer.tok2id[bp] for bp in tokenizer.target_bp], device=device
    )
    mask_int = tokenizer.tok2id[tokenizer.MASK]
    labels = torch.full_like(input_ids, -100, device=device)                    # labels of -100 are ignored within PyTorch CE loss

    # Default probs
    if mask_prob == 0.0 and rand_prob == 0.0 and keep_prob == 0.0:
        mask_prob = 0.8
        rand_prob = 0.1
        keep_prob = 0.1

    # Normalize probs
    tot = mask_prob + rand_prob + keep_prob
    mask_prob /= tot
    rand_prob /= tot
    keep_prob /= tot

    # Get mask
    ## Generate random values
    rand = torch.rand_like(input_ids.float())                           # [B, L]
    ## Skip any non-valid bps (ATCG)
    maskable = torch.isin(input_ids, maskable_ints).bool()              # [B, L]
    ## Ensure at least one masked token
    ## NOTE: no masks in a batch --> NaN loss
    mask = rand.masked_fill(~maskable, float('inf')).argmin(dim=-1)     # [B]: choose the smallest maskable column (nonmaskable are inf so never chosen)
    mask = F.one_hot(mask, num_classes=input_ids.size(1)).bool()        # [B, L]
    ## Random choice with mask prob
    selected = (rand < prob)                                            # [B, L]
    # Assign labels
    idx = (selected & maskable) | mask                                  # [B, L]
    labels[idx] = input_ids[idx]

    # Apply edits to the selected idxs (mask token, random base, keep)
    r2 = torch.rand_like(input_ids.float())
    t0, t1 = mask_prob, mask_prob + rand_prob                           # thresholds
    to_mask = idx & (r2 < t0)                                           # [B, L]
    to_rand = idx & (r2 >= t0) & (r2 < t1)                              # [B, L]
    # keep = sel & (v >= t1)                                            # [B, L]: this is a no-op so commented out 

    # Slice
    if to_mask.any():
        input_ids[to_mask] = mask_int
    if to_rand.any():
        options = torch.tensor(tokenizer.encode(tokenizer.target_bp), device=device)
        rand_choices = options[torch.randint(0, len(options), input_ids.shape, device=device)]
        input_ids[to_rand] = rand_choices[to_rand]
    return input_ids, labels


def round_pad_length(num, multiple, name):
    rounded = ((num + multiple - 1) // multiple) * multiple
    if rounded != num:
        warnings.warn(
            f"Rounding {name} from {num} to {rounded} so it is divisible by pad multiple {multiple}."
        )
    return rounded


####################################################################
##### Dummy DNA Dataset (leftover from original examples code) #####
####################################################################
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

class DNADataset(IterableDataset):
    def __init__(self, path, chunk_size=8192, base_seed=42, parallel_state=None):
        with open(path, "r", encoding="utf-8") as handle:
            self.dna_string = handle.read().strip().upper()
        self.tokenizer = BPTokenizer()
        self.chunk_size = chunk_size
        self.base_seed = base_seed
        self.parallel_state = parallel_state if parallel_state else ParallelState()
    
    def _encode(self, sequence):
        return [self.tokenizer.tok2id.get(ch, self.tokenizer.tok2id["N"]) for ch in sequence]

    def __iter__(self):
        """
        Note that the parametrization of start_idx having a maximum of len(seq) - chunk_size
        ensures that we always have enough sequence left to return a full chunk (i.e. we don't 
        need to worry about padding). The model will need to account for padding if future data
        includes padding. The sneaky error that could arise is if your chunk size is larger than
        your sequence length, but the default dataset sequence length is 100M which should be 
        big enough.
        """
        # Setup on worker
        # NOTE: Ensure different chunks on different ranks/workers by seeding differently
        worker = torch.utils.data.get_worker_info()
        rank_inc = self.parallel_state.rank if self.parallel_state is not None else 0
        worker_inc = worker.id if worker is not None else 0
        total_seed = self.base_seed + rank_inc + worker_inc
        rng = random.Random(total_seed)
        while True:
            if (self.parallel_state.sp_size == 1 or is_rank0(self.parallel_state.sp_group)):
                if self.chunk_size > len(self.dna_string):
                    raise ValueError(f"Chunk size {self.chunk_size} must be smaller than dataset sequence length {len(self.dna_string)}")
                start_idx = rng.randint(0, len(self.dna_string) - self.chunk_size)
                chunk = self.dna_string[start_idx : start_idx + self.chunk_size]
                token_ids = torch.tensor(self.tokenizer.encode(chunk), dtype=torch.long)  # Str -> Tensor[token_id]
                yield {"token_ids": token_ids}
            else:
                yield {"token_ids": None}


#################################################################
##### FASTA Dataset (must be at least 1 FASTA file to work) #####
#################################################################
def mlm_collate_batch(batch, tokenizer, min_pad_length, max_pad_length=None):
    # Setup
    pad_token_id = tokenizer.tok2id[tokenizer.PAD]
    B = len(batch)
    max_L = max(item["token_ids"].size(0) for item in batch)
    pad_length = max(max_L, min_pad_length)
    pad_length = min(pad_length, max_pad_length) if max_pad_length is not None else pad_length

    # Initialize objects that are present in every batch
    token_ids_batch = []

    # Unpack the batch inputs
    for item in batch:
        token_ids_batch.append(item["token_ids"])

    # Pad / Stack
    # NOTE: Pad mask unused for now
    token_ids_batch, pad_mask = pad(token_ids_batch, pad_val=pad_token_id, pad_length=pad_length)       # [B, max_L]

    # Mask
    input_ids, labels = bert_mlm_mask(
        token_ids_batch,
        tokenizer=tokenizer,
        prob=0.15,
        mask_prob=0.8,
        rand_prob=0.1,
        keep_prob=0.1,
    )
    return input_ids, labels


class MLMCollator:
    def __init__(self, tokenizer=None, min_pad_multiple=None, min_pad_length=2, max_pad_length=None, parallel_state=None):
        self.tokenizer = tokenizer if tokenizer is not None else BPTokenizer()
        if min_pad_multiple is not None:
            self.min_pad_multiple = min_pad_multiple
        elif parallel_state is not None:
            self.min_pad_multiple = parallel_state.sp_size
        else:
            self.min_pad_multiple = 1
        min_pad_length = max(min_pad_length, self.min_pad_multiple)
        self.min_pad_length = round_pad_length(min_pad_length, self.min_pad_multiple, "min_pad_length")
        self.max_pad_length = (
            round_pad_length(max_pad_length, self.min_pad_multiple, "max_pad_length")
            if max_pad_length is not None
            else None
        )
        self.parallel_state = parallel_state if parallel_state else ParallelState()
    def __call__(self, batch):
        if self.parallel_state.sp_size == 1 or is_rank0(self.parallel_state.sp_group):
            max_L = max(len(item["token_ids"]) for item in batch)
            min_pad_length = max(self.min_pad_length, max_L)
            min_pad_length = ((min_pad_length + self.min_pad_multiple - 1) // self.min_pad_multiple) * self.min_pad_multiple
            input_ids, labels = mlm_collate_batch(batch, self.tokenizer, min_pad_length=min_pad_length, max_pad_length=self.max_pad_length)
        else:
            input_ids = None
            labels = None
        return input_ids, labels


class FASTADataset(IterableDataset):
    def __init__(
        self, 
        fasta_dir,
        chunk_size: int,
        min_bp_proportion: float = 0.1,
        min_bps: int = 2,
        base_seed: int = 42,
        tokenizer=None,
        parallel_state=None,
    ):
        super().__init__()
        # Check
        if min_bps < 2:
            rank0_print("Setting min_bps < 2 in DataLoader does not make sense for Masked Language Modeling. Setting to 2.")
            min_bps = 2
        # Read
        self.fasta_dir = fasta_dir
        self.chunk_size = chunk_size
        self.min_bp_proportion = min_bp_proportion
        self.max_n_proportion = 1 - min_bp_proportion  # max proportion of N's allowed
        self.min_bps = min_bps
        self.base_seed = base_seed  # will be incremented by rank/worker to ensure different sampling
        self.tokenizer = tokenizer if tokenizer is not None else BPTokenizer()
        self.parallel_state = parallel_state if parallel_state else ParallelState()

        # Precompute objects for sampling
        self.file_dict = self.build_fasta_dict()

    def _check_sequence(self, seq):
        # Filter out sequences with too many 'N's
        n_count = seq.count('N')
        if n_count / len(seq) > self.max_n_proportion or  (len(seq) - n_count) < self.min_bps:
            return False
        return True
    
    def _filter_sequences(self, seqs):
        return [seq for seq in seqs if self._check_sequence(seq)]

    def _build_fasta_dict(self, fasta_path):
        file = pysam.FastaFile(fasta_path)
        seq_ids = list(file.references)
        filtered_seq_ids = [sid for sid in seq_ids if self._check_sequence(str(file.fetch(sid)))]
        if len(filtered_seq_ids) < len(seq_ids):
            rank0_print(f"Filtered out {len(seq_ids) - len(filtered_seq_ids)} sequences in file {fasta_path} \
                        due to high 'N' content or short length.")
        lengths_dict = {sid: file.get_reference_length(sid) for sid in filtered_seq_ids}
        file.close()

        total_length = sum(lengths_dict.values())
        probs_dict = {sid: length / total_length for sid, length in lengths_dict.items()}
        return lengths_dict, probs_dict
    
    def build_fasta_dict(self):
        # Build a master dictionary to help us sample files/sequences
        file_dict = {}

        # Uniform file sampling probability
        file_names = sorted(self.fasta_dir.rglob("*.fa"))
        file_prob = 1 / len(file_names)

        # Iterate through all files
        for fasta_path in file_names:
            file_name = fasta_path.stem
            lengths_dict, probs_dict = self._build_fasta_dict(fasta_path)
            file_dict[file_name] = {
                "prob": file_prob,
                "seqs": {
                    sid: {
                        "length": lengths_dict[sid],
                        "prob": probs_dict[sid],
                    }
                    for sid in lengths_dict
                },
            }
        return file_dict
    
    def sample_file(self):
        file_names = list(self.file_dict.keys())
        file_probs = [self.file_dict[fname]["prob"] for fname in file_names]
        chosen_file = self.random.choices(file_names, weights=file_probs, k=1)[0]
        return chosen_file
    
    def sample_sequence(self, file_name):
        seq_ids = list(self.file_dict[file_name]["seqs"].keys())
        seq_probs = [self.file_dict[file_name]["seqs"][sid]["prob"] for sid in seq_ids]
        chosen_seq = self.random.choices(seq_ids, weights=seq_probs, k=1)[0]
        return chosen_seq

    def _check_idx(self, start_idx, end_idx):
        if end_idx - start_idx < self.min_bps:
            return False
        return True

    def sample_idx(self, file_name, seq_id):
        length = self.file_dict[file_name]["seqs"][seq_id]["length"]
        start_idx = self.random.randint(0, length - self.min_bps)
        end_idx = min(start_idx + self.chunk_size, length)
        return start_idx, end_idx
    
    def sample_chunk(self):
        file_name = self.sample_file()
        seq_id = self.sample_sequence(file_name)

        count_tries = 0
        max_tries = 10
        file = pysam.FastaFile(self.fasta_dir / f"{file_name}.fa")

        while True:
            start_idx, end_idx = self.sample_idx(file_name, seq_id)
            chunk = str(file.fetch(seq_id, start_idx, end_idx))
            if self._check_idx(start_idx, end_idx) and self._check_sequence(chunk):
                break
            count_tries += 1
            if count_tries > max_tries:
                warnings.warn(f"Had trouble sampling a valid chunk from sequence {seq_id} \
                              in file {file_name} after {count_tries} tries.")
        file.close()
        return chunk

    def _init_random_state(self):
        # NOTE: Ensure different chunks on different ranks/workers by seeding differently
        worker = torch.utils.data.get_worker_info()
        rank_inc = self.parallel_state.rank if self.parallel_state is not None else 0
        worker_inc = worker.id if worker is not None else 0
        total_seed = self.base_seed + rank_inc + worker_inc
        self.random = random.Random(total_seed)

    def __iter__(self):
        # Setup on worker
        self._init_random_state()

        # Infinite yielding
        while True:
            # Only one of SP-group needs to yield sequence
            if (self.parallel_state.sp_size == 1 or is_rank0(self.parallel_state.sp_group)):
                # Get a random sequence chunk
                chunk = self.sample_chunk()
                # Encode
                token_ids = torch.tensor(self.tokenizer.encode(chunk), dtype=torch.long)  # Str -> Tensor[token_id]
                yield {"token_ids": token_ids}
            else:
                # Dummy chunk
                yield {"token_ids": None}


#################################################################
##################### Doubling Time Dataset #####################
#################################################################
def sequence_regression_collate_batch(batch, tokenizer, min_pad_length, max_pad_length=None):
    # Setup
    pad_token_id = tokenizer.tok2id[tokenizer.PAD]
    B = len(batch)
    max_L = max(item["token_ids"].size(0) for item in batch)
    pad_length = max(max_L, min_pad_length)
    pad_length = min(pad_length, max_pad_length) if max_pad_length is not None else pad_length

    # Initialize objects that are present in every batch
    token_ids_batch, temps_batch, labels_batch = [], [], []

    # Unpack the batch inputs
    for item in batch:
        token_ids_batch.append(item["token_ids"])
        temps_batch.append(item["temperatures"])
        labels_batch.append(item["labels"])

    # Pad / Stack
    # NOTE: Pad mask unused for now
    token_ids_batch, pad_mask = pad(token_ids_batch, pad_val=pad_token_id, pad_length=pad_length)       # [B, max_L]
    labels_batch = torch.stack(labels_batch, dim=0)                                                     # [B, output_dim]
    inputs = {"token_ids": token_ids_batch, "temperatures": torch.tensor(temps_batch, dtype=torch.float)}
    return inputs, labels_batch


class SequenceRegressionCollator:
    def __init__(self, tokenizer=None, min_pad_multiple=None, min_pad_length=2, max_pad_length=None, parallel_state=None):
        self.tokenizer = tokenizer if tokenizer is not None else BPTokenizer()
        if min_pad_multiple is not None:
            self.min_pad_multiple = min_pad_multiple
        elif parallel_state is not None:
            self.min_pad_multiple = parallel_state.sp_size
        else:
            self.min_pad_multiple = 1
        min_pad_length = max(min_pad_length, self.min_pad_multiple)
        self.min_pad_length = round_pad_length(min_pad_length, self.min_pad_multiple, "min_pad_length")
        self.max_pad_length = (
            round_pad_length(max_pad_length, self.min_pad_multiple, "max_pad_length")
            if max_pad_length is not None
            else None
        )
        self.parallel_state = parallel_state if parallel_state else ParallelState()
    def __call__(self, batch):
        if self.parallel_state.sp_size == 1 or is_rank0(self.parallel_state.sp_group):
            max_L = max(len(item["token_ids"]) for item in batch)
            min_pad_length = max(self.min_pad_length, max_L)
            min_pad_length = ((min_pad_length + self.min_pad_multiple - 1) // self.min_pad_multiple) * self.min_pad_multiple
            input_ids, labels = sequence_regression_collate_batch(batch, self.tokenizer, min_pad_length=min_pad_length, max_pad_length=self.max_pad_length)
        else:
            input_ids = None
            labels = None
        return input_ids, labels


class DoublingTimeDataset(Dataset):
    """
    Expects to read from a TSV/CSV file.
    """
    def __init__(
        self,
        df_path: Union[Path, str],
        sequence_col: str = "sequence",
        temperature_col: str = "growth_tmp",
        target_cols: list[str] | str | None = None,
        base_seed: int = 42,
        tokenizer=None,
        drop_missing: bool = True,
        parallel_state=None,
    ):
        super().__init__()
        self.df_path = Path(df_path)
        self.sequence_col = sequence_col
        self.temperature_col = temperature_col
        if target_cols is None:
            target_cols = ["log_dob_h"]
        self.target_cols = target_cols if isinstance(target_cols, list) else [target_cols]
        self.base_seed = base_seed
        self.tokenizer = tokenizer if tokenizer is not None else BPTokenizer()
        self.drop_missing = drop_missing
        self.parallel_state = parallel_state if parallel_state else ParallelState()
        self.random = None
        self.random_seed = None
        self.df = self._read_df(drop_missing=drop_missing)

    def _read_df(self, drop_missing: bool):
        # Read file
        if not self.df_path.exists():
            raise FileNotFoundError(f"Could not find dataframe file: {self.df_path}")
        suffix = self.df_path.suffix.lower()
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(self.df_path, sep=sep)

        # Filter columns
        if self.sequence_col not in df.columns:
            raise KeyError(f"Missing sequence column '{self.sequence_col}' in {self.df_path}")
        for target_col in self.target_cols:
            if target_col not in df.columns:
                raise KeyError(f"Missing target column '{target_col}' in {self.df_path}")
        if self.temperature_col not in df.columns:
            raise KeyError(f"Missing temperature column '{self.temperature_col}' in {self.df_path}")
        keep_cols = [self.sequence_col] + self.target_cols + [self.temperature_col]  # temperature_col is needed for the model but not necessarily a target
        if "assembly_id" in df.columns:
            keep_cols.append("assembly_id")
        df = df[keep_cols].copy()

        # Set types
        df[self.sequence_col] = df[self.sequence_col].astype("string")
        for target_col in self.target_cols:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        # Drop missing rows
        if drop_missing:
            df = df.dropna(subset=[self.sequence_col] + self.target_cols + [self.temperature_col])

        if len(df) == 0:
            raise ValueError(f"No usable rows found in dataframe file: {self.df_path}")
        df = df.reset_index(drop=True)
        return df

    def get_std_norm_stats(self):
        stats = {}
        for target_col in self.target_cols:
            mean = self.df[target_col].mean()
            std = self.df[target_col].std()
            stats[target_col] = {"mean": mean, "std": std}
        return stats

    def __len__(self):
        return len(self.df)

    def _init_random_state(self):
        worker = torch.utils.data.get_worker_info()
        rank_inc = self.parallel_state.rank if self.parallel_state is not None else 0
        worker_inc = worker.id if worker is not None else 0
        total_seed = self.base_seed + rank_inc + worker_inc
        if self.random_seed != total_seed:
            self.random = random.Random(total_seed)
            self.random_seed = total_seed

    def __getitem__(self, idx):
        # Setup
        self._init_random_state()

        # Only one of SP-group needs to read/process sequence
        if self.parallel_state.sp_size > 1 and not is_rank0(self.parallel_state.sp_group):
            return {
                "token_ids": None,
                "temperatures": None,
                "labels": None,
            }
        row = self.df.iloc[idx]
        sequence = str(row[self.sequence_col]).strip().upper()
        token_ids = torch.tensor(self.tokenizer.encode(sequence), dtype=torch.long)
        labels = torch.tensor([float(row[target_col]) for target_col in self.target_cols], dtype=torch.float32)
        item = {
            "token_ids": token_ids,
            "temperatures": float(row[self.temperature_col]),
            "labels": labels,
        }
        if "assembly_id" in self.df.columns and pd.notna(row["assembly_id"]):
            item["assembly_id"] = row["assembly_id"]
        return item
