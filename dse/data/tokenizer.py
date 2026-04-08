# General

# Torch
import torch

# DSE512



class BPTokenizer:
    """
    DNA tokenizer:
      - A=0, T=1, U=1, C=2, G=3, N=4, [PAD]=4, [MASK]=5

    Notes:
      - There are encoder collisions such that sometimes,
        decode(encode(x)) != x.
    """
    def __init__(self):
        # Repeated names
        self.PAD = "[PAD]"
        self.MASK = "[MASK]"

        # Token mappings
        self._init_ids()
        self.in_vocab_size = len(self.tok2id.keys())
        self.out_vocab_size = len(set(self.tok2id.values()))  # unique output tokens
        self.target_bp = ["A", "T", "C", "G"]
        self.special_tokens = [self.PAD, self.MASK]

    def _init_ids(self):
        """Initialize token ID mappings that can change by tokenizer."""
        self.tok2id = {
            "A": 0,
            "T": 1,
            "U": 1,  # collision U -> T
            "C": 2,
            "G": 3,
            "N": 4,
            self.PAD: 4,  # collision PAD -> N
            self.MASK: 5,
        }
        self.id2tok = {
            0: "A",
            1: "T",
            2: "C",
            3: "G",
            4: "N",
            5: "[MASK]",
        }

    def _tokenize(self, seq: str) -> list:
        """Match special tokens, else character-wise"""
        toks, i = [], 0
        while i < len(seq):
            matched = False
            for tok in self.special_tokens:
                if seq[i:i + len(tok)] == tok:
                    toks.append(tok)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                toks.append(seq[i].upper())
                i += 1
        return toks

    def encode(self, input) -> torch.LongTensor:
        if isinstance(input, str):
            tokens = self._tokenize(input)
        elif isinstance(input, list):
            tokens = input
        else:
            raise ValueError("Input must be a string or list of strings.")
        return [self.tok2id.get(tok.upper(), self.tok2id["N"]) for tok in tokens]  # unknown chars -> N
            
    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        tokens = [self.id2tok.get(int(i), "N") for i in ids]
        tokens = "".join(tokens)
        return tokens
