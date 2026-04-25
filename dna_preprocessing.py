# dna_preprocessing.py

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("Install PyTorch first: pip install torch torchvision torchaudio")

from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------
# Tokenizer Class
# ---------------------------------------------------------
class Tokenizer:
    """
    Turns DNA sequences into overlapping k-mers and builds a vocabulary.
    """

    def __init__(self, k=3):
        self.k = k
        self.vocab = {"<PAD>": 0, "<UNK>": 1}  # Start with special tokens

    def tokenize(self, seq):
        """
        Break a DNA string into overlapping k-mers.
        Example: ATGCG, k=3 -> ["ATG", "TGC", "GCG"]
        """
        return [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]

    def build_vocab(self, sequences):
        """
        Build vocabulary from a list of DNA sequences.
        Each unique k-mer gets a unique integer ID.
        """
        for seq in sequences:
            tokens = self.tokenize(seq)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

    def encode(self, tokens):
        """
        Convert list of tokens into list of integer IDs.
        Unknown tokens map to <UNK>.
        """
        return [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]


# ---------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------
class DNADataset(Dataset):
    """
    Loads DNA sequences from a FASTA file and converts them into
    fixed-length token ID tensors using the Tokenizer.
    """

    def __init__(self, fasta_file, tokenizer, max_length=512):
        self.sequences = self.load_fasta(fasta_file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_fasta(self, path):
        """
        Read a FASTA file and return only clean sequence lines.
        Lines starting with '>' are headers and are ignored.
        Keep only A/C/G/T characters for each sequence (uppercase), skip invalid lines.
        """
        seqs = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(">"):
                    continue
                # Normalize line and strip non-DNA characters
                clean = ''.join([c for c in line.upper() if c in {'A', 'C', 'G', 'T'}])
                if clean:
                    seqs.append(clean)
        return seqs

    def __getitem__(self, idx):
        """
        Return one encoded, padded DNA sequence as a tensor.
        """
        seq = self.sequences[idx]

        # Tokenize into k-mers
        tokens = self.tokenizer.tokenize(seq)

        # Convert tokens to integer IDs
        token_ids = self.tokenizer.encode(tokens)

        # Truncate if too long
        token_ids = token_ids[:self.max_length]

        # Pad if too short
        pad_id = self.tokenizer.vocab["<PAD>"]
        token_ids += [pad_id] * (self.max_length - len(token_ids))

        return torch.tensor(token_ids)

    def __len__(self):
        return len(self.sequences)


# ---------------------------------------------------------
# Dataloader Creation Function
# ---------------------------------------------------------
def create_dataloader(fasta_file, k=3, max_length=512, batch_size=32):
    """
    Convenience function that:
    1. Loads sequences
    2. Builds tokenizer vocabulary
    3. Creates dataset
    4. Wraps it in a DataLoader
    """

    # Step 1: Create tokenizer
    tokenizer = Tokenizer(k=k)

    # Step 2: Load sequences once to build vocab
    temp_dataset = DNADataset(fasta_file, tokenizer, max_length)
    tokenizer.build_vocab(temp_dataset.sequences)

    # Step 3: Create final dataset with full tokenizer
    dataset = DNADataset(fasta_file, tokenizer, max_length)

    # Step 4: Wrap in DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, tokenizer.vocab


# ---------------------------------------------------------
# Example Usage
# ---------------------------------------------------------
if __name__ == "__main__":
    fasta_path = r"C:\Users\jared\OneDrive - University of Tennessee\Documents\DSE 512\dna_project\data.fasta.txt"

    dataloader, vocab = create_dataloader(fasta_path)

    for i, batch in enumerate(dataloader):
        print(f"batch {i} shape: {batch.shape}")

        batch_file = f"batch_{i}.npy"
        import numpy as np
        np.save(batch_file, batch.numpy())
        print(f"saved batch to: {batch_file}")
        break
