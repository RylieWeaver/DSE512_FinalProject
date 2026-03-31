import pyfastx
from itertools import product
import random

class Tokenization:
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        
    def checked_loaded_data(self):
        if self.data is None:
            raise RuntimeError("Call load_data() first before tokenize")
    
    
    def load_data(self):
        self.data = pyfastx.Fasta(self.data_path)

        print(f"Number of sequences : {len(self.data):,}")
        print(f"Total genome size : {self.data.size:,} bp")
        print(f"GC content : {self.data.gc_content:.2f}%")

        for seq in self.data:
            print(f"  {seq.name:<30}  length={len(seq):>12,} bp")
        
        
    def character_level(self):
        self.checked_loaded_data()
        dna_voc = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        id2base = {v: k for k, v in dna_voc.items()} 
        
        all_token_ids = {}
        for seq in self.data:
            
            #Encode 'ACGTN' => [0, 1, 2, 3, 4]
            ids = [dna_voc.get(b.upper(), 4) for b in seq.seq]
            all_token_ids[seq.name] = ids
        
            #Decode for sanity check
            decoded = ''.join(id2base[i] for i in ids)
            
            print(f"{seq.name:<30}  {len(seq):>12,} bp  →  {len(ids):>12,} tokens")
            print(f"  original : {seq.seq[:20]}")
            print(f"  encoded  : {ids[:20]}")
            print(f"  decoded  : {decoded[:20]}")
            print()

        print(f"Done — {len(all_token_ids)} sequences tokenized")
        return all_token_ids
    
    
    
    def k_mer(self, k:int):
        self.checked_loaded_data()
        
        kmers = [''.join(p) for p in product('ACGT', repeat=k)] 
        kmer_voc = {kmer: i for i, kmer in enumerate(kmers)}
        all_token_ids = {}

        for seq in self.data:
            
            dna_string = seq.seq.upper()
            
            UNK = len(kmer_voc)
            ids = [kmer_voc.get(dna_string[i:i+k], UNK)  for i in range(0, len(dna_string) - k + 1, k)]
            
            all_token_ids[seq.name] = ids
            
            print(f"{seq.name:<30}  {len(seq):>12,} bp  →  {len(ids):>12,} tokens")
            print(f" first 3 k-mers: {[dna_string[i:i+k] for i in range(0, 3*k, k)]}")
            print(f" encoded: {ids[:3]}")
            print()
        return all_token_ids
    
    
    def overlaping_k_mer(self, k:int, stride:int):
        self.checked_loaded_data()
        kmers = [''.join(p) for p in product('ACGT', repeat=k)]
        kmer_voc = {kmer: i for i, kmer in enumerate(kmers)}

        print(f"Vocab size: {len(kmer_voc):,} possible {k}-mers")

        all_token_ids = {}

        for seq in self.data:

            dna_string = seq.seq.upper()
            UNK = len(kmer_voc)
            ids = [kmer_voc.get(dna_string[i:i+k], UNK) for i in range(0, len(dna_string) - k + 1, stride)]

            all_token_ids[seq.name] = ids
            
            print(f"{seq.name:<30}  {len(seq):>12,} bp  →  {len(ids):>12,} tokens")
            print(f"first 4 k-mers : {[dna_string[i:i+k] for i in range(0, 4*stride, stride)]}")
            print(f"encoded : {ids[:4]}")
            print()

        print(f"Done - {len(all_token_ids)} sequences tokenized")
        return all_token_ids
    

class On_the_Fly_Tokenization : 
    
    def __init__(self, data_path, chunk : int, k : int, stride: int):
        self.data_path = data_path
        self.data      = None
        self.names     = None
        self.weights   = None
        self.k       = k
        self.dna_voc = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        kmers = [''.join(p) for p in product('ACGT', repeat=k)]
        self.kmer_voc = {kmer: i for i, kmer in enumerate(kmers)}
        self.UNK  = len(self.kmer_voc)
        self.stride = stride
    
        
    def checked_loaded_data(self):
        if self.data is None:
            raise RuntimeError("Call load_data() first before tokenize")
    
    
    def load_data(self):
        self.data = pyfastx.Fasta(self.data_path)
        
        print(f"Number of sequences : {len(self.data):,}")
        print(f"Total genome size : {self.data.size:,} bp")
        print(f"GC content : {self.data.gc_content:.2f}%")

        self.names, lengths = [], []
        for seq in self.data:                        
            self.names.append(seq.name)
            lengths.append(len(seq))
            print(f"  {seq.name:<30}  length={len(seq):>12,} bp")

        total  = sum(lengths)
        self.weights = [l / total for l in lengths]
        
        
    def get_random_chunk(self, chunk_size=8000):
        self.checked_loaded_data()
        name = random.choices(self.names, weights=self.weights, k=1)[0]
        seq  = self.data[name]
        start = random.randint(0, len(seq) - chunk_size)
        return seq[start:start + chunk_size].seq 
           
    
    def on_the_fly_character_level(self, raw: str = None):
        # self.checked_loaded_data()
        # dna_voc = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        
        # if raw is not None:
        #     ids     = [dna_voc.get(b.upper(), 4) for b in raw]
        #     return ids
        raw = (raw or self.get_random_chunk()).upper()
        return [self.dna_voc.get(b, 4) for b in raw]
        
        
    def on_the_fly_k_mer(self, raw: str = None):
        # self.checked_loaded_data()
        
        # kmers = [''.join(p) for p in product('ACGT', repeat=k)]
        # kmer_voc = {kmer: i for i, kmer in enumerate(kmers)}
        # UNK = len(kmer_voc)

        # if raw is not None:
        #     raw = raw.upper()
        #     return [kmer_voc.get(raw[i:i+k], UNK) for i in range(0, len(raw) - k + 1, k)]
        
        raw = (raw or self.get_random_chunk()).upper()
        return [self.kmer_voc.get(raw[i:i+self.k], self.UNK) for i in range(0, len(raw) - self.k + 1, self.stride)]

        
    def on_the_fly_overlaping_k_mer(self, raw: str = None):
        # self.checked_loaded_data()
        
        # kmers = [''.join(p) for p in product('ACGT', repeat=k)]
        # kmer_voc = {kmer: i for i, kmer in enumerate(kmers)}
        # UNK = len(kmer_voc)

        # if raw is not None:
        #     raw = raw.upper()
        #     return [kmer_voc.get(raw[i:i+k], UNK) for i in range(0, len(raw) - k + 1, stride)]
        raw = (raw or self.get_random_chunk()).upper()
        return [self.kmer_voc.get(raw[i:i+self.k], self.UNK) for i in range(0, len(raw) - self.k + 1, self.stride)]