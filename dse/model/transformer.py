# General
import math
from typing import Optional

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

# DSE 512
from dse.distributed import ParallelState, _F_Gather_B_ReduceScatter, _F_Mean_B_ReduceScatter
from dse.utils.config import Config



def compute_rope(
    seq_len: int,
    dim: int,
    device,
    start_pos: int = 0,
    base: float = 100000.0,
    gamma: float = 2.0,
    dtype: torch.dtype = torch.float32,
):
    """
    Computing RoPE (Rotary Positional Embeddings) once and applying
    multiple times (rather than recomputing) probably gives some
    efficiency gains, even though it's likely minimal.

    We do a power-warping to bias towards lower frequencies.

    Compute cos/sin in fp32 by default.
    """
    # Setup
    assert dim % 2 == 0
    num_freqs = dim // 2

    # Compute
    pos = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=dtype)      # [S]
    freqs_lin = torch.arange(num_freqs, device=device, dtype=dtype) / num_freqs         # [D/2] (linearly spaced)
    freqs_warp = freqs_lin ** gamma                                                     # Warp t over the interval
    inv_freqs = base ** (-freqs_warp)                                                   # [D/2] (geometric progression with power-warping)
    angles = torch.einsum("s, d -> s d", pos, inv_freqs).to(dtype)                      # [S, D/2] (outer product with einsum to show intent)
    return angles.cos(), angles.sin()                                                   # Each [S, D/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x: [B, S, H, D]
    cos: [S, D/2]
    sin: [S, D/2]

    Apply in whatever dtype x is in.
    """
    # Setup
    assert x.size(-1) % 2 == 0, "RoPE requires an even last dimension"
    cos = rearrange(cos, "s d ->  () s () d").to(x.dtype)               # [1, S, 1, D/2]
    sin = rearrange(sin, "s d ->  () s () d").to(x.dtype)               # [1, S, 1, D/2]

    # Split x in 2 (RoPE is doing a 2D rotation)
    x_even = x[..., 0::2]                                               # [B, S, H, D/2]
    x_odd = x[..., 1::2]                                                # [B, S, H, D/2]

    # Apply RoPE 2D rotation
    x_rope_even = x_even * cos - x_odd * sin
    x_rope_odd = x_even * sin + x_odd * cos

    # Collect
    x = torch.stack((x_rope_even, x_rope_odd), dim=-1).flatten(-2)      # [B, S, H, D]
    return x                                                            # [B, S, H, D]


class Attention(nn.Module):
    def __init__(self, dim, num_heads, parallel_state, use_flash_attn=False):
        super().__init__()
        # Read/check
        self.parallel_state = parallel_state
        self.sp_group = parallel_state.sp_group
        self.sp_size = parallel_state.sp_size
        self.sp_rank = parallel_state.sp_rank
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash_attn = use_flash_attn and flash_attn_func is not None

        # Modules
        self.qkv = nn.Linear(dim, 3*dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, rope_cos, rope_sin):                                               # x: [B, S_sub, D] | rope_cos/sin: [S, D/2]
        # Setup
        B, S_sub, D = x.size()

        # Compute local QKV
        qkv = self.qkv(x)                                                                   # [B, S_sub, 3*D]
        qkv = qkv.view(B, S_sub, self.num_heads, 3 * self.head_dim)                         # [B, S_sub, H, 3*D_head]
        q, k, v = qkv.split(self.head_dim, dim=-1)                                          # [B, S_sub, H, D_head] (each)

        # Apply RoPE to local QKV
        q = apply_rope(q, rope_cos, rope_sin)                                               # [B, S_sub, H, D_head]
        k = apply_rope(k, rope_cos, rope_sin)                                               # [B, S_sub, H, D_head]

        # Collect global K,V
        if self.sp_size > 1:
            k = _F_Gather_B_ReduceScatter.apply(k, self.sp_group, 1)                        # [B, S, H, D_head]
            v = _F_Gather_B_ReduceScatter.apply(v, self.sp_group, 1)                        # [B, S, H, D_head]

        # Compute attention
        # NOTE: Distributes the O(S^2) logit computation across the sp group!
        if self.use_flash_attn:
            # NOTE: Flash attention is not compatible with any custom masks,
            # so we just perform all-to-all attention even with pads. This is
            # somewhat aligned with our data because (1) When sampling from
            # reference genomes, we can make sure our chunks start early enough
            # to have no pads and (2) When sampling from concattenated ribosomal
            # genes, our shorter sequences are moreso caused by lack of data
            # labeling than anything else. Still, future work could look at
            # working in the pad_mask... Flex Attn???
            # NOTE: Flash attention expects inputs in bfloat16 or float16, but
            # we can cast back to the original dtype after.
            attn_scaled_values = flash_attn_func(
                q.contiguous().to(torch.bfloat16),
                k.contiguous().to(torch.bfloat16),
                v.contiguous().to(torch.bfloat16),
                dropout_p=0.0,
                softmax_scale=self.scale,
                causal=False,
            )
            attn_scaled_values = attn_scaled_values.to(x.dtype)                             # [B, S_sub, H, D_head]
        else:
            attn_logits = torch.einsum("bshd,bShd->bsSh", q, k) * self.scale                # [B, S_sub, S, H]
            attn_weights = F.softmax(attn_logits, dim=2)                                    # [B, S_sub, S, H]
            attn_scaled_values = torch.einsum("bsSh,bShd->bshd", attn_weights, v)           # [B, S_sub, H, D_head]

        # Concat heads and output projection
        value = rearrange(attn_scaled_values, "b s h d -> b s (h d)")                       # [B, S_sub, D]
        value = self.out_proj(value)                                                        # [B, S_sub, D]
        return value                                                                        # [B, S_sub, D]


class MLP(nn.Module):
    def __init__(self, dim, parallel_state):
        super().__init__()
        self.dim = dim
        self.hidden_dim = 4 * dim
        self.fc1 = nn.Linear(dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):       # [B, S_sub, D]
        x = self.fc1(x)         # [B, S_sub, 4*D]
        x = F.silu(x)           # [B, S_sub, 4*D]
        x = self.fc2(x)         # [B, S_sub, D]
        return x                # [B, S_sub, D]


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, parallel_state, use_flash_attn=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, parallel_state, use_flash_attn=use_flash_attn)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, parallel_state)

    def forward(self, x, rope_cos, rope_sin):       # each [B, S_sub, D]
        h = self.ln1(x)                             # [B, S_sub, D]
        attn = self.attn(h, rope_cos, rope_sin)     # [B, S_sub, D]
        x = x + attn                                # [B, S_sub, D]
        x = x + self.mlp(self.ln2(x))               # [B, S_sub, D]
        return x                                    # [B, S_sub, D]


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, init_std=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.init_std = init_std if init_std is not None else 1 / math.sqrt(dim)

    def init_weights(self):
        # NOTE: Need to be careful with weight-tied initialization
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.init_std)

    def forward(self, input_ids):
        return self.embedding(input_ids)
    
    def embed(self, input_ids):
        return self.forward(input_ids)
    
    def unembed(self, x):
        return x @ self.embedding.weight.T


class TransformerConfig(Config):
    def __init__(
        self,
        vocab_size: int = 4,
        max_seq_len: int = 1024,
        dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        use_flash_attn: bool = False,
        init_std: Optional[float] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_flash_attn = use_flash_attn
        self.init_std = init_std if init_std is not None else 1 / math.sqrt(dim)


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: TransformerConfig, parallel_state: Optional[ParallelState] = None):
        super().__init__()
        # Read
        self.cfg = cfg
        vocab_size = cfg.vocab_size
        max_seq_len = cfg.max_seq_len
        dim = cfg.dim
        num_heads = cfg.num_heads
        num_layers = cfg.num_layers
        self.use_flash_attn = cfg.use_flash_attn and flash_attn_func is not None
        # NOTE: the class ParallelState holds non-distributed parallelism info by default
        parallel_state = parallel_state if parallel_state is not None else ParallelState()
        self.sp_group = parallel_state.sp_group
        self.sp_size = parallel_state.sp_size
        self.sp_rank = parallel_state.sp_rank

        # Modules
        self.token_emb = TokenEmbedding(vocab_size, dim, init_std=cfg.init_std)
        self._register_rope(max_seq_len=max_seq_len, base=10000.0)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, num_heads, parallel_state=parallel_state, use_flash_attn=self.use_flash_attn)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)
        
    def _register_rope(self, max_seq_len: int = None, base: float = 10000.0):
        """
        This can be re-registered easily if changing contexts during training, with:
        model._register_rope(new_max_seq_len, new_base)
        """
        max_seq_len = max_seq_len if max_seq_len is not None else self.cfg.max_seq_len
        head_dim = self.cfg.dim // self.cfg.num_heads
        rope_cos, rope_sin = compute_rope(
            seq_len=max_seq_len,
            dim=head_dim,
            device=next(self.parameters()).device,
            base=base,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)    # [S, D/2]
        self.register_buffer("rope_sin", rope_sin, persistent=False)    # [S, D/2]

    def forward(self, input_ids):                                       # [B, S_sub]
        # Setup
        B, S = input_ids.shape

        # Get sp-aware idx
        if self.sp_size > 1:
            assert S % self.sp_size == 0, "Sequence length must be divisible by sequence parallel size"
            S_sub = S // self.sp_size
            seq_start_idx = self.sp_rank * S_sub
            seq_end_idx = min((self.sp_rank + 1) * S_sub, S)
        else:
            seq_start_idx, seq_end_idx = 0, S

        # Split items for this sp rank's sequence chunk
        input_ids = input_ids[:, seq_start_idx:seq_end_idx]             # [B, S_sub]
        rope_cos = self.rope_cos[seq_start_idx:seq_end_idx, :]          # [S_sub, D/2]
        rope_sin = self.rope_sin[seq_start_idx:seq_end_idx, :]          # [S_sub, D/2]

        # Transformer blocks
        x = self.token_emb.embed(input_ids)                             # [B, S_sub, D]
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        # Output
        x = self.out_norm(x)                                            # [B, S_sub, D]
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig, parallel_state: Optional[ParallelState] = None):
        super().__init__()
        self.backbone = TransformerBackbone(cfg, parallel_state=parallel_state)

        # Initialization
        self.post_init()

    def post_init(self):
        for module in self.modules():
            if module is not self and hasattr(module, "init_weights") and callable(module.init_weights):
                module.init_weights()

    def no_wd_params(self):
        # Specify parameters that should not receive weight decay
        # (this is model specific and depends on variable names)
        explicit = {"backbone.token_emb.embedding.weight"}
        exist = ("norm", "gamma", "beta")
        affix = (".bias")

        skip = set(explicit)
        for name, _ in self.named_parameters():
            if any(s in name for s in exist) or name.endswith(affix):
                skip.add(name)
        return skip

    def _update_context_len(self, new_context_len: int):
        self.cfg.max_seq_len = new_context_len
        self.backbone._register_rope(new_context_len)


class MLMTransformer(Transformer):
    def __init__(self, cfg: TransformerConfig, parallel_state: Optional[ParallelState] = None):
        super().__init__(cfg, parallel_state=parallel_state)
        self.cfg = cfg
        # NOTE: the class ParallelState holds non-distributed parallelism info by default
        parallel_state = parallel_state if parallel_state is not None else ParallelState()
        self.sp_group = parallel_state.sp_group
        self.sp_size = parallel_state.sp_size
        self.sp_rank = parallel_state.sp_rank

    def forward(self, input_ids, labels):
        # Setup
        B, S = input_ids.shape

        # Get sp-aware idx
        if self.sp_size > 1:
            assert S % self.sp_size == 0, "Sequence length must be divisible by sequence parallel size"
            S_sub = S // self.sp_size
            seq_start_idx = self.sp_rank * S_sub
            seq_end_idx = min((self.sp_rank + 1) * S_sub, S)
        else:
            seq_start_idx, seq_end_idx = 0, S

        # Split labels
        labels = labels[:, seq_start_idx:seq_end_idx]           # [B, S_sub]
        
        # Run through backbone
        x = self.backbone(input_ids)                            # [B, S_sub, D]

        # Task-specific head
        logits = self.backbone.token_emb.unembed(x)             # [B, S_sub, V]
        return logits, labels
