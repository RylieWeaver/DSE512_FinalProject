import math

import torch
import torch.nn as nn

from dse.utils import Config


class GenomeRegressionConfig(Config):
    """Configuration for regression over precomputed genome chunk embeddings."""

    def __init__(
        self,
        input_dim: int,
        max_position_embeddings: int,
        dim: int = 256,
        num_filters: int = 16,
        filter_dim: int | None = None,
        num_heads: int = 8,
        num_transformer_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        output_dim: int = 1,
        **kwargs,
    ):
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if num_filters < 1:
            raise ValueError("num_filters must be at least 1")
        if num_transformer_layers < 1:
            raise ValueError("num_transformer_layers must be at least 1")
        if max_position_embeddings < 1:
            raise ValueError("max_position_embeddings must be at least 1")
        self.input_dim = input_dim
        self.max_position_embeddings = max_position_embeddings
        self.dim = dim
        self.num_filters = num_filters
        self.filter_dim = filter_dim if filter_dim is not None else dim
        if self.filter_dim < 1:
            raise ValueError("filter_dim must be positive")
        self.num_heads = num_heads
        self.num_transformer_layers = num_transformer_layers
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.output_dim = output_dim
        for key, value in kwargs.items():
            setattr(self, key, value)


class ResidualMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(self.norm(x))


class GenomeChunkRegressionModel(nn.Module):
    """
    Hierarchical regression model over ragged genome chunk embeddings.

    Chunk padding is represented by ``chunk_mask=False`` and is excluded from
    chromosome/genome means and assigned ``-inf`` before filter softmax.
    """

    def __init__(self, cfg: GenomeRegressionConfig):
        super().__init__()
        self.cfg = cfg

        self.input_projection = nn.Linear(cfg.input_dim, cfg.dim)
        self.chromosome_context_in = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim), nn.GELU())
        self.chromosome_context_out = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.Dropout(cfg.dropout))
        self.genome_context_in = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim), nn.GELU())
        self.genome_context_out = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.Dropout(cfg.dropout))

        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, cfg.dim)
        self.temperature_embedding = nn.Sequential(
            nn.Linear(1, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.chunk_mlp = ResidualMLP(cfg.dim, cfg.mlp_ratio, cfg.dropout)

        self.filter_norm = nn.LayerNorm(cfg.dim)
        self.filter_projection = nn.Linear(cfg.dim, cfg.filter_dim)
        self.filters = nn.Parameter(torch.empty(cfg.num_filters, cfg.filter_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_transformer_layers,
            norm=nn.LayerNorm(cfg.dim),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(cfg.dim),
            nn.Linear(cfg.dim, cfg.dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.dim, cfg.output_dim),
        )
        nn.init.normal_(self.filters, mean=0.0, std=1 / math.sqrt(cfg.filter_dim))

    @staticmethod
    def _masked_genome_mean(x, chunk_mask):
        mask = chunk_mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_chromosome_context(x, chromosome_ids, chunk_mask):
        """Return the mean for the chromosome associated with every chunk."""
        batch_size, _, dim = x.shape
        safe_ids = chromosome_ids.masked_fill(~chunk_mask, 0)
        num_chromosomes = int(safe_ids.max().item()) + 1

        sums = x.new_zeros(batch_size, num_chromosomes, dim)
        counts = x.new_zeros(batch_size, num_chromosomes, 1)
        expanded_ids = safe_ids.unsqueeze(-1).expand(-1, -1, dim)
        mask = chunk_mask.unsqueeze(-1).to(x.dtype)
        sums.scatter_add_(1, expanded_ids, x * mask)
        counts.scatter_add_(1, safe_ids.unsqueeze(-1), mask)
        means = sums / counts.clamp_min(1.0)
        return means.gather(1, expanded_ids)

    def contextualize_chunks(self, inputs):
        chunks = inputs["chunk_embeddings"]
        chromosome_ids = inputs["chromosome_ids"]
        position_ids = inputs["position_ids"]
        chunk_mask = inputs["chunk_mask"]
        temperatures = inputs["temperatures"]

        if not chunk_mask.any(dim=1).all():
            raise ValueError("Every genome must contain at least one non-padding chunk")
        if position_ids[chunk_mask].max().item() >= self.cfg.max_position_embeddings:
            raise ValueError(
                "A position ID exceeds max_position_embeddings="
                f"{self.cfg.max_position_embeddings}"
            )

        chunks = self.input_projection(chunks)

        chromosome_values = self.chromosome_context_in(chunks)
        chromosome_means = self._masked_chromosome_context(
            chromosome_values, chromosome_ids, chunk_mask
        )
        chunks = chunks + self.chromosome_context_out(chromosome_means)

        genome_values = self.genome_context_in(chunks)
        genome_mean = self._masked_genome_mean(genome_values, chunk_mask)
        chunks = chunks + self.genome_context_out(genome_mean).unsqueeze(1)

        chunks = chunks + self.position_embedding(position_ids)
        chunks = chunks + self.temperature_embedding(temperatures.reshape(-1, 1)).unsqueeze(1)
        chunks = self.chunk_mlp(chunks)
        return chunks.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)

    def pool_chunks(self, chunks, chunk_mask):
        score_features = self.filter_projection(self.filter_norm(chunks))
        logits = torch.einsum("bnf,kf->bkn", score_features, self.filters)
        logits = logits / math.sqrt(self.cfg.filter_dim)
        logits = logits.masked_fill(~chunk_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        tokens = torch.einsum("bkn,bnd->bkd", weights, chunks)
        return tokens, weights

    def forward(self, inputs, labels=None):
        chunks = self.contextualize_chunks(inputs)
        tokens, _ = self.pool_chunks(chunks, inputs["chunk_mask"])
        tokens = self.token_transformer(tokens)
        preds = self.output_head(tokens.mean(dim=1))
        return preds, labels
