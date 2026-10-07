"""MLP regression over stored genome chunk embeddings."""

import torch
from torch import nn


class EmbeddingMLPRegressor(nn.Module):
    """Two hidden layers per chunk, then an equal-weight masked mean."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.model_type = "mlp"
        self.network = nn.Sequential(
            nn.Linear(cfg.input_dim + 1, cfg.dim), nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim), nn.GELU(),
            nn.Linear(cfg.dim, cfg.output_dim),
        )
        self.to(dtype=getattr(torch, getattr(cfg, "dtype", "float32")))

    def forward(self, batch, labels=None):
        chunks, temperatures = batch["chunk_embeddings"], batch["temperatures"]
        chunk_mask = batch["chunk_mask"]
        if chunk_mask.shape != chunks.shape[:2] or not chunk_mask.any(dim=1).all():
            raise ValueError("Every genome needs at least one valid chunk")
        # Exclude padding before the MLP as well as before aggregation.
        chunks = chunks.masked_fill(~chunk_mask.unsqueeze(-1), 0)
        temperature = temperatures.reshape(-1, 1, 1).expand(-1, chunks.shape[1], 1)
        x = torch.cat((chunks, temperature), dim=-1)
        scores = self.network(x)
        scores = scores.masked_fill(~chunk_mask.unsqueeze(-1), 0)
        predictions = scores.sum(dim=1) / chunk_mask.sum(dim=1, keepdim=True)
        return predictions, labels
