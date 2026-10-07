from typing import Optional

import torch

import torch.nn as nn
import torch.nn.functional as F

from dse.utils import Config


class AutoEncoderConfig(Config):
    """Configuration for the flattened, single-vector DNA autoencoder."""

    def __init__(
        self,
        vocab_size: int = 6,
        chunk_size: int = 2048,
        latent_dim: int = 768,
        expansion_factor: float = 4.0,
        num_layers: int = 1,
        dropout: float = 0.0,
        pad_token_id: int = 4,
        architecture_version: int = 14,
        hidden_dim: Optional[int] = None,
        dtype: str = "float32",
        **kwargs,
    ):
        if chunk_size < 1 or latent_dim < 1 or expansion_factor <= 0 or num_layers < 1:
            raise ValueError("chunk_size, latent_dim, expansion_factor, and num_layers must be positive")
        if architecture_version != 14:
            raise ValueError("Only residual MLP autoencoder architecture_version=14 is supported")
        if hidden_dim is not None and hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.expansion_factor = expansion_factor
        self.num_layers = num_layers
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.architecture_version = architecture_version
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        for key, value in kwargs.items():
            setattr(self, key, value)


class ResidualMLPBlock(nn.Module):
    """Pre-BatchNorm residual block; the identity path is untouched."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, momentum=0.01)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return residual + x


class DNAAutoEncoder(nn.Module):
    """
    MNIST-style DNA autoencoder with exactly one vector across the bottleneck.

    ``[B, S] -> [B, S*V] -> [B, latent_dim] -> [B, S*V]``
    """

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.flat_dim = cfg.chunk_size * cfg.vocab_size
        self.hidden_dim = cfg.hidden_dim or max(1, round(cfg.expansion_factor * cfg.latent_dim))

        encoder_layers = [
            nn.Linear(self.flat_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        ]
        for _ in range(cfg.num_layers - 1):
            encoder_layers.append(ResidualMLPBlock(self.hidden_dim, cfg.dropout))
        encoder_layers.extend([
            nn.Linear(self.hidden_dim, cfg.latent_dim),
            nn.GELU(),
        ])
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(cfg.latent_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        ]
        for _ in range(cfg.num_layers - 1):
            decoder_layers.append(ResidualMLPBlock(self.hidden_dim, cfg.dropout))
        decoder_layers.extend([
            nn.Linear(self.hidden_dim, self.flat_dim),
        ])
        self.decoder = nn.Sequential(*decoder_layers)
        # Cast before loading checkpoints, so float64 weights are never rounded
        # through float32. Old configs default to the original float32 behavior.
        self.to(dtype=getattr(torch, cfg.dtype))

    def _pad_to_chunk_size(self, input_ids):
        original_length = input_ids.size(1)
        if original_length > self.cfg.chunk_size:
            raise ValueError(
                f"Input length {original_length} exceeds configured chunk_size "
                f"{self.cfg.chunk_size}"
            )
        if original_length < self.cfg.chunk_size:
            input_ids = F.pad(
                input_ids,
                (0, self.cfg.chunk_size - original_length),
                value=self.cfg.pad_token_id,
            )
        return input_ids, original_length

    def encode(self, input_ids):
        input_ids, _ = self._pad_to_chunk_size(input_ids)
        one_hot = F.one_hot(input_ids, num_classes=self.cfg.vocab_size).to(
            dtype=next(self.parameters()).dtype
        )
        return self.encoder(one_hot.flatten(start_dim=1))

    def decode(self, latent, output_length: Optional[int] = None):
        if latent.ndim != 2 or latent.size(-1) != self.cfg.latent_dim:
            raise ValueError(
                f"Expected latent shape [B, {self.cfg.latent_dim}], got {tuple(latent.shape)}"
            )
        logits = self.decoder(latent)
        logits = logits.reshape(latent.size(0), self.cfg.chunk_size, self.cfg.vocab_size)
        output_length = self.cfg.chunk_size if output_length is None else output_length
        return logits[:, :output_length]

    def forward(self, input_ids, labels=None):
        original_length = input_ids.size(1)
        latent = self.encode(input_ids)
        logits = self.decode(latent, output_length=original_length)
        return logits, input_ids if labels is None else labels
