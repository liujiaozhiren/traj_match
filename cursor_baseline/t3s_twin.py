"""
T3S-style (Yang et al., ICDE 2021) twin encoders: **spatial** BiLSTM on coordinates +
**structural** Transformer on projected coordinates, fused by **sum** + LayerNorm.

Full T3S also uses a **grid/cell sequence** with self-attention; without map discretization we
use the Transformer path on the same lon/lat sequence as a structural encoder (common workaround).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _sinusoidal_pe(max_len: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model, device=device, dtype=dtype)
    position = torch.arange(0, max_len, dtype=dtype, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=dtype, device=device) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class T3STrajEncoder(nn.Module):
    """Single-branch T3S-style encoder -> (B, embed_dim)."""

    def __init__(
        self,
        embed_dim: int,
        lstm_hidden: int,
        *,
        tf_layers: int = 2,
        nhead: int = 8,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by nhead ({nhead})")
        self.embed_dim = embed_dim
        self.lstm = nn.LSTM(2, lstm_hidden, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(2 * lstm_hidden, embed_dim)

        self.in_proj = nn.Linear(2, embed_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tf = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.norm = nn.LayerNorm(embed_dim)
        pe = _sinusoidal_pe(max_len, embed_dim, torch.device("cpu"), torch.float32)
        self.register_buffer("_pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 2) lon/lat."""
        _, L, _ = x.shape
        o, _ = self.lstm(x)
        lstm_e = self.lstm_proj(o.mean(dim=1))

        pe = self._pe[:, :L, :].to(device=x.device, dtype=x.dtype)
        h = self.in_proj(x) + pe
        h = self.tf(h)
        tf_e = h.mean(dim=1)
        return self.norm(lstm_e + tf_e)


class T3STwin(nn.Module):
    """Separate pre/post T3S-style encoders; embeddings are (B, embed_dim)."""

    def __init__(
        self,
        embed_dim: int = 128,
        lstm_hidden: int = 64,
        tf_layers: int = 2,
        nhead: int = 8,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.enc_pre = T3STrajEncoder(
            embed_dim, lstm_hidden, tf_layers=tf_layers, nhead=nhead, max_len=max_len, dropout=dropout
        )
        self.enc_post = T3STrajEncoder(
            embed_dim, lstm_hidden, tf_layers=tf_layers, nhead=nhead, max_len=max_len, dropout=dropout
        )

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_pre(x)

    def encode_post(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_post(x)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pre(pre), self.encode_post(post)
