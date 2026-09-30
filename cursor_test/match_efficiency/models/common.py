"""Shared trajectory encoders (dim=128)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import EMBED_DIM, N_HEADS, RNN_HIDDEN


class GruTrajEncoder(nn.Module):
    """(B, L, 2) -> (B, embed_dim) mean-pool BiGRU."""

    def __init__(self, embed_dim: int = EMBED_DIM, rnn_hidden: int = RNN_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(2, rnn_hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * rnn_hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.proj(y.mean(dim=1))


class GruLastStepEncoder(nn.Module):
    """Traj2SimVec style: last timestep."""

    def __init__(self, embed_dim: int = EMBED_DIM, rnn_hidden: int = RNN_HIDDEN):
        super().__init__()
        self.gru = nn.GRU(2, rnn_hidden, batch_first=True)
        self.proj = nn.Linear(rnn_hidden, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.proj(y[:, -1, :])


class TwinEncoder(nn.Module):
    def __init__(self, encoder_cls, **kw):
        super().__init__()
        self.enc_pre = encoder_cls(**kw)
        self.enc_post = encoder_cls(**kw)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_pre(x)

    def encode_post(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_post(x)


class T3SEncoder(nn.Module):
    """BiLSTM + Transformer branch, fused sum (dim=128)."""

    def __init__(self, embed_dim: int = EMBED_DIM, lstm_hidden: int = RNN_HIDDEN):
        super().__init__()
        assert embed_dim % N_HEADS == 0
        self.lstm = nn.LSTM(2, lstm_hidden, batch_first=True, bidirectional=True)
        self.lstm_proj = nn.Linear(2 * lstm_hidden, embed_dim)
        self.in_proj = nn.Linear(2, embed_dim)
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=N_HEADS, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_h = self.lstm_proj(self.lstm(x)[0].mean(dim=1))
        t = self.in_proj(x)
        t = self.tf(t).mean(dim=1)
        return self.norm(lstm_h + t)


class NeuTrajEncoder(nn.Module):
    """Normalized lon/lat + BiGRU (stard path surrogate)."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.register_buffer("mean", torch.zeros(2))
        self.register_buffer("std", torch.ones(2))
        self.core = GruTrajEncoder(embed_dim)

    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = (x - self.mean.view(1, 1, 2)) / self.std.view(1, 1, 2).clamp(min=1e-6)
        return self.core(xn)


def batched_cdist_scores(z_q: torch.Tensor, z_gallery: torch.Tensor) -> torch.Tensor:
    """z_q (D,), z_gallery (N,D) -> (N,) higher=better."""
    return -torch.cdist(z_q.unsqueeze(0), z_gallery).squeeze(0)


class PairMlpScorer(nn.Module):
    """DAN/TAT pair head: (B, 2D) -> (B,) or batched N pairs."""

    def __init__(self, embed_dim: int = EMBED_DIM, mlp_hidden: int = 256):
        super().__init__()
        self.ln = nn.LayerNorm(2 * embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )

    def score_all(self, u_q: torch.Tensor, u_gallery: torch.Tensor) -> torch.Tensor:
        """u_q (D,), u_gallery (N,D) -> (N,) logits."""
        n = u_gallery.shape[0]
        pair = torch.cat([u_q.unsqueeze(0).expand(n, -1), u_gallery], dim=-1)
        return self.mlp(self.ln(pair)).squeeze(-1)


class ChainGnnEncoder(nn.Module):
    """TrajGAT / ST2Vec surrogate: chain graph over subsampled points."""

    def __init__(self, embed_dim: int = EMBED_DIM, n_layers: int = 2):
        super().__init__()
        self.node = nn.Linear(2, embed_dim)
        self.convs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.node(x)
        for conv in self.convs:
            left = F.pad(h[:, :-1], (0, 0, 1, 0))
            right = F.pad(h[:, 1:], (0, 0, 0, 1))
            neigh = 0.5 * (left + right)
            h = F.relu(conv(neigh) + h)
        return h.mean(dim=1)
