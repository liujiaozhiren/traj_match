# -*- coding: utf-8 -*-
"""
MOT Deep Affinity Network on trajectories.

v1 simplified: BiGRU + pair MLP, optional ``(B+1)×(B+1)`` unmatched row/column
with bidirectional softmax (SST-style) at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajFeatureNet(nn.Module):
    """共享编码器: (B, L, 2) → (B, embed_dim)。"""

    def __init__(self, *, rnn_hidden: int = 128, embed_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.rnn_hidden = int(rnn_hidden)
        self.embed_dim = int(embed_dim)
        self.rnn = nn.GRU(
            input_size=2,
            hidden_size=self.rnn_hidden,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
        )
        rnn_out = 2 * self.rnn_hidden
        self.out_proj = nn.Linear(rnn_out, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.rnn(x)
        pooled = y.mean(dim=1)
        return self.out_proj(pooled)


class DANMotTrajSimplified(nn.Module):
    """
    DAN-traj: pairing MLP; optional unmatched dimension for full affinity matrix.
    """

    def __init__(
        self,
        *,
        rnn_hidden: int = 128,
        embed_dim: int = 128,
        rnn_layers: int = 1,
        mlp_hidden: int = 256,
        use_unmatched_dim: bool = False,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.use_unmatched_dim = bool(use_unmatched_dim)
        self.backbone = TrajFeatureNet(rnn_hidden=rnn_hidden, embed_dim=embed_dim, num_layers=rnn_layers)
        in_pair = 2 * embed_dim
        self.pair_ln = nn.LayerNorm(in_pair)
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_pair, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.unmatched_embed = nn.Parameter(torch.zeros(self.embed_dim))

    def _pair_logits(self, f_pre: torch.Tensor, f_post: torch.Tensor) -> torch.Tensor:
        bp, bq = f_pre.shape[0], f_post.shape[0]
        c = f_pre.shape[1]
        pre_e = f_pre.unsqueeze(1).expand(bp, bq, c)
        post_e = f_post.unsqueeze(0).expand(bp, bq, c)
        pair = torch.cat([pre_e, post_e], dim=-1)
        z = self.pair_ln(pair.reshape(bp * bq, -1))
        return self.pair_mlp(z).view(bp, bq)

    def _append_unmatched(self, f: torch.Tensor) -> torch.Tensor:
        dummy = self.unmatched_embed.unsqueeze(0).expand(f.shape[0], -1)
        return torch.cat([f, dummy], dim=0)

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        f_pre = self.backbone(pre)
        f_post = self.backbone(post)
        if self.use_unmatched_dim:
            f_pre = self._append_unmatched(f_pre)
            f_post = self._append_unmatched(f_post)
        logits = self._pair_logits(f_pre, f_post)
        if self.use_unmatched_dim:
            return self._bidirectional_affinity(logits)
        return logits

    @staticmethod
    def _bidirectional_affinity(logits: torch.Tensor) -> torch.Tensor:
        x_f = F.softmax(logits, dim=1)
        x_t = F.softmax(logits, dim=0)
        return (x_f + x_t) / 2.0

    def score_query_gallery(self, u_q: torch.Tensor, u_gallery: torch.Tensor) -> torch.Tensor:
        """1 query vs N gallery; with unmatched → (2)×(N+1) affinity, return N match scores."""
        n = u_gallery.shape[0]
        if not self.use_unmatched_dim:
            pair = torch.cat([u_q.unsqueeze(0).expand(n, -1), u_gallery], dim=-1)
            return self.pair_mlp(self.pair_ln(pair)).squeeze(-1)

        f_pre = torch.stack([u_q, self.unmatched_embed], dim=0)
        f_post = torch.cat([u_gallery, self.unmatched_embed.unsqueeze(0)], dim=0)
        aff = self._bidirectional_affinity(self._pair_logits(f_pre, f_post))
        return aff[0, :n]
