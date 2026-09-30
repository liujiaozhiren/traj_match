"""
Traj2SimVec twin encoders (pre/post), last-timestep embedding — aligned with model/handler traj2simvec use.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cursor_baseline.model_lib.traj2simvec.model import Traj2SimVec


class Traj2SimVecTwin(nn.Module):
    """pre/post Traj2SimVec; forward returns z_pre, z_post (B, dim) from last timestep."""

    def __init__(
        self,
        rnn_dim: int,
        *,
        num_lstm_layers: int = 4,
        num_subpart_blocks: int = 1,
    ):
        super().__init__()
        self.rnn_dim = rnn_dim
        kw = dict(num_lstm_layers=num_lstm_layers, num_subpart_blocks=num_subpart_blocks)
        self.enc_pre = Traj2SimVec(rnn_dim, **kw)
        self.enc_post = Traj2SimVec(rnn_dim, **kw)

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 2)"""
        return self.enc_pre(x)[:, -1, :]

    def encode_post(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_post(x)[:, -1, :]

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pre(pre), self.encode_post(post)
