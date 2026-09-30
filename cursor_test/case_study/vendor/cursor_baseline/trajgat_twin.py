"""
Twin GraphTransformer encoders (official trajGAT / model/trajGAT/model.py).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from cursor_baseline.model_lib.trajGAT.model import GraphTransformer


class TrajGATTwin(nn.Module):
    def __init__(
        self,
        pre_embedding: torch.Tensor,
        *,
        d_model: int,
        d_input: int = 4,
        num_head: int = 8,
        num_encoder_layers: int = 3,
        d_lap_pos: int = 8,
        encoder_dropout: float = 0.01,
        in_feat_dropout: float = 0.0,
    ):
        super().__init__()
        emb1 = copy.deepcopy(pre_embedding)
        emb2 = copy.deepcopy(pre_embedding)
        self.enc_pre = GraphTransformer(
            emb1,
            d_input=d_input,
            d_model=d_model,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            d_lap_pos=d_lap_pos,
            encoder_dropout=encoder_dropout,
            layer_norm=False,
            batch_norm=True,
            in_feat_dropout=in_feat_dropout,
        )
        self.enc_post = GraphTransformer(
            emb2,
            d_input=d_input,
            d_model=d_model,
            num_head=num_head,
            num_encoder_layers=num_encoder_layers,
            d_lap_pos=d_lap_pos,
            encoder_dropout=encoder_dropout,
            layer_norm=False,
            batch_norm=True,
            in_feat_dropout=in_feat_dropout,
        )

    def encode_pre(self, g) -> torch.Tensor:
        return self.enc_pre(g)

    def encode_post(self, g) -> torch.Tensor:
        return self.enc_post(g)

    def forward(self, g_pre, g_post) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_pre(g_pre), self.encode_post(g_post)
