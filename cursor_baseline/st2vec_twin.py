"""
Twin ST2Vec encoders (repo `model/ST2Vec.model.model_network.ST_Encoder`): GCN + time branch +
co-attention + BiLSTM pooling. Shared road-network `torch_geometric.data.Data`.
"""

from __future__ import annotations

import torch.nn as nn
from torch_geometric.data import Data

from cursor_baseline.st2vec_vendor_load import get_st_encoder_class

ST_Encoder = get_st_encoder_class()


class ST2VecTwin(nn.Module):
    def __init__(self, road_network: Data, embed_dim: int, device_str: str):
        super().__init__()
        if embed_dim % 4 != 0:
            raise ValueError(f"ST2Vec embed_dim must be divisible by 4 (Transformer nhead=4), got {embed_dim}")
        self.road = road_network
        self.embed_dim = int(embed_dim)
        self.pre = ST_Encoder(
            feature_size=embed_dim,
            date2vec_size=embed_dim,
            embedding_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            dropout_rate=0,
            device=device_str,
        )
        self.post = ST_Encoder(
            feature_size=embed_dim,
            date2vec_size=embed_dim,
            embedding_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            dropout_rate=0,
            device=device_str,
        )

    def _net(self) -> Data:
        d = next(self.parameters()).device
        return self.road.to(d)

    def encode_pre(self, traj_seqs: list, time_seqs: list):
        return self.pre(self._net(), traj_seqs, time_seqs)

    def encode_post(self, traj_seqs: list, time_seqs: list):
        return self.post(self._net(), traj_seqs, time_seqs)

    def forward(self, pre_ids: list, pre_te: list, post_ids: list, post_te: list):
        net = self._net()
        return self.pre(net, pre_ids, pre_te), self.post(net, post_ids, post_te)
