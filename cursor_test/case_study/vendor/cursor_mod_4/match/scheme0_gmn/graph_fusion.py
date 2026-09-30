import torch
import torch.nn as nn

from .mygraphmatch import GraphEncoderDGL, GraphAggregatorDGL, GraphMatchingNetDGL


class GraphEmbedding(nn.Module):
    """
    Migrated from match_model/my/graph_fusion.py (scheme0 exact behavior).
    """

    def __init__(self, node_in=2, use_edge_feat=True, node_state_dim=128, rep_dim=128):
        super().__init__()
        if use_edge_feat:
            edge_in = 1
            edge_state_dim = 64
            edge_hidden_sizes = (64,)
        else:
            edge_in = None
            edge_state_dim = 0
            edge_hidden_sizes = None

        encoder = GraphEncoderDGL(node_in=node_in, edge_in=edge_in, node_hidden_sizes=(node_state_dim,), edge_hidden_sizes=edge_hidden_sizes)
        aggregator = GraphAggregatorDGL(
            node_state_dim=node_state_dim,
            node_hidden_sizes=(rep_dim,),
            graph_transform_sizes=(),
            gated=True,
            readout="sum",
        )
        gmn = GraphMatchingNetDGL(
            encoder=encoder,
            aggregator=aggregator,
            node_state_dim=node_state_dim,
            edge_state_dim=edge_state_dim,
            edge_hidden_sizes=(128,),
            node_hidden_sizes=(128,),
            n_prop_layers=3,
            node_update_type="residual",
            use_reverse_direction=True,
            reverse_dir_param_different=True,
            layer_norm=False,
            similarity="dotproduct",
        )
        self.gmn = gmn
        self.pair_head = nn.Sequential(
            nn.Linear(rep_dim, 256),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, g, pairs=False):
        reps = self.gmn(g)
        if pairs:
            reps = self.pair_head(reps).squeeze(-1)
        return reps

