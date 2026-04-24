
import torch

from match_model.my.mygraphmatch import GraphEncoderDGL, GraphAggregatorDGL, GraphMatchingNetDGL
import torch.nn as nn

class PairClassifier(nn.Module):
    def __init__(self, gmn: GraphMatchingNetDGL, rep_dim=128):
        super().__init__()
        self.gmn = gmn
        #rep_dim = self.aggregator.out_dim
        self.pair_head = nn.Sequential(
            nn.Linear(rep_dim * 4, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(rep_dim*2, rep_dim),
            nn.ReLU(),
            nn.Linear(rep_dim, 1)
        )

    def forward(self, g):
        """
        g: 按对配好顺序的 batched 图 (DGLGraph)
        返回：每对的二分类 logits [B, 1]
        """
        reps = self.gmn(g)
        g1, g2 = reps[0::2], reps[1::2]  # [B, D], [B, D]
        pair_feat = torch.cat([g1, g2, (g1 - g2).abs(), g1 * g2], dim=-1)  # [B, 4D]
        logits = self.pair_head(pair_feat).squeeze(-1)  # [B]
        return logits


class GraphEmbedding(nn.Module):
    def __init__(self,node_in=2, use_edge_feat=True, node_state_dim=128, rep_dim=128):
        super().__init__()
        if use_edge_feat:
            edge_in = 1  # 我们把边长放到 edata['ef']，维度=1
            edge_state_dim = 64
            edge_hidden_sizes = (64,)
        else:
            edge_in = None
            edge_state_dim = 0
            edge_hidden_sizes = None

        encoder = GraphEncoderDGL(
            node_in=node_in,
            edge_in=edge_in,
            node_hidden_sizes=(node_state_dim,),
            edge_hidden_sizes=edge_hidden_sizes
        )
        aggregator = GraphAggregatorDGL(
            node_state_dim=node_state_dim,
            node_hidden_sizes=(rep_dim,),  # 聚合后向量维度
            graph_transform_sizes=(),  # 如需再变换可填 (rep_dim,)
            gated=True,
            readout='sum'
        )
        gmn = GraphMatchingNetDGL(
            encoder=encoder,
            aggregator=aggregator,
            node_state_dim=node_state_dim,
            edge_state_dim=edge_state_dim,
            edge_hidden_sizes=(128,),
            node_hidden_sizes=(128,),
            n_prop_layers=3,
            node_update_type='residual',
            use_reverse_direction=True,
            reverse_dir_param_different=True,
            layer_norm=False,
            similarity='dotproduct'
        )

        self.gmn = gmn
        self.pair_head = nn.Sequential(
            nn.Linear(rep_dim, 256), nn.ReLU(),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, g, pairs=False):
        """
        g: 按对配好顺序的 batched 图 (DGLGraph)
        返回：每对的二分类 logits [B, 1]
        """
        reps = self.gmn(g)
        # logits = self.pair_head(reps).squeeze(-1)  # [B]
        if pairs:
            reps = self.pair_head(reps).squeeze(-1)  # [B]
        return reps


