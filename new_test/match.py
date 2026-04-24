import dgl
import torch,cfg
from match_model.my.my_infer_demo import batch_pairs_graph
from match_model.my.graph_fusion import GraphEmbedding
from new_test.attn import AttnMatch
from new_test.rule import Sim

class Match(torch.nn.Module):
    def __init__(self, dim, device=cfg.device, method='gnn', n=4):
        super(Match, self).__init__()
        self.method = method
        if self.method == 'gnn':
            self.match = GraphEmbedding(node_in=2, use_edge_feat=True, rep_dim=dim).to(device)
        elif self.method == 'attn':
            self.match = AttnMatch(L=10, n=n, in_dim=2, dim=dim).to(device)
        elif self.method == 'sim':
            self.match = Sim(dim=dim).to(device)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def forward(self, info_pre, info_post, hydra_pre, hydra_post):
        if self.method == 'gnn':
            bg_pairs = get_hydra_graph(info_pre, info_post, hydra_pre, hydra_post)
            ret = self.match(bg_pairs, pairs=cfg.pair_match)
        elif self.method == 'attn':
            ret = self.match(info_pre, info_post, hydra_pre, hydra_post)
        elif self.method == 'sim':
            ret = self.match.fusion_sim(info_pre, info_post, hydra_pre, hydra_post)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return ret.view(-1)

    def valid(self, info_pre, info_post, hydra_pre, hydra_post):
        B1, B2 = info_pre.size(0) , info_post.size(0)
        if self.method == 'gnn':
            bg_pairs = get_valid_hydra_graph(info_pre, info_post, hydra_pre, hydra_post)
            ret = self.match(bg_pairs, pairs=cfg.pair_match)
        elif self.method == 'attn':
            info_pre = info_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1*B2, -1, 2)
            info_post = info_post.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1*B2, -1, 2)
            hydra_pre = hydra_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1*B2, -1, 2, 2)
            hydra_post = hydra_post.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1*B2, -1, 2, 2)
            ret = self.match(info_pre, info_post, hydra_pre, hydra_post)
        elif self.method == 'sim':
            info_pre = info_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1*B2, -1, 2)
            info_post = info_post.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1*B2, -1, 2)
            hydra_pre = hydra_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1*B2, -1, 2, 2)
            hydra_post = hydra_post.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1*B2, -1, 2, 2)
            ret = self.match.fusion_sim(info_pre, info_post, hydra_pre, hydra_post)

        else:
            raise ValueError(f"Unknown method: {self.method}")
        return ret.view(B1, B2)

    def freeze(self, flag=True):
        for param in self.match.parameters():
            param.requires_grad = not flag


def get_hydra_graph(trajA_head, trajB_head, trajA_tail, trajB_tail):
    if trajA_head.shape[-1] != 2:
        trajA_head = torch.transpose(trajA_head, 1, 2)  # (B, k0, 2)
        trajB_head = torch.transpose(trajB_head, 1, 2)  # (B, k0, 2)
        trajA_tail = torch.transpose(trajA_tail, 2, 3)  # (B, n, k, 2)
        trajB_tail = torch.transpose(trajB_tail, 2, 3)  # (B, n, k, 2)

    B = trajB_head.shape[0]
    device = trajA_head.device

    g = batch_pairs_graph(trajA_head, trajA_tail, trajB_head, trajB_tail, device=device)

    return g

def get_valid_hydra_graph(trajA_head, trajB_head, trajA_tail, trajB_tail, pair=True):

    B1 = trajA_head.shape[0]
    B2 = trajB_head.shape[0]
    device = cfg.device
    gs = []
    for i in range(B1):
        # repeat A head for B2 times
        tmp_A_head = trajA_head[i].unsqueeze(0).repeat(B2, 1, 1)  # (B, k0, 2)
        tmp_A_tail = trajA_tail[i].unsqueeze(0).repeat(B2, 1, 1, 1)
        gs.append(batch_pairs_graph(tmp_A_head, tmp_A_tail, trajB_head, trajB_tail, device=device))
    g_pos = dgl.batch(gs)
    return g_pos
