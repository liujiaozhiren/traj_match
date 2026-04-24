import dgl
import torch

import cfg
from match_model.my.my_infer_demo import batch_pairs_graph, batch_embedding_graph


def get_hydra_graph_label(trajA_head, trajB_head, trajA_tail, trajB_tail, pair=True, valid=False):
    if trajA_head.shape[-1] != 2:
        trajA_head = torch.transpose(trajA_head, 1, 2)  # (B, k0, 2)
        trajB_head = torch.transpose(trajB_head, 1, 2)  # (B, k0, 2)
        trajA_tail = torch.transpose(trajA_tail, 2, 3)  # (B, n, k, 2)
        trajB_tail = torch.transpose(trajB_tail, 2, 3)  # (B, n, k, 2)

    if valid:
         return get_hydra_graph_valid(trajA_head, trajB_head, trajA_tail, trajB_tail, pair=pair)

    def derangement_indices(n, device):
        """生成一个无固定点的置换（derangement）。n=1 时无法生成。"""
        if n == 1:
            raise ValueError("batch size=1 无法构造负样本（derangement）。")
        idx = torch.arange(n, device=cfg.device)
        shift = torch.randint(1, n, (1,), device=device).item()  # 1..n-1 之间的循环移位
        return (idx + shift) % n


    batch_func = batch_pairs_graph if pair else batch_embedding_graph
    B = trajB_head.shape[0]
    device = cfg.device

    g_pos = batch_func(trajA_head, trajA_tail, trajB_head, trajB_tail, device=device)
    y_pos = torch.ones(B, dtype=torch.long, device=device)


    perm = derangement_indices(B, device)
    g_neg = batch_func(trajA_head, trajA_tail, trajB_head[perm], trajB_tail[perm], device=device)
    y_neg = torch.zeros(B, dtype=torch.long, device=device)

    g_all = dgl.batch([g_pos, g_neg]) if isinstance(g_pos, dgl.DGLGraph) else (g_pos + g_neg)
    y_all = torch.cat([y_pos, y_neg], dim=0)

    return g_all, y_all

def get_hydra_graph_valid(trajA_head, trajB_head, trajA_tail, trajB_tail, pair=True):

    batch_func = batch_pairs_graph if pair else batch_embedding_graph
    B1 = trajA_head.shape[0]
    B2 = trajB_head.shape[0]
    device = cfg.device
    gs = []
    for i in range(B1):
        # repeat A head for B2 times
        tmp_A_head = trajA_head[i].unsqueeze(0).repeat(B2, 1, 1)  # (B, k0, 2)
        tmp_A_tail = trajA_tail[i].unsqueeze(0).repeat(B2, 1, 1, 1)
        gs.append(batch_func(tmp_A_head, tmp_A_tail, trajB_head, trajB_tail, device=device))
    g_pos = dgl.batch(gs)
    return g_pos

def hydra_graph(trajA_head, trajB_head, trajA_tail, trajB_tail, pair=True):
    if trajA_head.shape[-1] != 2:
        trajA_head = torch.transpose(trajA_head, 1, 2)  # (B, k0, 2)
        trajB_head = torch.transpose(trajB_head, 1, 2)  # (B, k0, 2)
        trajA_tail = torch.transpose(trajA_tail, 2, 3)  # (B, n, k, 2)
        trajB_tail = torch.transpose(trajB_tail, 2, 3)  # (B, n, k, 2)
    batch_func = batch_pairs_graph if pair else batch_embedding_graph
    # B = trajB_head.shape[0]
    device = cfg.device

    g = batch_func(trajA_head, trajA_tail, trajB_head, trajB_tail, device=device)
    return g
