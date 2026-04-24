from __future__ import annotations

from typing import Callable, Tuple

import dgl
import torch

import cfg


BatchPairsFunc = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str], dgl.DGLGraph]


def _ensure_xy(trajA_head, trajB_head, trajA_tail, trajB_tail):
    if trajA_head.shape[-1] != 2:
        trajA_head = torch.transpose(trajA_head, 1, 2)
        trajB_head = torch.transpose(trajB_head, 1, 2)
        trajA_tail = torch.transpose(trajA_tail, 2, 3)
        trajB_tail = torch.transpose(trajB_tail, 2, 3)
    return trajA_head, trajB_head, trajA_tail, trajB_tail


def _derangement_indices(n: int, device):
    if n == 1:
        raise ValueError("batch size=1 无法构造负样本（derangement）。")
    idx = torch.arange(n, device=device)
    shift = torch.randint(1, n, (1,), device=device).item()
    return (idx + shift) % n


def get_hydra_graph_label_with_batch_func(
    *,
    batch_pairs_graph: BatchPairsFunc,
    trajA_head: torch.Tensor,
    trajB_head: torch.Tensor,
    trajA_tail: torch.Tensor,
    trajB_tail: torch.Tensor,
    valid: bool = False,
) -> Tuple[dgl.DGLGraph, torch.Tensor] | dgl.DGLGraph:
    trajA_head, trajB_head, trajA_tail, trajB_tail = _ensure_xy(trajA_head, trajB_head, trajA_tail, trajB_tail)
    device = cfg.device
    if valid:
        B1 = trajA_head.shape[0]
        B2 = trajB_head.shape[0]
        gs = []
        for i in range(B1):
            tmp_A_head = trajA_head[i].unsqueeze(0).repeat(B2, 1, 1)
            tmp_A_tail = trajA_tail[i].unsqueeze(0).repeat(B2, 1, 1, 1)
            gs.append(batch_pairs_graph(tmp_A_head, tmp_A_tail, trajB_head, trajB_tail, device=device))
        return dgl.batch(gs)

    B = trajB_head.shape[0]
    g_pos = batch_pairs_graph(trajA_head, trajA_tail, trajB_head, trajB_tail, device=device)
    y_pos = torch.ones(B, dtype=torch.long, device=device)
    perm = _derangement_indices(B, device=device)
    g_neg = batch_pairs_graph(trajA_head, trajA_tail, trajB_head[perm], trajB_tail[perm], device=device)
    y_neg = torch.zeros(B, dtype=torch.long, device=device)
    g_all = dgl.batch([g_pos, g_neg])
    y_all = torch.cat([y_pos, y_neg], dim=0)
    return g_all, y_all

