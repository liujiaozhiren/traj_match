from __future__ import annotations

from typing import List

import dgl
import torch
import torch.nn.functional as F

import cfg

from .mygraph import build_one_graph_dgl, build_pair_graph_dgl
from .pairmatch import get_pair_map


def batch_embedding_graph(trajA_pre, trajA_post, trajB_pre, trajB_post, device="cpu"):
    B = len(trajA_pre)
    pairs = []
    for i in range(B):
        gA = build_one_graph_dgl(trajA_pre[i], trajA_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)
        gB = build_one_graph_dgl(trajB_pre[i], trajB_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)
        pairs.append((gA, gB))

    graphs: List = []
    for gA, gB in pairs:
        graphs += [gA.to(device), gB.to(device)]
        bg = dgl.batch(graphs).to(device)
    if "elen" in bg.edata and "ef" not in bg.edata:
        bg.edata["ef"] = bg.edata["elen"]
    return bg


def batch_pairs_graph(trajA_pre, trajA_post, trajB_pre, trajB_post, device="cpu"):
    return batch_pairs_graph_with_pair_map_fn(trajA_pre, trajA_post, trajB_pre, trajB_post, pair_map_fn=get_pair_map, device=device)


def batch_pairs_graph_with_pair_map_fn(trajA_pre, trajA_post, trajB_pre, trajB_post, *, pair_map_fn, device="cpu"):
    B = len(trajA_pre)
    pairs = []
    for i in range(B):
        gA = build_one_graph_dgl(trajA_pre[i], trajA_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)
        gB = build_one_graph_dgl(trajB_pre[i], trajB_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)
        pair_map = pair_map_fn(trajA_post[i], trajB_post[i])
        merged_g = build_pair_graph_dgl(
            gA,
            gB,
            k0=trajA_pre.size(1),
            n=trajA_post.size(1),
            k=trajA_post.size(2),
            t_gap=cfg.diff_infer_len + 1,
            device=device,
            pair_map=pair_map,
        )
        pairs.append(merged_g)
    bg = dgl.batch(pairs).to(device)
    if "elen" in bg.edata and "ef" not in bg.edata:
        bg.edata["ef"] = bg.edata["elen"]
    return bg


def pair_loss(logits, labels):
    if labels.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        labels = labels.float()
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if loss.dim() == 1:
        return loss
    return loss.mean(dim=1)

