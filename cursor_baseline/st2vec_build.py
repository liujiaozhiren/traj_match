"""
ST2Vec-style road graph + mapper for `train_repr_baseline` (repo `model/ST2Vec` stack).

Uses `build_graph_from_trajs` (rounded lon/lat → node ids + edges). Node features are small
random init; GCN in `ST_Encoder` learns from them. Time branch uses sinusoidal encoding (no
external Date2Vec checkpoint).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
from torch_geometric.data import Data

from cursor_baseline.lonlat_traj import pair_pre_post_lonlat12, traj_to_lonlat_tensor
from cursor_baseline.st2vec_vendor_load import get_build_graph_from_trajs


def _sinusoidal_time_matrix(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """(L, dim) positional encoding (no learned weights)."""
    if length <= 0:
        return torch.zeros(0, dim, device=device, dtype=torch.float32)
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half = dim // 2
    div = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1)))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div.unsqueeze(0))
    pe[:, 1::2] = torch.cos(pos * div.unsqueeze(0))
    if dim % 2 == 1:
        pe[:, -1] = torch.sin(pos.squeeze(1) * 0.01)
    return pe


def time_seq_to_st2vec_list(te: torch.Tensor) -> list[list[float]]:
    """TimeEmbedding expects nested lists (batch of (L, D))."""
    return te.detach().float().cpu().numpy().tolist()


def flat_trajs_for_st2vec_graph(train_pairs: Sequence[tuple]) -> list[list[list[float]]]:
    """Full pre/post trajectories as [[t_ms, lon, lat], ...] for graph construction."""
    cpu = torch.device("cpu")
    out: list[list[list[float]]] = []
    for item in train_pairs:
        _, A, B, *_ = item
        for traj in (A, list(reversed(B))):
            xy = traj_to_lonlat_tensor(traj, cpu)
            if xy.numel() == 0:
                continue
            n = int(xy.shape[0])
            pts = [[1000.0 * i, float(xy[i, 0]), float(xy[i, 1])] for i in range(n)]
            out.append(pts)
    return out


def build_st2vec_road_network(
    train_pairs: Sequence[tuple],
    *,
    embed_dim: int,
    mapper_k: int,
    device: torch.device,
) -> tuple[Data, Any]:
    build_graph_from_trajs = get_build_graph_from_trajs()

    trajs = flat_trajs_for_st2vec_graph(train_pairs)
    if not trajs:
        raise RuntimeError("ST2Vec: no trajectories to build graph (empty train_pairs?)")
    edge_df, _node_df, _edge_index, mapper = build_graph_from_trajs(trajs, k=int(mapper_k))
    num_node = len(mapper.id2coord)
    ei = torch.as_tensor(edge_df[["s_node", "e_node"]].to_numpy(), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_df[["dist"]].to_numpy(), dtype=torch.float32, device=device)
    if ei.numel() == 0 or ei.size(1) == 0:
        ei = torch.arange(num_node, dtype=torch.long, device=device).unsqueeze(0).repeat(2, 1)
        edge_attr = torch.ones(num_node, edge_attr.size(1) if edge_attr.numel() else 1, device=device)
    x = torch.randn(num_node, int(embed_dim), device=device, dtype=torch.float32) * 0.02
    road = Data(x=x, edge_index=ei.to(device=device), edge_attr=edge_attr)
    return road, mapper


def clamp_node_ids(ids: list, max_idx: int) -> list:
    """
    TrajMapper can assign new ids after the road graph was built; GCN node rows are fixed.
    Clamp indices to [0, max_idx] so ``index_select`` stays in range (avoids CUDA assert).
    """
    if max_idx < 0:
        return [0] * len(ids)
    out: list = []
    for i in ids:
        ii = int(i)
        if ii < 0:
            ii = 0
        elif ii > max_idx:
            ii = max_idx
        out.append(ii)
    return out


def lonlat_tensor_to_mapper_traj(xy: torch.Tensor, t0_ms: float) -> list[list[float]]:
    """(L,2) lon/lat degrees → [[t, lon, lat], ...] for TrajMapper."""
    xy = xy.detach().float().cpu()
    L = int(xy.shape[0])
    return [[t0_ms + i * 60_000.0, float(xy[i, 0]), float(xy[i, 1])] for i in range(L)]


def batch_pairs_to_st2vec_inputs(
    mapper: Any,
    batch: list,
    *,
    n_points: int,
    noise_std_deg: float,
    noise_base: int,
    embed_dim: int,
    device: torch.device,
    max_node_idx: int,
) -> tuple[list, list, list, list]:
    """Lists for ST_Encoder: pre/post id sequences + time nested lists."""
    cpu = torch.device("cpu")
    pre_ids: list = []
    pre_te: list = []
    post_ids: list = []
    post_te: list = []
    for j, item in enumerate(batch):
        pre, post = pair_pre_post_lonlat12(
            item,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            noise_seed_pre=int(noise_base) + j * 3,
            noise_seed_post=int(noise_base) + j * 3 + 1,
            device=cpu,
        )
        t0p = float(noise_base) + j * 1_000_000.0
        t0q = t0p + 50_000_000.0
        traj_p = lonlat_tensor_to_mapper_traj(pre, t0p)
        traj_q = lonlat_tensor_to_mapper_traj(post, t0q)
        ids_p = mapper.traj_to_id_seq(traj_p)
        ids_q = mapper.traj_to_id_seq(traj_q)
        if not ids_p:
            ids_p = [0]
        if not ids_q:
            ids_q = [0]
        ids_p = clamp_node_ids(ids_p, max_node_idx)
        ids_q = clamp_node_ids(ids_q, max_node_idx)
        te_p = _sinusoidal_time_matrix(len(ids_p), embed_dim, device)
        te_q = _sinusoidal_time_matrix(len(ids_q), embed_dim, device)
        pre_ids.append(ids_p)
        post_ids.append(ids_q)
        pre_te.append(time_seq_to_st2vec_list(te_p))
        post_te.append(time_seq_to_st2vec_list(te_q))
    return pre_ids, pre_te, post_ids, post_te


def st2vec_valid_encode_batch(
    mapper: Any,
    valid_pairs: list,
    samp: torch.Tensor,
    *,
    n_points: int,
    noise_std_deg: float,
    noise_base: int,
    embed_dim: int,
    device: torch.device,
    max_node_idx: int,
) -> tuple[list, list, list, list]:
    batch = [valid_pairs[int(samp[j].item())] for j in range(int(samp.numel()))]
    return batch_pairs_to_st2vec_inputs(
        mapper,
        batch,
        n_points=n_points,
        noise_std_deg=noise_std_deg,
        noise_base=noise_base,
        embed_dim=embed_dim,
        device=device,
        max_node_idx=max_node_idx,
    )
