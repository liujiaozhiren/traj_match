"""
Quadtree + node2vec cell embeddings + TrajGraphDataLoader graph construction (official trajGAT stack).
"""

from __future__ import annotations

from typing import Any, Sequence

import dgl
import torch

from cursor_baseline.lonlat_traj import pair_pre_post_lonlat12, traj_to_lonlat_tensor
from cursor_baseline.model_lib.trajGAT.loader import TrajGraphDataLoader
from cursor_baseline.model_lib.trajGAT.traj_stats import traj_statistics
from cursor_baseline.model_lib.trajGAT.tree_build import build_qtree, get_pre_embedding


def tensor_to_lonlat_tuple_list(t: torch.Tensor) -> list[tuple[float, float]]:
    t = t.detach().cpu()
    return [(float(t[i, 0]), float(t[i, 1])) for i in range(int(t.shape[0]))]


def _pair_full_lon_lat_trajs(item: tuple) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    _, A, B, _, _, _, _ = item
    brev = list(reversed(B))
    cpu = torch.device("cpu")
    pre = tensor_to_lonlat_tuple_list(traj_to_lonlat_tensor(A, cpu))
    post = tensor_to_lonlat_tuple_list(traj_to_lonlat_tensor(brev, cpu))
    return pre, post


def flat_trajs_from_train_pairs(train_pairs: Sequence[tuple]) -> list[list[tuple[float, float]]]:
    out: list[list[tuple[float, float]]] = []
    for item in train_pairs:
        pre, post = _pair_full_lon_lat_trajs(item)
        out.append(pre)
        out.append(post)
    return out


def build_trajgat_index(
    train_pairs: Sequence[tuple],
    *,
    d_model: int,
    max_items: int = 50,
    max_depth: int = 50,
    d_lap_pos: int = 8,
) -> tuple[Any, Any, torch.Tensor, TrajGraphDataLoader]:
    """
    Returns:
      stats, qtree, qtree_name2id, pre_embedding (for GraphTransformer),
      a TrajGraphDataLoader instance used only for _collate_func (graph build).
    """
    traj_all = flat_trajs_from_train_pairs(train_pairs)
    stats = traj_statistics(traj_all)
    qtree = build_qtree(traj_all, stats["x_range"], stats["y_range"], max_items=max_items, max_depth=max_depth)
    qtree_name2id, pre_embedding = get_pre_embedding(qtree, d_model)

    dummy_pairs = [[[], []]]
    helper = TrajGraphDataLoader(
        dummy_pairs,
        qtree,
        qtree_name2id,
        d_lap_pos=d_lap_pos,
        num_workers=0,
        traj_stats=stats,
    )
    return stats, qtree, qtree_name2id, pre_embedding, helper


def batch_pairs_to_graphs(
    helper: TrajGraphDataLoader,
    pair_items: list[tuple],
    *,
    n_points: int,
    noise_std_deg: float,
    noise_base: int,
    device: torch.device,
    noise_blend_alpha: float | None = None,
    noise_blend_check_label_ts: bool = False,
) -> tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """
    pair_items: raw clean_pairs 7-tuples; noisy 12-pt lon/lat -> trajGAT graphs.
    Returns batched DGL graphs for pre and post.
    """
    pairs_pp: list[list] = []
    cpu = torch.device("cpu")
    for j, item in enumerate(pair_items):
        pre_t, post_t = pair_pre_post_lonlat12(
            item,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            noise_seed_pre=int(noise_base) + j * 3,
            noise_seed_post=int(noise_base) + j * 3 + 1,
            device=cpu,
            noise_blend_alpha=noise_blend_alpha,
            noise_blend_check_label_ts=noise_blend_check_label_ts,
        )
        pairs_pp.append([tensor_to_lonlat_tuple_list(pre_t), tensor_to_lonlat_tuple_list(post_t)])

    trajgraph_list_list, _ = helper._collate_func(pairs_pp)
    g_pre = dgl.batch(trajgraph_list_list[0])
    g_post = dgl.batch(trajgraph_list_list[1])
    return g_pre.to(device), g_post.to(device)


def graphs_for_valid_indices(
    helper: TrajGraphDataLoader,
    valid_pairs: list,
    samp: torch.Tensor,
    *,
    n_points: int,
    noise_std_deg: float,
    noise_base: int,
    device: torch.device,
    noise_blend_alpha: float | None = None,
    noise_blend_check_label_ts: bool = False,
) -> tuple[dgl.DGLGraph, dgl.DGLGraph]:
    """samp: (k,) indices into valid_pairs."""
    items = [valid_pairs[int(samp[j].item())] for j in range(samp.numel())]
    return batch_pairs_to_graphs(
        helper,
        items,
        n_points=n_points,
        noise_std_deg=noise_std_deg,
        noise_base=noise_base,
        device=device,
        noise_blend_alpha=noise_blend_alpha,
        noise_blend_check_label_ts=noise_blend_check_label_ts,
    )


