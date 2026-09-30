"""
Equal-length trajectory distances on R^2 (lon/lat degrees or meters): batch (B, L, 2).
Similarity for retrieval: -distance (higher = closer).
"""

from __future__ import annotations

import torch


def dtw_batch_equal_len(trajs1: torch.Tensor, trajs2: torch.Tensor) -> torch.Tensor:
    """(B,L,2) -> (B,) sum along DTW path."""
    if trajs1.shape != trajs2.shape:
        raise ValueError(f"shape mismatch {trajs1.shape} vs {trajs2.shape}")
    B, L, _ = trajs1.shape
    cost = torch.cdist(trajs1, trajs2, p=2)
    dp = torch.empty((B, L, L), device=trajs1.device, dtype=trajs1.dtype)
    dp[:, 0, 0] = cost[:, 0, 0]
    for i in range(1, L):
        dp[:, i, 0] = cost[:, i, 0] + dp[:, i - 1, 0]
    for j in range(1, L):
        dp[:, 0, j] = cost[:, 0, j] + dp[:, 0, j - 1]
    for i in range(1, L):
        for j in range(1, L):
            m = torch.stack([dp[:, i - 1, j], dp[:, i, j - 1], dp[:, i - 1, j - 1]], dim=1)
            m = torch.min(m, dim=1).values
            dp[:, i, j] = cost[:, i, j] + m
    return dp[:, -1, -1]


def hausdorff_batch_equal_len(trajs1: torch.Tensor, trajs2: torch.Tensor) -> torch.Tensor:
    """Directed Hausdorff max(min) symmetric; same as new_test.rule.Sim.hausdorff_batch."""
    cost = torch.cdist(trajs1, trajs2, p=2)  # B, L, L
    min_ab = cost.min(dim=2).values
    h_ab = min_ab.max(dim=1).values
    min_ba = cost.min(dim=1).values
    h_ba = min_ba.max(dim=1).values
    return torch.maximum(h_ab, h_ba)


def discrete_frechet_batch_equal_len(trajs1: torch.Tensor, trajs2: torch.Tensor) -> torch.Tensor:
    """Discrete Fréchet distance (Eiter & Mannila), L2 on points."""
    if trajs1.shape != trajs2.shape:
        raise ValueError(f"shape mismatch {trajs1.shape} vs {trajs2.shape}")
    B, L, _ = trajs1.shape
    cost = torch.cdist(trajs1, trajs2, p=2)
    ca = torch.empty((B, L, L), device=trajs1.device, dtype=trajs1.dtype)
    ca[:, 0, 0] = cost[:, 0, 0]
    for i in range(1, L):
        ca[:, i, 0] = torch.maximum(cost[:, i, 0], ca[:, i - 1, 0])
    for j in range(1, L):
        ca[:, 0, j] = torch.maximum(cost[:, 0, j], ca[:, 0, j - 1])
    for i in range(1, L):
        for j in range(1, L):
            prev = torch.min(
                torch.stack([ca[:, i - 1, j], ca[:, i, j - 1], ca[:, i - 1, j - 1]], dim=1),
                dim=1,
            ).values
            ca[:, i, j] = torch.maximum(cost[:, i, j], prev)
    return ca[:, -1, -1]


def _directional_mean_min_pt_to_segments_batch(trajs_from: torch.Tensor, trajs_to: torch.Tensor) -> torch.Tensor:
    """
    Mean over vertices of trajs_from of min distance to polyline trajs_to (segment-wise).
    trajs_* : (B, L, 2)
    """
    B, L, _ = trajs_from.shape
    if L == 1:
        return torch.norm(trajs_from - trajs_to, dim=-1).squeeze(-1)  # (B,)

    seg0 = trajs_to[:, :-1, :]  # B, L-1, 2
    seg1 = trajs_to[:, 1:, :]
    p = trajs_from.unsqueeze(2)  # B, L, 1, 2
    a = seg0.unsqueeze(1)  # B, 1, L-1, 2
    b = seg1.unsqueeze(1)
    ab = b - a
    ap = p - a
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-12)
    t = (ap * ab).sum(dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    proj = a + t.unsqueeze(-1) * ab
    dists = torch.norm(p - proj, dim=-1)  # B, L, L-1
    per_v = dists.min(dim=-1).values  # B, L
    return per_v.mean(dim=-1)  # B


def sspd_batch_equal_len(trajs1: torch.Tensor, trajs2: torch.Tensor) -> torch.Tensor:
    """
    Symmetric Segment-Path style distance:
      0.5 * ( mean_i min_seg d(p_i, Q) + mean_j min_seg d(q_j, P) )
    where min is over segments of the other polyline (vertices as length-1 polyline).
    """
    d12 = _directional_mean_min_pt_to_segments_batch(trajs1, trajs2)
    d21 = _directional_mean_min_pt_to_segments_batch(trajs2, trajs1)
    return 0.5 * (d12 + d21)


_BATCH_FNS = {
    "dtw": dtw_batch_equal_len,
    "hausdorff": hausdorff_batch_equal_len,
    "frechet": discrete_frechet_batch_equal_len,
    "sspd": sspd_batch_equal_len,
}


def pairwise_neg_distance_matrix(
    pre: torch.Tensor,
    post: torch.Tensor,
    metric: str,
    *,
    chunk_queries: int | None = None,
) -> torch.Tensor:
    """
    pre, post: (N, L, 2). Returns (N, N) with M[i,j] = -dist(pre[i], post[j]) (higher = better match).
    """
    if metric not in _BATCH_FNS:
        raise ValueError(f"metric must be one of {list(_BATCH_FNS)}, got {metric!r}")
    fn = _BATCH_FNS[metric]
    n, l, _ = pre.shape
    device = pre.device
    out = torch.empty((n, n), device=device, dtype=pre.dtype)
    chunk = n if chunk_queries is None else int(chunk_queries)
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        b = i1 - i0
        q = pre[i0:i1].unsqueeze(1).expand(b, n, l, 2).reshape(b * n, l, 2)
        p = post.unsqueeze(0).expand(b, n, l, 2).reshape(b * n, l, 2)
        d = fn(q, p).view(b, n)
        out[i0:i1] = -d
    return out


def topk_hit_rate(similarity: torch.Tensor, k: int) -> float:
    n = similarity.shape[0]
    if n == 0:
        return 0.0
    kk = min(k, n)
    topk_idx = torch.topk(similarity, k=kk, dim=1).indices
    correct = torch.arange(n, device=similarity.device, dtype=topk_idx.dtype).unsqueeze(1)
    hits = (topk_idx == correct).any(dim=1).float()
    return float(hits.mean().item())
