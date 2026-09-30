from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def _greedy_bipartite_from_scores(scores: torch.Tensor) -> List[int]:
    """
    scores: (n,n) larger is better.
    Returns a permutation-like assignment list assign[i]=j (injective, greedy).
    """
    n = scores.size(0)
    flat = scores.reshape(-1)
    order = torch.argsort(flat, descending=True)
    used_i = torch.zeros(n, dtype=torch.bool, device=scores.device)
    used_j = torch.zeros(n, dtype=torch.bool, device=scores.device)
    assign = [-1] * n
    for idx in order.tolist():
        i = idx // n
        j = idx % n
        if not used_i[i] and not used_j[j]:
            assign[i] = j
            used_i[i] = True
            used_j[j] = True
            if used_i.all():
                break
    # fill any remaining rows (shouldn't happen often) with first free col
    if any(a < 0 for a in assign):
        free = [j for j in range(n) if not used_j[j]]
        it = iter(free)
        for i in range(n):
            if assign[i] < 0:
                assign[i] = next(it)
    return assign


def sinkhorn(scores: torch.Tensor, n_iters: int = 30, eps: float = 1e-6) -> torch.Tensor:
    """
    scores: (n,n) unnormalized log-scores (larger is better).
    Returns a soft assignment P (n,n) approximately doubly-stochastic.
    """
    # log-space stability
    logp = scores - scores.max()
    P = torch.exp(logp)
    for _ in range(int(n_iters)):
        P = P / (P.sum(dim=1, keepdim=True) + eps)
        P = P / (P.sum(dim=0, keepdim=True) + eps)
    return P


def scheme1_pair_map_sinkhorn(trajA_post: torch.Tensor, trajB_post: torch.Tensor, tau: float = 1.0, n_iters: int = 30) -> List[int]:
    """
    trajA_post, trajB_post: (n,k,2) in same coordinate system.
    Uses Euclidean reverse-mean distance -> Sinkhorn -> greedy hard assignment.
    """
    assert trajA_post.shape == trajB_post.shape and trajA_post.ndim == 3
    n, k, _ = trajA_post.shape
    B_rev = trajB_post[:, torch.arange(k - 1, -1, -1, device=trajB_post.device), :]
    # cost (n,n): mean_t ||A[i,t]-B[j,rev_t]||
    A = trajA_post[:, None, :, :]  # (n,1,k,2)
    B = B_rev[None, :, :, :]  # (1,n,k,2)
    cost = torch.norm(A - B, dim=-1).mean(dim=-1)  # (n,n)
    scores = (-cost / float(tau)).to(trajA_post.device)
    P = sinkhorn(scores, n_iters=n_iters)
    return _greedy_bipartite_from_scores(P)


def scheme2_pair_map_attn(trajA_post: torch.Tensor, trajB_post: torch.Tensor) -> List[int]:
    """
    C1-style: tail-token dot-product attention -> greedy hard assignment.
    """
    assert trajA_post.shape == trajB_post.shape and trajA_post.ndim == 3
    # deterministic tail token: mean point (n,2)
    a = trajA_post.mean(dim=1)
    b = trajB_post.mean(dim=1)
    # dot-product attention score (n,n)
    scores = a @ b.t()
    return _greedy_bipartite_from_scores(scores)

