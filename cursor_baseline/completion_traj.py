"""
Raw v2 maxdtw pair trajectories: whole A -> whole B_hat (same length), lon/lat only.

No pair_pre_post_lonlat12 resampling / extra Gaussian noise on top of pkl.
"""

from __future__ import annotations

from typing import Any

import torch

from cursor_baseline.lonlat_traj import traj_to_lonlat_tensor


def pair_to_ab_lonlat(item: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    Returns (A_xy, B_xy) each (T, 2) float32, or None if unusable.
    Requires len(A) == len(B) >= 2.
    """
    if not isinstance(item, (list, tuple)) or len(item) < 3:
        return None
    A, B = item[1], item[2]
    if len(A) != len(B) or len(A) < 2:
        return None
    xa = traj_to_lonlat_tensor(A, device)
    xb = traj_to_lonlat_tensor(B, device)
    if xa.shape[0] != xb.shape[0] or xa.numel() == 0:
        return None
    return xa, xb


def mean_step_l2(pred: torch.Tensor, true_b: torch.Tensor) -> torch.Tensor:
    """
    pred, true_b: same shape (T, 2) or (B, T, 2).
    Scalar mean over steps of per-step Euclidean length in (lon, lat).
    """
    d = torch.linalg.vector_norm(pred - true_b, dim=-1)
    return d.mean()


def mean_step_l2_pairwise(pred: torch.Tensor, true_b: torch.Tensor) -> torch.Tensor:
    """
    pred: (k, T, 2), true_b: (k, T, 2)
    dist[i,j] = mean_t ||pred[i,t] - true_b[j,t]||_2
    """
    diff = pred.unsqueeze(1) - true_b.unsqueeze(0)
    d = torch.linalg.vector_norm(diff, dim=-1).mean(dim=-1)
    return d


def scores_neg_mean_step_l2(pred: torch.Tensor, true_b: torch.Tensor) -> torch.Tensor:
    """Higher is better: scores[i,j] = -mean_t L2(pred_i,t, true_j,t)."""
    return -mean_step_l2_pairwise(pred, true_b)
