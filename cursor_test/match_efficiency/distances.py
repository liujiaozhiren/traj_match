"""Trajectory distances on CPU — geometric baselines are not GPU workloads."""

from __future__ import annotations

import numpy as np
import torch

_METRICS = ("dtw", "hausdorff", "frechet", "sspd")
_CPU = torch.device("cpu")


def _as_cpu_f32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device=_CPU, dtype=torch.float32)


def _dtw_numpy(a: np.ndarray, b: np.ndarray) -> float:
    m, n = a.shape[0], b.shape[0]
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    dp = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i, j] = dist[i - 1, j - 1] + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[m, n])


def _frechet_numpy(a: np.ndarray, b: np.ndarray) -> float:
    m, n = a.shape[0], b.shape[0]
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    ca = np.full((m, n), np.inf, dtype=np.float64)
    ca[0, 0] = dist[0, 0]
    for i in range(1, m):
        ca[i, 0] = max(dist[i, 0], ca[i - 1, 0])
    for j in range(1, n):
        ca[0, j] = max(dist[0, j], ca[0, j - 1])
    for i in range(1, m):
        for j in range(1, n):
            ca[i, j] = max(dist[i, j], min(ca[i - 1, j], ca[i, j - 1], ca[i - 1, j - 1]))
    return float(ca[m - 1, n - 1])


def hausdorff_batch_equal_len(trajs1: torch.Tensor, trajs2: torch.Tensor) -> torch.Tensor:
    cost = torch.cdist(trajs1, trajs2, p=2)
    h_ab = cost.min(dim=2).values.max(dim=1).values
    h_ba = cost.min(dim=1).values.max(dim=1).values
    return torch.maximum(h_ab, h_ba)


def sspd_batch_equal_len(trajs1: torch.Tensor, trajs2: torch.Tensor) -> torch.Tensor:
    def _dir(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        _b, l, _ = a.shape
        if l == 1:
            return torch.norm(a - b, dim=-1).squeeze(-1)
        seg0, seg1 = b[:, :-1], b[:, 1:]
        p = a.unsqueeze(2)
        ab = seg1.unsqueeze(1) - seg0.unsqueeze(1)
        ap = p - seg0.unsqueeze(1)
        denom = (ab * ab).sum(dim=-1).clamp(min=1e-12)
        t = ((ap * ab).sum(dim=-1) / denom).clamp(0.0, 1.0)
        proj = seg0.unsqueeze(1) + t.unsqueeze(-1) * ab
        return torch.norm(p - proj, dim=-1).min(dim=-1).values.mean(dim=-1)

    return 0.5 * (_dir(trajs1, trajs2) + _dir(trajs2, trajs1))


def _dp_numpy_distances(
    query: np.ndarray,
    gallery: np.ndarray,
    metric: str,
    *,
    j0: int,
    j1: int,
) -> np.ndarray:
    fn = _dtw_numpy if metric == "dtw" else _frechet_numpy
    out = np.empty(j1 - j0, dtype=np.float64)
    for k, j in enumerate(range(j0, j1)):
        out[k] = fn(query, gallery[j])
    return out


def _vectorized_cpu_distances(
    query: torch.Tensor,
    gallery: torch.Tensor,
    metric: str,
    *,
    j0: int,
    j1: int,
) -> torch.Tensor:
    fn = hausdorff_batch_equal_len if metric == "hausdorff" else sspd_batch_equal_len
    b = j1 - j0
    l = query.shape[0]
    q_exp = query.unsqueeze(0).expand(b, l, 2)
    g_exp = gallery[j0:j1]
    return fn(q_exp, g_exp)


def query_gallery_distances(
    query: torch.Tensor,
    gallery: torch.Tensor,
    metric: str,
    *,
    gallery_batch: int | None = None,
) -> torch.Tensor:
    """
    query: (L, 2), gallery: (N, L, 2) -> (N,) distances (lower = closer).
    Always runs on CPU. DTW / Fréchet use NumPy DP; Hausdorff / SSPD use vectorized CPU torch.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {list(_METRICS)}")
    query = _as_cpu_f32(query)
    gallery = _as_cpu_f32(gallery)
    n = gallery.shape[0]
    chunk = n if gallery_batch is None else min(int(gallery_batch), n)
    out = np.empty(n, dtype=np.float64)
    q_np = query.numpy()
    g_np = gallery.numpy()
    for j0 in range(0, n, chunk):
        j1 = min(j0 + chunk, n)
        if metric in ("dtw", "frechet"):
            out[j0:j1] = _dp_numpy_distances(q_np, g_np, metric, j0=j0, j1=j1)
        else:
            out[j0:j1] = _vectorized_cpu_distances(query, gallery, metric, j0=j0, j1=j1).numpy()
    return torch.from_numpy(out)


def query_gallery_scores(query: torch.Tensor, gallery: torch.Tensor, metric: str, **kw) -> torch.Tensor:
    return -query_gallery_distances(query, gallery, metric, **kw)
