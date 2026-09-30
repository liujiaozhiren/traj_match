"""Synthetic / pickle trajectory pairs — self-contained, no repo imports."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import torch

from .constants import INFO_LEN, N_POINTS


def make_synthetic_pairs(n: int, *, seed: int = 0, t_min: int = 16, t_max: int = 48) -> list[tuple]:
    """7-tuple compatible with repo layout: (mid, A, B, L_A, L_B, F_A, F_B)."""
    g = torch.Generator().manual_seed(seed)
    pairs: list[tuple] = []
    for i in range(n):
        ta = int(torch.randint(t_min, t_max + 1, (1,), generator=g).item())
        tb = int(torch.randint(t_min, t_max + 1, (1,), generator=g).item())
        base_lon = 113.9 + 0.01 * (i % 50)
        base_lat = 22.5 + 0.01 * (i % 37)
        A = _random_walk(ta, base_lon, base_lat, g)
        B = _random_walk(tb, base_lon + 0.002, base_lat + 0.002, g)
        pairs.append((f"id_{i}", A, B, A, B, None, None))
    return pairs


def _random_walk(n: int, lon0: float, lat0: float, g: torch.Generator) -> list:
    pts = []
    lon, lat = lon0, lat0
    for t in range(n):
        lon += float(torch.randn((), generator=g).item()) * 1e-4
        lat += float(torch.randn((), generator=g).item()) * 1e-4
        pts.append([float(t), lon, lat])
    return pts


def load_pairs_pkl(path: str | Path, max_pairs: int | None = None) -> list[tuple]:
    raw = pickle.load(open(path, "rb"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"expected non-empty list pickle: {path}")
    if max_pairs is not None:
        raw = raw[: int(max_pairs)]
    return raw


def traj_to_lonlat_tensor(traj: Sequence, device: torch.device) -> torch.Tensor:
    """(T, 2) lon/lat from [t, lon, lat] or [lon, lat]."""
    if not traj:
        return torch.zeros(0, 2, device=device)
    row0 = traj[0]
    if len(row0) >= 3:
        xy = [(float(p[1]), float(p[2])) for p in traj]
    else:
        xy = [(float(p[0]), float(p[1])) for p in traj]
    return torch.tensor(xy, dtype=torch.float32, device=device)


def subsample_lonlat12(traj: Sequence, *, n_points: int = N_POINTS, device: torch.device) -> torch.Tensor:
    """(n_points, 2) evenly subsampled lon/lat."""
    t = traj_to_lonlat_tensor(traj, device=torch.device("cpu"))
    if t.numel() == 0:
        return torch.zeros(n_points, 2, device=device)
    if t.shape[0] == 1:
        out = t.repeat(n_points, 1)
    else:
        idx = torch.linspace(0, t.shape[0] - 1, n_points).round().long()
        out = t[idx]
    return out.to(device)


def pair_pre_post_tensors(
    item: tuple,
    *,
    n_points: int = N_POINTS,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """pre from A, post from reversed B — match protocol."""
    _, A, B, *_ = item
    pre = subsample_lonlat12(A, n_points=n_points, device=device)
    brev = list(reversed(B))
    post = subsample_lonlat12(brev, n_points=n_points, device=device)
    return pre, post


def pair_to_info_xy8(item: tuple, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """8-point xy meters proxy: scale lon/lat for self-contained demo."""
    pre, post = pair_pre_post_tensors(item, n_points=INFO_LEN, device=device)
    scale = 1e5
    return pre * scale, post * scale


def head_to_diff_input(head_xy: torch.Tensor) -> torch.Tensor:
    """(L, 2) -> (1, 2, L) for diffusion UNet."""
    return head_xy.T.unsqueeze(0).contiguous()
