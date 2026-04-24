"""
Build (lon, lat) tensors from clean_pairs trajectories, resample to fixed length, add noise.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch

from cursor_baseline.model_lib.geo_origin import DEFAULT_ORIGIN_LAT, DEFAULT_ORIGIN_LON


def _choose_lon_lat(a: float, b: float) -> tuple[float, float]:
    d1 = abs(a - DEFAULT_ORIGIN_LON) + abs(b - DEFAULT_ORIGIN_LAT)
    d2 = abs(a - DEFAULT_ORIGIN_LAT) + abs(b - DEFAULT_ORIGIN_LON)
    if d1 <= d2:
        return a, b
    return b, a


def raw_traj_lonlat_envelope(traj: Sequence[Sequence[float]]) -> tuple[float, float, float, float] | None:
    """
    Min/max lon/lat for one raw trajectory. Uses **one** (ch1, ch2) column order per trajectory
    (from the first point’s `_choose_lon_lat` decision), so min/max are not polluted by mixing
    lon/lat into the same axis across points — important for geographic bounding boxes / grids.
    """
    if not traj:
        return None
    a0, b0 = float(traj[0][1]), float(traj[0][2])
    l0, t0 = _choose_lon_lat(a0, b0)
    use_ch1_ch2 = abs(l0 - a0) < 1e-7 and abs(t0 - b0) < 1e-7
    min_lon = min_lat = math.inf
    max_lon = max_lat = -math.inf
    for p in traj:
        a, b = float(p[1]), float(p[2])
        lon, lat = (a, b) if use_ch1_ch2 else (b, a)
        min_lon = min(min_lon, lon)
        max_lon = max(max_lon, lon)
        min_lat = min(min_lat, lat)
        max_lat = max(max_lat, lat)
    return min_lon, max_lon, min_lat, max_lat


def traj_to_lonlat_tensor(traj: Sequence[Sequence[float]], device: torch.device) -> torch.Tensor:
    """traj: list of [t, ch1, ch2] from clean_pairs -> (L,2) lon, lat degrees."""
    pts: List[List[float]] = []
    for p in traj:
        lon, lat = _choose_lon_lat(float(p[1]), float(p[2]))
        pts.append([lon, lat])
    if not pts:
        return torch.zeros(0, 2, device=device, dtype=torch.float32)
    return torch.tensor(pts, device=device, dtype=torch.float32)


def resample_polyline_lonlat(xy: torch.Tensor, n: int) -> torch.Tensor:
    """
    Uniform arc-length resampling along the polyline in (lon, lat) space (L2 on degrees).
    xy: (L, 2)
    """
    if xy.numel() == 0:
        return torch.zeros(n, 2, device=xy.device, dtype=xy.dtype)
    L = xy.size(0)
    if L == 1:
        return xy.repeat(n, 1)
    seg_len = torch.norm(xy[1:] - xy[:-1], dim=1)
    cum = torch.cat([torch.zeros(1, device=xy.device, dtype=xy.dtype), seg_len.cumsum(0)])
    total = cum[-1]
    if float(total.item()) < 1e-12:
        return xy[0:1].repeat(n, 1)
    ts = torch.linspace(0.0, float(total.item()), n, device=xy.device, dtype=xy.dtype)
    idx_hi = torch.searchsorted(cum, ts)
    idx_hi = idx_hi.clamp(1, L - 1)
    idx_lo = idx_hi - 1
    c0 = cum[idx_lo]
    c1 = cum[idx_hi]
    seg_t = c1 - c0
    seg_t = torch.where(seg_t < 1e-12, torch.ones_like(seg_t), seg_t)
    w = (ts - c0) / seg_t
    p0 = xy[idx_lo]
    p1 = xy[idx_hi]
    return (1.0 - w.unsqueeze(1)) * p0 + w.unsqueeze(1) * p1


def add_lonlat_noise(xy: torch.Tensor, std: float, rng_seed: int) -> torch.Tensor:
    if std <= 0:
        return xy
    with torch.random.fork_rng():
        torch.manual_seed(int(rng_seed))
        noise = torch.randn(xy.shape, device=xy.device, dtype=xy.dtype)
    return xy + noise * float(std)


def pair_pre_post_lonlat12(
    item,
    *,
    n_points: int,
    noise_std_deg: float,
    noise_seed_pre: int,
    noise_seed_post: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre = traj A; post = reversed B (same convention as match training).
    Both resampled to n_points in lon/lat, then independent Gaussian noise.
    """
    _, A, B, _, _, _, _ = item
    Brev = list(reversed(B))
    pre = traj_to_lonlat_tensor(A, device)
    post = traj_to_lonlat_tensor(Brev, device)
    pre12 = resample_polyline_lonlat(pre, n_points)
    post12 = resample_polyline_lonlat(post, n_points)
    pre12 = add_lonlat_noise(pre12, noise_std_deg, noise_seed_pre)
    post12 = add_lonlat_noise(post12, noise_std_deg, noise_seed_post)
    return pre12, post12


def stack_subsample_lonlat(
    valid_pairs: list,
    samp: torch.Tensor,
    *,
    n_points: int,
    noise_std_deg: float,
    noise_seed_base: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """samp: (k,) long indices into valid_pairs; returns (k, n_points, 2) lon/lat."""
    k = int(samp.numel())
    pres: list[torch.Tensor] = []
    posts: list[torch.Tensor] = []
    for j in range(k):
        idx = int(samp[j].item())
        pre, post = pair_pre_post_lonlat12(
            valid_pairs[idx],
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            noise_seed_pre=int(noise_seed_base) + j * 3,
            noise_seed_post=int(noise_seed_base) + j * 3 + 1,
            device=device,
        )
        pres.append(pre)
        posts.append(post)
    return torch.stack(pres, dim=0), torch.stack(posts, dim=0)
