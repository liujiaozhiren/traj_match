"""
Coordinate utilities for Shenzhen match_traj pairs.

Input points from `new_test/proc.clean_pairs` are like:
  [dt, a, b]  where `a` and `b` are near (114.xx, 22.xx) after normalization.

Unfortunately the repo historically mixes names (lat/lon). We therefore:
- treat the two channels as "lon_like" and "lat_like" by proximity to origin;
- convert degrees to local planar meters (x_east, y_north) around DEFAULT_ORIGIN.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch

DEFAULT_ORIGIN_LON = 114.057868
DEFAULT_ORIGIN_LAT = 22.543099


def _choose_lon_lat(a: float, b: float, *, origin_lon: float, origin_lat: float) -> Tuple[float, float]:
    """
    Given two numbers that should be (lon, lat) but might be swapped, pick the ordering
    that is closer to the expected Shenzhen origin.
    """
    d1 = abs(a - origin_lon) + abs(b - origin_lat)  # assume (a=lon, b=lat)
    d2 = abs(a - origin_lat) + abs(b - origin_lon)  # assume swapped
    if d1 <= d2:
        return a, b
    return b, a


def lonlat_deg_to_xy_m(
    lon: float,
    lat: float,
    *,
    origin_lon: float = DEFAULT_ORIGIN_LON,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
) -> Tuple[float, float]:
    """
    Equirectangular approximation around origin (good for small areas).
    """
    lat0 = math.radians(origin_lat)
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * math.cos(lat0)
    x = (lon - origin_lon) * m_per_deg_lon
    y = (lat - origin_lat) * m_per_deg_lat
    return x, y


def match_traj_pts_to_xy_m(
    pts: Sequence[Sequence[float]],
    *,
    origin_lon: float = DEFAULT_ORIGIN_LON,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
) -> List[Tuple[float, float]]:
    """
    Convert a match_traj sequence (list of [dt,a,b]) into list of (x,y) meters.
    """
    out: List[Tuple[float, float]] = []
    for p in pts:
        a = float(p[1])
        b = float(p[2])
        lon, lat = _choose_lon_lat(a, b, origin_lon=origin_lon, origin_lat=origin_lat)
        out.append(lonlat_deg_to_xy_m(lon, lat, origin_lon=origin_lon, origin_lat=origin_lat))
    return out


def match_traj_pts_to_xy_m_torch(
    pts: torch.Tensor,
    *,
    origin_lon: float = DEFAULT_ORIGIN_LON,
    origin_lat: float = DEFAULT_ORIGIN_LAT,
) -> torch.Tensor:
    """
    pts: (L,3) or (B,L,3) tensor, where last dim is [dt,a,b].
    returns: (L,2) or (B,L,2) meters.
    """
    if pts.dim() == 2:
        a = pts[:, 1]
        b = pts[:, 2]
        # pick order by comparing to origin (vectorized heuristic)
        d1 = (a - origin_lon).abs() + (b - origin_lat).abs()
        d2 = (a - origin_lat).abs() + (b - origin_lon).abs()
        lon = torch.where(d1 <= d2, a, b)
        lat = torch.where(d1 <= d2, b, a)
        lat0 = math.radians(origin_lat)
        m_per_deg_lat = 110540.0
        m_per_deg_lon = 111320.0 * math.cos(lat0)
        x = (lon - origin_lon) * m_per_deg_lon
        y = (lat - origin_lat) * m_per_deg_lat
        return torch.stack([x, y], dim=-1)

    if pts.dim() == 3:
        a = pts[:, :, 1]
        b = pts[:, :, 2]
        d1 = (a - origin_lon).abs() + (b - origin_lat).abs()
        d2 = (a - origin_lat).abs() + (b - origin_lon).abs()
        lon = torch.where(d1 <= d2, a, b)
        lat = torch.where(d1 <= d2, b, a)
        lat0 = math.radians(origin_lat)
        m_per_deg_lat = 110540.0
        m_per_deg_lon = 111320.0 * math.cos(lat0)
        x = (lon - origin_lon) * m_per_deg_lon
        y = (lat - origin_lat) * m_per_deg_lat
        return torch.stack([x, y], dim=-1)

    raise ValueError(f"pts must be 2D or 3D, got shape {tuple(pts.shape)}")


def rmse_l2_m(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> torch.Tensor:
    """
    pred_xy/target_xy: (B,2,L) or (B,L,2) tensors in meters.
    returns: scalar RMSE in meters (averaged over time points per sample, then mean over batch)
    """
    if pred_xy.dim() != 3 or target_xy.dim() != 3:
        raise ValueError("pred_xy and target_xy must be 3D tensors")
    if pred_xy.shape != target_xy.shape:
        raise ValueError(f"shape mismatch: {tuple(pred_xy.shape)} vs {tuple(target_xy.shape)}")
    # normalize to (B,2,L)
    if pred_xy.shape[1] == 2:
        p = pred_xy
        t = target_xy
    else:
        p = pred_xy.transpose(1, 2)
        t = target_xy.transpose(1, 2)
    per_t = ((p - t) ** 2).sum(dim=1).mean(dim=1)  # (B,)
    return per_t.sqrt().mean()

