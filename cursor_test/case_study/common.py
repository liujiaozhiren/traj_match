"""Stack-agnostic helpers shared by the workers and the orchestrator.

No dependency on either ``cfg`` stack, so safe to import anywhere.
"""

from __future__ import annotations

import pickle
import random
from typing import Sequence

import numpy as np
import torch


def load_pairs(path) -> list:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"pairs pkl must be a non-empty list: {path}")
    if not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7:
        raise ValueError("pairs items must be 7-tuples (mid,A,B,L_A,L_B,F_A,F_B)")
    return raw


def valid_global_indices(n: int, sample_ratio: float, sample_seed: int) -> list[int]:
    """Identical valid split to both stacks (sample_db / split_train_valid_indices).

    Both call ``random.seed(seed); random.sample(range(n)|pairs, k)`` so positions match.
    """
    random.seed(int(sample_seed))
    k = max(1, int(n * float(sample_ratio)))
    return random.sample(range(n), k)


def pool_global_indices(
    n_pairs: int,
    *,
    frac: float,
    sample_ratio: float,
    sample_seed: int,
    pool_seed: int,
) -> list[int]:
    """A small fixed pool: subsample ``frac`` of the held-out valid set."""
    vg = valid_global_indices(n_pairs, sample_ratio, sample_seed)
    nv = len(vg)
    kk = max(2, int(round(nv * float(frac))))
    kk = min(kk, nv)
    g = torch.Generator()
    g.manual_seed(int(pool_seed))
    perm = torch.randperm(nv, generator=g)[:kk].tolist()
    return [vg[i] for i in perm]


def straightness(xy: np.ndarray) -> float:
    """Net displacement / path length in [0,1]. 1 = straight line, ~0 = tangled."""
    if len(xy) < 2:
        return 0.0
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path = float(seg.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    return net / max(path, 1e-12)


def pool_smooth_indices(
    all_pairs,
    *,
    frac: float,
    sample_ratio: float,
    sample_seed: int,
    pool_seed: int,
    smooth_lo: float = 0.6,
    smooth_hi: float = 0.985,
) -> list[int]:
    """Pool of the *cleanest-looking* held-out pairs.

    Among the valid split, keep pairs whose PRE and POST are both fairly straight
    (``min(straightness) in [smooth_lo, smooth_hi]`` -- the upper bound drops
    near-perfect lines that all look identical), then take the straightest
    ``frac`` of them. ``pool_seed`` only breaks ties / shuffles within equal score.
    """
    vg = valid_global_indices(len(all_pairs), sample_ratio, sample_seed)
    clean = []
    for g in vg:
        m = min(straightness(pre_lonlat(all_pairs[g])), straightness(post_lonlat(all_pairs[g])))
        if smooth_lo <= m <= smooth_hi:
            clean.append(g)
    kk = max(2, int(round(len(vg) * float(frac))))
    if len(clean) <= kk:
        return clean
    # randomly sample kk of the clean pairs -> regular-looking pool with seed variety
    rng = np.random.default_rng(int(pool_seed))
    return [int(g) for g in rng.choice(np.array(clean), size=kk, replace=False)]


def pre_lonlat(pair) -> np.ndarray:
    """Clean query-PRE polyline (lon,lat) from field A. For plotting only."""
    A = pair[1]
    return np.asarray([[float(p[1]), float(p[2])] for p in A], dtype=np.float64)


def post_lonlat(pair) -> np.ndarray:
    """Clean POST polyline (lon,lat) = reversed(B). For plotting only."""
    B = list(reversed(pair[2]))
    return np.asarray([[float(p[1]), float(p[2])] for p in B], dtype=np.float64)
