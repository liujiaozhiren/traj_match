"""
Train/valid split shared by `run_ft_match_84` and diffusion FT on `--pairs-pkl`.

Uses the same semantics as legacy `sample_db` style: hold out a random subset of
**indices** into the fixed-order list (no shuffle of the list before splitting).
`sample_ratio` is the fraction of all pairs used as **validation** (match flag `--sample-ratio`).
"""

from __future__ import annotations

import random
from typing import Sequence


def split_train_valid_indices(all_pairs: Sequence, sample_ratio: float, sample_seed: int | None):
    """
    Returns:
      train_pairs, valid_pairs, train_gidx, valid_gidx
    Global indices refer to positions in the input `all_pairs` list (0..n-1).
    """
    all_pairs = list(all_pairs)
    if sample_seed is not None:
        random.seed(int(sample_seed))
    n = len(all_pairs)
    k = max(1, int(n * float(sample_ratio)))
    valid_idx = random.sample(range(n), k)
    valid_set = set(valid_idx)
    train_pairs = [all_pairs[i] for i in range(n) if i not in valid_set]
    valid_pairs = [all_pairs[i] for i in valid_idx]
    train_gidx = [i for i in range(n) if i not in valid_set]
    valid_gidx = list(valid_idx)
    return train_pairs, valid_pairs, train_gidx, valid_gidx
