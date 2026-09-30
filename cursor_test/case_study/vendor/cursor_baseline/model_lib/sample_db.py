"""Vendored from raw_data_proc.search (sample_db only)."""

from __future__ import annotations

import random
from typing import List, Tuple


def sample_db(pairs: List[Tuple], ratio: float, seed: int | None = None) -> List[Tuple]:
    if seed is not None:
        random.seed(seed)
    n = max(1, int(len(pairs) * ratio))
    return random.sample(pairs, n)
