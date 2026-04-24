"""
Backward-compatible re-exports; implementations live in traj_distances.py.
"""

from __future__ import annotations

from cursor_baseline.traj_distances import (
    dtw_batch_equal_len,
    pairwise_neg_distance_matrix,
    topk_hit_rate,
)


def pairwise_neg_dtw_matrix(pre_xy, post_xy, *, chunk_queries=None):
    return pairwise_neg_distance_matrix(pre_xy, post_xy, "dtw", chunk_queries=chunk_queries)


__all__ = ["dtw_batch_equal_len", "pairwise_neg_dtw_matrix", "topk_hit_rate"]
