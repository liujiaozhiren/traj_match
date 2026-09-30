"""
Validation-time noise vs label (L_A / L_B) blending.

Pair layout: ``(mid, A, B, L_A, L_B, F_A, F_B)``.  Pre window mirrors ``F_A[:len(A)]``;
post window mirrors ``F_B[-len(B):]`` (same as ``fuse_b[-len(B):]`` in v1 construction).
Six linear α levels: 1 → full noisy pipeline, 0 → label-only lon/lat (no Gaussian noise on blend).

**Timestamps:** v2 max-DTW often replaces ``A``/``B`` (and ``F_*``) without rewriting ``L_*``,
so ``A[i].t`` may differ from ``L_A[i].t`` while still matching ``F_A[i].t``. By default we
only assert **A↔F_A** and **B↔F_B**; set ``check_label_ts_match=True`` to also require
``L_A``/``L_B`` timestamps to match ``A``/``B``.
"""

from __future__ import annotations

from typing import Sequence

# Level 1 … 0: linear α for validation sweeps
NOISE_BLEND_LEVEL_ALPHAS: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)


def _ts(p: Sequence[float]) -> float:
    return float(p[0])


def assert_pair_ts_labelfuse_aligned(item: tuple, *, ts_atol: float = 0.0, check_label_ts_match: bool = False) -> None:
    """
    Always: ``A[i].t`` matches ``F_A[i].t``; ``B[j].t`` matches ``F_B`` tail slice.

    If ``check_label_ts_match`` is True, also require ``L_A[i].t`` / ``L_B`` tail to match
    ``A`` / ``B`` (holds for v1-style pairs; often **fails** on v2 max-DTW pickles).
    """
    if not isinstance(item, (list, tuple)) or len(item) != 7:
        raise ValueError(f"expected 7-tuple pair item, got {type(item).__name__} len={len(item) if isinstance(item, (list, tuple)) else 'n/a'}")
    _mid, A, B, L_A, L_B, F_A, F_B = item
    n_a, n_b = len(A), len(B)
    if n_a <= 0 or n_b <= 0:
        raise ValueError(f"empty A or B: n_a={n_a} n_b={n_b}")
    if len(L_A) < n_a or len(L_B) < n_b:
        raise ValueError(f"L_A/L_B shorter than A/B: len(L_A)={len(L_A)} len(L_B)={len(L_B)} n_a={n_a} n_b={n_b}")
    if len(F_A) < n_a or len(F_B) < n_b:
        raise ValueError(f"F_A/F_B shorter than A/B: len(F_A)={len(F_A)} len(F_B)={len(F_B)} n_a={n_a} n_b={n_b}")

    def _close(a: float, b: float) -> bool:
        return abs(a - b) <= ts_atol

    for i in range(n_a):
        ta, tf = _ts(A[i]), _ts(F_A[i])
        if not _close(ta, tf):
            raise AssertionError(f"timestamp mismatch pre i={i}: A={ta} F_A={tf} (atol={ts_atol})")
        if check_label_ts_match:
            tl = _ts(L_A[i])
            if not _close(ta, tl):
                raise AssertionError(f"timestamp mismatch pre i={i}: A={ta} L_A={tl} (atol={ts_atol})")

    off_b = len(L_B) - n_b
    off_f = len(F_B) - n_b
    for j in range(n_b):
        tb, tf = _ts(B[j]), _ts(F_B[off_f + j])
        if not _close(tb, tf):
            raise AssertionError(f"timestamp mismatch post j={j}: B={tb} F_B={tf} (atol={ts_atol})")
        if check_label_ts_match:
            tl = _ts(L_B[off_b + j])
            if not _close(tb, tl):
                raise AssertionError(f"timestamp mismatch post j={j}: B={tb} L_B={tl} (atol={ts_atol})")
