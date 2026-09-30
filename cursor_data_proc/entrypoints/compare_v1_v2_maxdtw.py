"""
Compare `all_pairs` v1 vs v2 (maxdtw) and verify which segments are bitwise-identical
(float equality via numpy.allclose).

`gen_noisy_pairs_v2` (maxdtw) is built from a deep copy of v1, then it only replaces:
  - F_A[8:12]  (4 points)
  - F_B[0:4]  (4 points)

Everything else in v1, including L_A and L_B, is carried over unchanged in the output tuple
(L_A, L_B are the same object sequence as in v1).

Consequences (len F_A = len F_B = 12, len A = len B = 10 in our dumps):
  - F_A[0:8] should match v1
  - F_B[4:12] should match v1
  - A = F_A[0:10]  => indices 8..9 come from replaced F_A[8..9]  => A[0:7] and usually A[0:8]
     match v1 (here A[0:8] == v1 A[0:8] because only A[8].. are affected, and 8<8 is false; see note below)
  - A[8:10] should *usually* differ (unless a candidate pick coincidentally matches the old values)
  - B = F_B[2:12] (last 10) => v2's replacement of F_B[0:4] affects the first 2 B points: B0=F_B[2], B1=F_B[3]
  - B[2:8] (and generally B[2:10] tail) should still match v1, because the replaced region does not go past F_B[3]
     for a 10-point B window? Actually B also uses F_B[4:11] for middle points; v2 only touches 0:4,
     so B[2:9] should be identical; we check B[2:8] as a conservative middle slice.

The script reports:
  - #pairs where a slice mismatches
  - #pairs where A[8:10] differs
  - #pairs where B[0:2] differs
  - a few first mismatching examples
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Any, Iterable

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--v1", type=str, default="cursor_mod_4/pipeline/all_pairs_v1.pkl")
    p.add_argument("--v2", type=str, default="cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl")
    p.add_argument("--rtol", type=float, default=0.0)
    p.add_argument("--atol", type=float, default=0.0)
    p.add_argument("--max-examples", type=int, default=3)
    return p.parse_args()


def _as_float(x: Any) -> float:
    return float(x)


def _pt_eq(a: Iterable[Any], b: Iterable[Any], *, rtol: float, atol: float) -> bool:
    a = list(a)
    b = list(b)
    if len(a) != len(b):
        return False
    return bool(np.allclose([_as_float(x) for x in a], [_as_float(x) for x in b], rtol=rtol, atol=atol))


def _seq_eq(s1: list, s2: list, *, rtol: float, atol: float) -> bool:
    if len(s1) != len(s2):
        return False
    return all(_pt_eq(s1[i], s2[i], rtol=rtol, atol=atol) for i in range(len(s1)))


def _print_example_pair(k: int, v1, v2, label: str):
    m1, A1, B1, la1, lb1, fa1, fb1 = v1[k]
    m2, A2, B2, la2, lb2, fa2, fb2 = v2[k]
    print(f"[cmp] example {label} k={k}", flush=True)
    print("  v1 A tail:", A1[8:10], flush=True)
    print("  v2 A tail:", A2[8:10], flush=True)
    print("  v1 B head:", B1[0:2], flush=True)
    print("  v2 B head:", B2[0:2], flush=True)


def main() -> None:
    args = _parse_args()
    rtol, atol = float(args.rtol), float(args.atol)

    p1 = os.path.abspath(args.v1)
    p2 = os.path.abspath(args.v2)
    a = pickle.load(open(p1, "rb"))
    b = pickle.load(open(p2, "rb"))
    if len(a) != len(b):
        raise SystemExit(f"len mismatch: v1={len(a)} v2={len(b)}")

    n = len(a)
    print(f"[cmp] v1={p1}", flush=True)
    print(f"[cmp] v2={p2}", flush=True)
    print(f"[cmp] N={n} rtol={rtol} atol={atol}", flush=True)

    bad_mid = 0
    bad_la = 0
    bad_lb = 0
    bad_fa0_8 = 0
    bad_fb4_12 = 0

    bad_a0_7 = 0
    bad_a0_8 = 0
    diff_a8_10 = 0

    diff_b0_2 = 0
    bad_b2_8 = 0
    ex = 0

    for k in range(n):
        m1, A1, B1, la1, lb1, fa1, fb1 = a[k]
        m2, A2, B2, la2, lb2, fa2, fb2 = b[k]
        if m1 != m2:
            bad_mid += 1

        if not _seq_eq(la1, la2, rtol=rtol, atol=atol):
            bad_la += 1
        if not _seq_eq(lb1, lb2, rtol=rtol, atol=atol):
            bad_lb += 1

        if not _seq_eq(fa1[0:8], fa2[0:8], rtol=rtol, atol=atol):
            bad_fa0_8 += 1
        if not _seq_eq(fb1[4:12], fb2[4:12], rtol=rtol, atol=atol):
            bad_fb4_12 += 1

        if not _seq_eq(A1[0:7], A2[0:7], rtol=rtol, atol=atol):
            bad_a0_7 += 1
        if not _seq_eq(A1[0:8], A2[0:8], rtol=rtol, atol=atol):
            bad_a0_8 += 1
        if not _seq_eq(A1[8:10], A2[8:10], rtol=rtol, atol=atol):
            diff_a8_10 += 1
            if ex < int(args.max_examples):
                _print_example_pair(k, a, b, "A[8:10] differs")
                ex += 1

        if not _seq_eq(B1[0:2], B2[0:2], rtol=rtol, atol=atol):
            diff_b0_2 += 1
        if not _seq_eq(B1[2:8], B2[2:8], rtol=rtol, atol=atol):
            bad_b2_8 += 1

    print("[cmp] mismatch counts (for equality checks, 0 means perfect match for all pairs):", flush=True)
    print(f"  mid: {bad_mid}", flush=True)
    print(f"  L_A full: {bad_la}   L_B full: {bad_lb}", flush=True)
    print(f"  F_A[0:8]: {bad_fa0_8}   F_B[4:12]: {bad_fb4_12}", flush=True)
    print(f"  A[0:7] (expected unchanged by v2 tail edit): {bad_a0_7}", flush=True)
    print(f"  A[0:8] (8 points, expected unchanged here): {bad_a0_8}", flush=True)
    print(f"  A[8:10] diffs (expected if tail changed): {diff_a8_10} / {n}", flush=True)
    print(f"  B[0:2] diffs (expected from F_B[0:4] edit): {diff_b0_2} / {n}", flush=True)
    print(f"  B[2:8] (should match v1): {bad_b2_8}", flush=True)

    if bad_mid == 0 and bad_la == 0 and bad_lb == 0 and bad_fa0_8 == 0 and bad_fb4_12 == 0 and bad_a0_7 == 0 and bad_b2_8 == 0:
        print("[cmp] OK: labels + unmodified fuse halves + stable A prefix + stable B mid match v1.", flush=True)


if __name__ == "__main__":
    main()
