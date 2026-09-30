"""
Augment v2 maxdtw pairs using preLA/postLB context.

Inputs:
  - v2 pairs: cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl (7-tuple list)
      (mid, A, B, L_A, L_B, F_A, F_B)
      A,B are length-10 sequences of [t,lon,lat]
  - ctx file: cursor_data_proc/out/pairs_la_lb_ctx_extrap.pkl
      list of dicts aligned with v2 order:
        {"L_A":..., "L_B":..., "preLA":(len5), "postLB":(len5)}

We generate 6 augmented datasets (gap=1..6) with the following rules:

  A_k: prepend last k points from preLA to A, then drop last k points of A (keep length 10)
  B_k: append first k points from postLB to B, then drop first k points of B (keep length 10)

Mapping:
  gap=1 => A1
  gap=2 => B1
  gap=3 => A2 + B1
  gap=4 => A2 + B2
  gap=5 => A3 + B2
  gap=6 => A3 + B3

We also keep internal consistency with F_A/F_B:
  - If A is modified: overwrite F_A[:10] with new A (keep tail points unchanged).
  - If B is modified: overwrite F_B[-10:] with new B (keep head points unchanged).

Outputs (default directory cursor_data_proc/out):
  - all_pairs_v2_maxdtw_gap1.pkl
  - ...
  - all_pairs_v2_maxdtw_gap6.pkl
"""

from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_rp = str(_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v2",
        type=str,
        default=str(_ROOT / "cursor_mod_4" / "pipeline" / "all_pairs_v2_maxdtw.pkl"),
        help="Input v2 maxdtw pickle (7-tuple list).",
    )
    p.add_argument(
        "--ctx",
        type=str,
        default=str(_ROOT / "cursor_data_proc" / "out" / "pairs_la_lb_ctx_extrap.pkl"),
        help="Context pickle aligned with v2 order (preLA/postLB).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT / "cursor_data_proc" / "out"),
        help="Directory to write augmented pickles.",
    )
    return p.parse_args()


def _a_k(A: list, preLA: list, k: int) -> list:
    if k <= 0:
        return list(A)
    assert len(A) == 10
    assert len(preLA) >= k
    return list(preLA[-k:]) + list(A[: 10 - k])


def _b_k(B: list, postLB: list, k: int) -> list:
    if k <= 0:
        return list(B)
    assert len(B) == 10
    assert len(postLB) >= k
    return list(B[k:]) + list(postLB[:k])


_GAP_TO_K = {
    1: (1, 0),
    2: (0, 1),
    3: (2, 1),
    4: (2, 2),
    5: (3, 2),
    6: (3, 3),
}


def _apply_gap(item: tuple, ctx_item: dict, gap: int) -> tuple:
    mid, A, B, L_A, L_B, F_A, F_B = item
    preLA = ctx_item["preLA"]
    postLB = ctx_item["postLB"]

    ak, bk = _GAP_TO_K[int(gap)]
    A2 = _a_k(A, preLA, ak)
    B2 = _b_k(B, postLB, bk)

    FA2 = copy.deepcopy(F_A)
    FB2 = copy.deepcopy(F_B)
    if ak > 0:
        FA2[:10] = copy.deepcopy(A2)
    if bk > 0:
        FB2[-10:] = copy.deepcopy(B2)

    return (mid, A2, B2, L_A, L_B, FA2, FB2)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    v2 = pickle.load(open(args.v2, "rb"))
    ctx = pickle.load(open(args.ctx, "rb"))
    if len(v2) != len(ctx):
        raise SystemExit(f"len mismatch: v2={len(v2)} ctx={len(ctx)}")

    for gap in range(1, 7):
        out = [_apply_gap(v2[i], ctx[i], gap) for i in range(len(v2))]
        out_path = out_dir / f"all_pairs_v2_maxdtw_gap{gap}.pkl"
        pickle.dump(out, open(out_path, "wb"))
        # minimal sanity print
        mid, A, B, L_A, L_B, F_A, F_B = out[0]
        print(
            f"[gap{gap}] saved -> {out_path} n={len(out)} sample0 lens: "
            f"A={len(A)} B={len(B)} L_A={len(L_A)} L_B={len(L_B)} F_A={len(F_A)} F_B={len(F_B)}",
            flush=True,
        )


if __name__ == "__main__":
    main()

