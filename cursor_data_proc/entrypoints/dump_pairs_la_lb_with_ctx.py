"""
Dump per-pair label windows (L_A, L_B) plus context points:
  - preLA:  5 points immediately BEFORE L_A[0]
  - postLB: 5 points immediately AFTER  L_B[-1]

Padding / extrapolation rule:
  - Always output lists of length 5.
  - If missing (out of bounds), extrapolate using:
    * preLA: use (L_A[0], L_A[1]) step and extrapolate backwards.
    * postLB: use (L_B[-2], L_B[-1]) step and extrapolate forwards.

All returned points are in the same normalized frame as cursor_data_proc.proc.clean_pairs:
  t'   = t - mid_t + 0
  lon' = lon - mid_lon + default_shenzhen_lon
  lat' = lat - mid_lat + default_shenzhen_lat

Output format: a list of dicts, one per pair:
  {
    "L_A":    list[[t,lon,lat]] length 12,
    "L_B":    list[[t,lon,lat]] length 12,
    "preLA":  list[[t,lon,lat]] length 5,
    "postLB": list[[t,lon,lat]] length 5
  }
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

from cursor_data_proc import cfg  # noqa: E402
from cursor_data_proc import proc  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "cursor_data_proc" / "out" / "pairs_la_lb_ctx.pkl"),
        help="Output pickle path.",
    )
    p.add_argument(
        "--max-raw-trajs",
        type=int,
        default=None,
        help="Optional: only use first N raw single trajectories (debug).",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional: only dump first N pairs (debug).",
    )
    p.add_argument("--avg-length", type=int, default=11)
    p.add_argument("--overlap", type=int, default=2)
    p.add_argument("--ctx", type=int, default=5, help="Context points on each side (default 5).")
    return p.parse_args()


def _pad_left(xs: list, n: int) -> list:
    if len(xs) >= n:
        return xs[-n:]
    return [None] * (n - len(xs)) + xs


def _pad_right(xs: list, n: int) -> list:
    if len(xs) >= n:
        return xs[:n]
    return xs + [None] * (n - len(xs))


def _extrap_preLA(pre_norm: list, L_A_norm: list) -> list:
    """
    pre_norm: list of length ctx_n containing [t,lon,lat] or None, left-padded.
    L_A_norm: normalized L_A (length 12), each point [t,lon,lat].
    Fill None by extrapolating backwards from L_A[0], L_A[1].
    """
    p0 = L_A_norm[0]
    p1 = L_A_norm[1]
    dt = float(p1[0]) - float(p0[0])
    dlon = float(p1[1]) - float(p0[1])
    dlat = float(p1[2]) - float(p0[2])

    out = list(pre_norm)
    for i in range(len(out) - 1, -1, -1):
        if out[i] is not None:
            continue
        # distance in steps from L_A[0]: for ctx_n=5 => i=4 is 1 step before, i=0 is 5 steps before
        steps = len(out) - i
        out[i] = [float(p0[0]) - dt * steps, float(p0[1]) - dlon * steps, float(p0[2]) - dlat * steps]
    return out


def _extrap_postLB(post_norm: list, L_B_norm: list) -> list:
    """
    post_norm: list of length ctx_n containing [t,lon,lat] or None, right-padded.
    L_B_norm: normalized L_B (length 12), each point [t,lon,lat].
    Fill None by extrapolating forwards from L_B[-2], L_B[-1].
    """
    p0 = L_B_norm[-2]
    p1 = L_B_norm[-1]
    dt = float(p1[0]) - float(p0[0])
    dlon = float(p1[1]) - float(p0[1])
    dlat = float(p1[2]) - float(p0[2])

    out = list(post_norm)
    for i in range(len(out)):
        if out[i] is not None:
            continue
        steps = i + 1  # 1 step after L_B[-1], 2 steps after, ...
        out[i] = [float(p1[0]) + dt * steps, float(p1[1]) + dlon * steps, float(p1[2]) + dlat * steps]
    return out

def _keep_t_lon_lat(pt) -> list[float]:
    # raw pt is length>=3; keep [t, lon, lat]
    return [pt[0], pt[1], pt[2]]


def _norm_pts(pts: list, mid4: list[float]) -> list:
    # Normalize using the same proc._normalize_traj used by clean_pairs.
    return proc._normalize_traj([_keep_t_lon_lat(p) for p in pts], mid4)  # type: ignore[attr-defined]


def _norm_pt_or_none(pt, mid4: list[float]):
    if pt is None:
        return None
    return _norm_pts([pt], mid4)[0]


def main() -> None:
    args = _parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    if args.max_raw_trajs is not None:
        single_traj_cmb = single_traj_cmb[: int(args.max_raw_trajs)]

    stdevs = proc._get_stdevs(single_traj_cmb)  # type: ignore[attr-defined]
    avg_length = int(args.avg_length)
    overlap = int(args.overlap)
    assert overlap % 2 == 0
    r = overlap // 2
    ctx_n = int(args.ctx)

    out = []

    for traj in single_traj_cmb:
        n = len(traj)
        core_lens = proc._avg_dif_length(n, avg_length)  # type: ignore[attr-defined]
        if not core_lens:
            continue

        core_starts, core_ends = [0], []
        for ln in core_lens:
            core_ends.append(core_starts[-1] + ln)
            core_starts.append(core_ends[-1])
        core_starts.pop()

        for i in range(len(core_ends) - 1):
            k = core_ends[i]
            a_start = core_starts[i]
            a_end = k - r
            b_start = k + r
            b_end = core_ends[i + 1]

            # labels (NO noise applied to these slices)
            label_a = copy.deepcopy(traj[a_start : a_end + 2 * r])
            label_b = copy.deepcopy(traj[b_start - 2 * r : b_end])
            if not (len(label_a) == 12 and len(label_b) == 12):
                continue

            # keep same eligibility filter as proc.fix_overlap_length_single_traj_split
            sub_a_len = len(traj[a_start:a_end])
            sub_b_len = len(traj[b_start:b_end])
            if not (10 <= sub_a_len <= 40 and 10 <= sub_b_len <= 40):
                continue

            # context points from raw traj around labels
            pre_raw = copy.deepcopy(traj[max(0, a_start - ctx_n) : a_start])
            post_raw = copy.deepcopy(traj[b_end : min(n, b_end + ctx_n)])
            pre_raw = _pad_left(pre_raw, ctx_n)
            post_raw = _pad_right(post_raw, ctx_n)

            # normalize into pair frame (same as clean_pairs)
            mid_point = copy.deepcopy(traj[k])
            mid4 = _keep_t_lon_lat(mid_point)
            L_A_n = _norm_pts(label_a, mid4)
            L_B_n = _norm_pts(label_b, mid4)
            pre_n = [_norm_pt_or_none(p, mid4) for p in pre_raw]
            post_n = [_norm_pt_or_none(p, mid4) for p in post_raw]

            # Extrapolate missing context points (instead of None padding).
            pre_n = _extrap_preLA(pre_n, L_A_n)
            post_n = _extrap_postLB(post_n, L_B_n)

            out.append({"L_A": L_A_n, "L_B": L_B_n, "preLA": pre_n, "postLB": post_n})
            if args.max_pairs is not None and len(out) >= int(args.max_pairs):
                pickle.dump(out, open(out_path, "wb"))
                print(f"[dump_la_lb_ctx] saved (early) -> {out_path} n={len(out)}", flush=True)
                return

    pickle.dump(out, open(out_path, "wb"))
    print(f"[dump_la_lb_ctx] cfg.con_file={cfg.con_file}", flush=True)
    print(f"[dump_la_lb_ctx] saved -> {out_path}", flush=True)
    print(f"[dump_la_lb_ctx] num_pairs={len(out)}", flush=True)


if __name__ == "__main__":
    main()

