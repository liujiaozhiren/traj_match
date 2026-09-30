"""
End-to-end pipeline: trajectory list (.pkl) -> v1 pairs -> v2 maxdtw pairs.

Input: a pickle containing `List[traj]`, where each `traj` is a list of points.
Point formats supported:
  - [t, lon, lat] (timestamp present; kept but can be overwritten)
  - [lon, lat]    (timestamp missing; will be synthesized BEFORE pair cutting)

Outputs:
  - <output-dir>/all_pairs_v1.pkl
  - <output-dir>/all_pairs_v1_top10k_overlap.pkl
  - <output-dir>/all_pairs_v2_maxdtw.pkl

This script matches the pair semantics used by cursor_mod_4:
  (mid, A10, B10, L_A, L_B, F_A, F_B)
and matches cursor_mod_4.proc.clean_pairs normalization by shifting each pair to the
Shenzhen anchor using its own mid-point.
"""

from __future__ import annotations

import argparse
import math
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Tuple

DEFAULT_TS0 = 1476400000
DEFAULT_GAP_S = 3
DEFAULT_SZ_LON = 114.057868
DEFAULT_SZ_LAT = 22.543099


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        default="/home/jh/traj_data/chengdu/trajs_10000.pkl",
        help="Pickle path containing a list of trajectories.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="cursor_mod_run_chengdu/data",
        help="Directory to write v1/v2 pair pickles.",
    )
    p.add_argument("--avg-length", type=int, default=11, help="Core segment length for splitting.")
    p.add_argument("--overlap", type=int, default=2, help="Even overlap; r=overlap//2.")
    p.add_argument("--min-seg", type=int, default=10, help="Min sub segment length constraint.")
    p.add_argument("--max-seg", type=int, default=40, help="Max sub segment length constraint.")
    p.add_argument("--ts0", type=int, default=DEFAULT_TS0, help="Synthesized start timestamp (seconds).")
    p.add_argument("--gap-s", type=int, default=DEFAULT_GAP_S, help="Synthesized timestamp gap (seconds).")
    p.add_argument(
        "--no-force-synth-ts",
        action="store_true",
        help="Disable timestamp overwrite (debug only). Default behavior overwrites.",
    )
    p.add_argument(
        "--no-drop-stationary-pairs",
        action="store_true",
        help="Disable stationary-step filtering inside each pair (debug only).",
    )
    p.add_argument(
        "--topk-pairs",
        type=int,
        default=10_000,
        help="After scoring pre/post overlap, keep this many pairs (smallest score).",
    )
    p.add_argument(
        "--K",
        type=int,
        default=10,
        help="Minimum index gap treated as 'time-far' for cross pre/post overlap scoring.",
    )
    p.add_argument(
        "--d0-mult",
        type=float,
        default=5.0,
        help="Distance threshold multiplier: d0 = d0_mult * avg_step_m.",
    )
    p.add_argument("--skip-v2", action="store_true", help="Only build and filter v1 pairs; do not generate v2 maxdtw.")
    p.add_argument("--prefilter-m", type=int, default=64)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--max-pairs", type=int, default=None, help="Debug: only process first N pairs for v2.")
    return p.parse_args()


def _force_synth_t_lon_lat(traj: list, *, ts0: int, gap_s: int) -> list[list[float]]:
    """Overwrite / synthesize timestamps; output points as [t, lon, lat]."""
    if not traj:
        return []
    out = []
    t = int(ts0)
    for p in traj:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        lon = float(p[1]) if len(p) >= 3 else float(p[0])
        lat = float(p[2]) if len(p) >= 3 else float(p[1])
        out.append([float(t), lon, lat])
        t += int(gap_s)
    return out


def _has_stationary_step(seq_tll: list[list[float]]) -> bool:
    """True if any consecutive points have identical lon/lat."""
    for i in range(1, len(seq_tll)):
        if float(seq_tll[i][1]) == float(seq_tll[i - 1][1]) and float(seq_tll[i][2]) == float(seq_tll[i - 1][2]):
            return True
    return False


def _normalize_to_shenzhen(seq: list[list[float]], mid: list[float]) -> list[list[float]]:
    """
    Match cursor_mod_4.proc.clean_pairs normalization:
      t'   = t - mid_t + 0
      lon' = lon - mid_lon + DEFAULT_SZ_LON
      lat' = lat - mid_lat + DEFAULT_SZ_LAT
    """
    mt, mlon, mlat = float(mid[0]), float(mid[1]), float(mid[2])
    return [[float(p[0]) - mt, float(p[1]) - mlon + DEFAULT_SZ_LON, float(p[2]) - mlat + DEFAULT_SZ_LAT] for p in seq]


_EARTH_R_M = 6371000.0


def _dist_m_equirect(p: list[float], q: list[float]) -> float:
    """Fast meter distance for nearby lon/lat degrees. p,q are [t, lon, lat]."""
    lon1 = float(p[1])
    lat1 = float(p[2])
    lon2 = float(q[1])
    lat2 = float(q[2])
    lat0 = math.radians((lat1 + lat2) * 0.5)
    dx = math.radians(lon2 - lon1) * _EARTH_R_M * math.cos(lat0)
    dy = math.radians(lat2 - lat1) * _EARTH_R_M
    return (dx * dx + dy * dy) ** 0.5


def _ctx22_from_v1_item(item: tuple) -> list[list[float]]:
    """A10(10) + overlap2(2) + B10(10); overlap2 = L_A[-2:]."""
    _mid, A10, B10, L_A, _L_B, _F_A, _F_B = item
    ov2 = list(L_A[-2:])
    return list(A10) + ov2 + list(B10)


def _avg_step_m_over_pairs(v1_pairs: list[tuple]) -> float:
    total = 0.0
    cnt = 0
    for it in v1_pairs:
        ctx = _ctx22_from_v1_item(it)
        for i in range(21):
            total += _dist_m_equirect(ctx[i], ctx[i + 1])
            cnt += 1
    return total / max(1, cnt)


def _overlap_violation_score(ctx22: list[list[float]], *, K: int, d0_m: float) -> int:
    """
    Cross pre/post overlap score:
      i in pre indices 0..9, j in post indices 12..21
      count if (j-i)>=K and dist(i,j)<d0_m.
    """
    viol = 0
    for i in range(10):
        for j in range(12, 22):
            if (j - i) < int(K):
                continue
            if _dist_m_equirect(ctx22[i], ctx22[j]) < float(d0_m):
                viol += 1
    return int(viol)


def _split_one_traj_to_v1_pairs(
    traj_tll: list[list[float]],
    *,
    avg_length: int,
    overlap: int,
    min_seg: int,
    max_seg: int,
    drop_stationary_pairs: bool,
) -> list[tuple]:
    """Build v1-style 7-tuples (already normalized to Shenzhen frame)."""
    n = len(traj_tll)
    if n < avg_length * 2:
        return []
    if overlap % 2 != 0:
        raise ValueError("overlap must be even")
    num_segments = n // int(avg_length)
    if num_segments <= 1:
        return []
    core_lens = [int(avg_length)] * num_segments
    core_starts, core_ends = [0], []
    for ln in core_lens:
        core_ends.append(core_starts[-1] + ln)
        core_starts.append(core_ends[-1])
    core_starts.pop()

    r = overlap // 2
    out: list[tuple] = []
    for i in range(len(core_ends) - 1):
        k = core_ends[i]
        a_start = core_starts[i]
        a_end = k - r
        b_start = k + r
        b_end = core_ends[i + 1]

        sub_a = traj_tll[a_start:a_end]
        sub_b = traj_tll[b_start:b_end]
        if not (min_seg <= len(sub_a) <= max_seg and min_seg <= len(sub_b) <= max_seg):
            continue

        fuse_a = traj_tll[a_start : a_end + 2 * r]
        fuse_b = traj_tll[b_start - 2 * r : b_end]
        label_a = traj_tll[a_start : a_end + 2 * r]
        label_b = traj_tll[b_start - 2 * r : b_end]
        if not (len(label_a) == 12 and len(label_b) == 12 and len(fuse_a) == 12 and len(fuse_b) == 12):
            continue

        if drop_stationary_pairs:
            overlap_pts = traj_tll[k - r : k + r]  # len = overlap
            ctx22 = fuse_a[:10] + overlap_pts + fuse_b[-10:]
            if _has_stationary_step(ctx22):
                continue

        mid = traj_tll[k]
        mid_n = _normalize_to_shenzhen([mid], mid)[0]
        A10 = _normalize_to_shenzhen(fuse_a[:10], mid)
        B10 = _normalize_to_shenzhen(fuse_b[-10:], mid)
        L_A = _normalize_to_shenzhen(label_a, mid)
        L_B = _normalize_to_shenzhen(label_b, mid)
        F_A = _normalize_to_shenzhen(fuse_a, mid)
        F_B = _normalize_to_shenzhen(fuse_b, mid)
        out.append((mid_n, A10, B10, L_A, L_B, F_A, F_B))
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v1_path = out_dir / "all_pairs_v1.pkl"
    v1_top_path = out_dir / "all_pairs_v1_top10k_overlap.pkl"
    v2_path = out_dir / "all_pairs_v2_maxdtw.pkl"

    trajs = pickle.load(open(in_path, "rb"))
    if not isinstance(trajs, list):
        raise TypeError(f"expected list of trajs, got {type(trajs).__name__}")

    v1_pairs: list[tuple] = []
    drop_stationary_pairs = not bool(args.no_drop_stationary_pairs)
    force_ts = not bool(args.no_force_synth_ts)
    if not force_ts:
        print("[chengdu_pipeline] WARNING: --no-force-synth-ts set; keeping original timestamps if present.", flush=True)
    for traj in trajs:
        traj3 = _force_synth_t_lon_lat(traj, ts0=int(args.ts0), gap_s=int(args.gap_s)) if force_ts else _force_synth_t_lon_lat(traj, ts0=int(args.ts0), gap_s=int(args.gap_s))
        v1_pairs.extend(
            _split_one_traj_to_v1_pairs(
                traj3,
                avg_length=int(args.avg_length),
                overlap=int(args.overlap),
                min_seg=int(args.min_seg),
                max_seg=int(args.max_seg),
                drop_stationary_pairs=drop_stationary_pairs,
            )
        )

    pickle.dump(v1_pairs, open(v1_path, "wb"))
    print(
        f"[chengdu_pipeline] input_trajs={len(trajs)} force_synth_ts={force_ts} drop_stationary_pairs={drop_stationary_pairs}",
        flush=True,
    )
    print(f"[chengdu_pipeline] v1_pairs={len(v1_pairs)} -> {v1_path}", flush=True)

    # ---- rank by pre/post overlap and keep top-K ----
    avg_step_m = _avg_step_m_over_pairs(v1_pairs)
    d0_m = float(args.d0_mult) * float(avg_step_m)
    K = int(args.K)
    keep = min(int(args.topk_pairs), len(v1_pairs))
    print(f"[chengdu_pipeline] overlap_score: avg_step_m={avg_step_m:.3f} d0_m={d0_m:.3f} K={K} keep={keep}", flush=True)

    scored: list[Tuple[int, int]] = []
    for idx, it in enumerate(v1_pairs):
        ctx = _ctx22_from_v1_item(it)
        sc = _overlap_violation_score(ctx, K=K, d0_m=d0_m)
        scored.append((int(sc), int(idx)))
    scored.sort(key=lambda x: (x[0], x[1]))

    keep_idx = set(i for _sc, i in scored[:keep])
    v1_top = [v1_pairs[i] for i in range(len(v1_pairs)) if i in keep_idx]
    pickle.dump(v1_top, open(v1_top_path, "wb"))
    print(f"[chengdu_pipeline] v1_top_pairs={len(v1_top)} -> {v1_top_path}", flush=True)

    if bool(args.skip_v2):
        print("[chengdu_pipeline] --skip-v2 set; done after v1 filtering.", flush=True)
        return

    # ---- v2 maxdtw: reuse existing script ----
    gen = Path(__file__).resolve().parent / "gen_noisy_pairs_v2.py"
    cmd = [
        sys.executable,
        str(gen),
        "--input",
        str(v1_top_path),
        "--output-dir",
        str(out_dir),
        "--schemes",
        "maxdtw",
        "--prefilter-m",
        str(int(args.prefilter_m)),
        "--topk",
        str(int(args.topk)),
        "--workers",
        str(int(args.workers)),
        "--log-every",
        "200",
    ]
    if args.max_pairs is not None:
        cmd.extend(["--max-pairs", str(int(args.max_pairs))])

    print(f"[chengdu_pipeline] running v2 maxdtw: {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd)

    v2_pairs = pickle.load(open(v2_path, "rb"))
    print(f"[chengdu_pipeline] v2_maxdtw_pairs={len(v2_pairs)} -> {v2_path}", flush=True)


if __name__ == "__main__":
    main()

def _force_synth_t_lon_lat(traj: list, *, ts0: int, gap_s: int) -> list[list[float]]:
    """
    traj3 points are [t, lon, lat] with t in seconds.
    """
    if not traj:
        return []
    out = []
    t = int(ts0)
    for p in traj:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        # Accept either [t,lon,lat] or [lon,lat]; we ALWAYS overwrite time anyway.
        lon = float(p[1]) if len(p) >= 3 else float(p[0])
        lat = float(p[2]) if len(p) >= 3 else float(p[1])
        out.append([float(t), lon, lat])
        t += int(gap_s)
    return out


def _has_stationary_step(seq_tll: list[list[float]]) -> bool:
    """True if any consecutive points have identical lon/lat."""
    for i in range(1, len(seq_tll)):
        if float(seq_tll[i][1]) == float(seq_tll[i - 1][1]) and float(seq_tll[i][2]) == float(seq_tll[i - 1][2]):
            return True
    return False


def _normalize_to_shenzhen(seq: list[list[float]], mid: list[float]) -> list[list[float]]:
    """
    Match cursor_mod_4.proc.clean_pairs normalization:
      t'   = t - mid_t + 0
      lon' = lon - mid_lon + DEFAULT_SZ_LON
      lat' = lat - mid_lat + DEFAULT_SZ_LAT
    """
    mt, mlon, mlat = float(mid[0]), float(mid[1]), float(mid[2])
    return [[float(p[0]) - mt, float(p[1]) - mlon + DEFAULT_SZ_LON, float(p[2]) - mlat + DEFAULT_SZ_LAT] for p in seq]


_EARTH_R_M = 6371000.0


def _dist_m_equirect(p: list[float], q: list[float]) -> float:
    """Fast meter distance for nearby lon/lat degrees (equirectangular). p,q are [t, lon, lat]."""
    lon1 = float(p[1])
    lat1 = float(p[2])
    lon2 = float(q[1])
    lat2 = float(q[2])
    lat0 = math.radians((lat1 + lat2) * 0.5)
    dx = math.radians(lon2 - lon1) * _EARTH_R_M * math.cos(lat0)
    dy = math.radians(lat2 - lat1) * _EARTH_R_M
    return (dx * dx + dy * dy) ** 0.5


def _ctx22_from_v1_item(item: tuple) -> list[list[float]]:
    """
    v1 tuple: (mid, A10, B10, L_A, L_B, F_A, F_B)
    ctx22: A10(10) + overlap2(2) + B10(10); overlap2 = L_A[-2:].
    """
    _mid, A10, B10, L_A, _L_B, _F_A, _F_B = item
    ov2 = list(L_A[-2:])
    return list(A10) + ov2 + list(B10)


def _avg_step_m_over_pairs(v1_pairs: list[tuple]) -> float:
    total = 0.0
    cnt = 0
    for it in v1_pairs:
        ctx = _ctx22_from_v1_item(it)
        for i in range(21):
            total += _dist_m_equirect(ctx[i], ctx[i + 1])
            cnt += 1
    return total / max(1, cnt)


def _overlap_violation_score(ctx22: list[list[float]], *, K: int, d0_m: float) -> int:
    """
    Cross pre/post overlap score:
      i in pre indices 0..9, j in post indices 12..21
      count if (j-i)>=K and dist(i,j)<d0_m.
    """
    viol = 0
    for i in range(10):
        for j in range(12, 22):
            if (j - i) < int(K):
                continue
            if _dist_m_equirect(ctx22[i], ctx22[j]) < float(d0_m):
                viol += 1
    return int(viol)

def _split_one_traj_to_v1_pairs(
    traj_tll: list[list[float]],
    *,
    avg_length: int,
    overlap: int,
    min_seg: int,
    max_seg: int,
    drop_stationary_pairs: bool,
) -> list[tuple]:
    """
    Build v1-style 7-tuples (already normalized to Shenzhen frame):
      (mid, A10, B10, L_A, L_B, F_A, F_B)
    All sequences contain points as [t, lon, lat].
    """
    n = len(traj_tll)
    if n < avg_length * 2:
        return []
    if overlap % 2 != 0:
        raise ValueError("overlap must be even")
    num_segments = n // int(avg_length)
    if num_segments <= 1:
        return []
    core_lens = [int(avg_length)] * num_segments
    core_starts, core_ends = [0], []
    for ln in core_lens:
        core_ends.append(core_starts[-1] + ln)
        core_starts.append(core_ends[-1])
    core_starts.pop()

    r = overlap // 2
    out: list[tuple] = []
    for i in range(len(core_ends) - 1):
        k = core_ends[i]
        a_start = core_starts[i]
        a_end = k - r
        b_start = k + r
        b_end = core_ends[i + 1]

        # core sub segments
        sub_a = traj_tll[a_start:a_end]
        sub_b = traj_tll[b_start:b_end]
        # fused segments and labels (both are length 12 when avg_length=11 and overlap=2)
        fuse_a = traj_tll[a_start : a_end + 2 * r]
        fuse_b = traj_tll[b_start - 2 * r : b_end]
        label_a = traj_tll[a_start : a_end + 2 * r]
        label_b = traj_tll[b_start - 2 * r : b_end]

        if not (len(label_a) == 12 and len(label_b) == 12 and len(fuse_a) == 12 and len(fuse_b) == 12):
            continue
        if not (min_seg <= len(sub_a) <= max_seg and min_seg <= len(sub_b) <= max_seg):
            continue

        mid = traj_tll[k]

        if drop_stationary_pairs:
            # Stationary-step filter on the 22-point context:
            # A10 (10 pts) + overlap (2 pts) + B10 (10 pts) = 22 pts.
            # Overlap points are the shared boundary points between label_a/label_b (length=overlap).
            overlap_pts = traj_tll[k - r : k + r]  # len = 2r == overlap
            ctx22 = fuse_a[:10] + overlap_pts + fuse_b[-10:]
            if _has_stationary_step(ctx22):
                continue

        # Normalize ALL components to Shenzhen frame using this pair's mid-point.
        mid_n = _normalize_to_shenzhen([mid], mid)[0]
        A10 = _normalize_to_shenzhen(fuse_a[:10], mid)
        B10 = _normalize_to_shenzhen(fuse_b[-10:], mid)
        L_A = _normalize_to_shenzhen(label_a, mid)
        L_B = _normalize_to_shenzhen(label_b, mid)
        F_A = _normalize_to_shenzhen(fuse_a, mid)
        F_B = _normalize_to_shenzhen(fuse_b, mid)
        out.append((mid_n, A10, B10, L_A, L_B, F_A, F_B))
    return out


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v1_path = out_dir / "all_pairs_v1.pkl"
    v1_top_path = out_dir / "all_pairs_v1_top10k_overlap.pkl"
    v2_path = out_dir / "all_pairs_v2_maxdtw.pkl"

    trajs = pickle.load(open(in_path, "rb"))
    if not isinstance(trajs, list):
        raise TypeError(f"expected list of trajs, got {type(trajs).__name__}")

    v1_pairs: list[tuple] = []
    drop_stationary_pairs = not bool(args.no_drop_stationary_pairs)
    force_ts = not bool(args.no_force_synth_ts)
    if not force_ts:
        print("[chengdu_pipeline] WARNING: --no-force-synth-ts set; keeping original timestamps if present.", flush=True)
    for traj in trajs:
        traj3 = _force_synth_t_lon_lat(traj, ts0=int(args.ts0), gap_s=int(args.gap_s)) if force_ts else _force_synth_t_lon_lat(traj, ts0=int(traj[0][0]), gap_s=int(args.gap_s))
        v1_pairs.extend(
            _split_one_traj_to_v1_pairs(
                traj3,
                avg_length=int(args.avg_length),
                overlap=int(args.overlap),
                min_seg=int(args.min_seg),
                max_seg=int(args.max_seg),
                drop_stationary_pairs=drop_stationary_pairs,
            )
        )

    pickle.dump(v1_pairs, open(v1_path, "wb"))
    print(
        f"[chengdu_pipeline] input_trajs={len(trajs)} force_synth_ts={force_ts} "
        f"drop_stationary_pairs={drop_stationary_pairs}",
        flush=True,
    )
    print(f"[chengdu_pipeline] v1_pairs={len(v1_pairs)} -> {v1_path}", flush=True)

    # ---- rank by pre/post overlap and keep top-K ----
    avg_step_m = _avg_step_m_over_pairs(v1_pairs)
    d0_m = float(args.d0_mult) * float(avg_step_m)
    K = int(args.K)
    keep = min(int(args.topk_pairs), len(v1_pairs))
    print(f"[chengdu_pipeline] overlap_score: avg_step_m={avg_step_m:.3f} d0_m={d0_m:.3f} K={K} keep={keep}", flush=True)

    scored: list[Tuple[int, int]] = []
    for idx, it in enumerate(v1_pairs):
        ctx = _ctx22_from_v1_item(it)
        sc = _overlap_violation_score(ctx, K=K, d0_m=d0_m)
        scored.append((int(sc), int(idx)))
    scored.sort(key=lambda x: (x[0], x[1]))

    keep_idx = set(i for _sc, i in scored[:keep])
    v1_top = [v1_pairs[i] for i in range(len(v1_pairs)) if i in keep_idx]
    pickle.dump(v1_top, open(v1_top_path, "wb"))
    if scored:
        def q(frac: float) -> int:
            return scored[min(len(scored) - 1, int(round(frac * (len(scored) - 1))))][0]
        print(f"[chengdu_pipeline] overlap_score_quantiles p0/p10/p50/p90={[q(0.0), q(0.1), q(0.5), q(0.9)]}", flush=True)
    print(f"[chengdu_pipeline] v1_top_pairs={len(v1_top)} -> {v1_top_path}", flush=True)

    if bool(args.skip_v2):
        print("[chengdu_pipeline] --skip-v2 set; done after v1 filtering.", flush=True)
        return

    # v2 maxdtw: reuse existing script to avoid re-implementing.
    gen = Path(__file__).resolve().parent / "gen_noisy_pairs_v2.py"
    cmd = [
        sys.executable,
        str(gen),
        "--input",
        str(v1_top_path),
        "--output-dir",
        str(out_dir),
        "--schemes",
        "maxdtw",
        "--prefilter-m",
        str(int(args.prefilter_m)),
        "--topk",
        str(int(args.topk)),
        "--workers",
        str(int(args.workers)),
        "--log-every",
        "200",
    ]
    if args.max_pairs is not None:
        cmd += ["--max-pairs", str(int(args.max_pairs))]

    print(f"[chengdu_pipeline] running v2 maxdtw: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)

    if not v2_path.exists():
        raise RuntimeError(f"expected v2 output at {v2_path} but file is missing")
    v2_pairs = pickle.load(open(v2_path, "rb"))
    print(f"[chengdu_pipeline] v2_maxdtw_pairs={len(v2_pairs)} -> {v2_path}", flush=True)


if __name__ == "__main__":
    main()

