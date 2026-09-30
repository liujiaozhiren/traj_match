"""Orchestrator: build the pool, run both score workers, pick the case-study
anchors for each direction, and dump everything needed for plotting.

Runs the two stacks as SEPARATE subprocesses (incompatible top-level ``cfg``).
This script itself only uses numpy + common (no cfg).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys

import numpy as np

import registry
from common import load_pairs, pool_global_indices, pool_smooth_indices, post_lonlat, pre_lonlat

# Display order for the figure grid. mod4 is the hero (last).
METHOD_ORDER = [
    "dtw", "hausdorff", "frechet", "sspd",
    "t3s", "traj2simvec", "neutraj", "trajgat", "st2vec", "trajcl",
    "dan", "attnmove", "mod4",
]
HERO = "mod4"
TOPK = 5


def _run(cmd: list[str]):
    print("[compute] $", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=os.path.dirname(os.path.abspath(__file__)))


def _ranking(score_row: np.ndarray) -> np.ndarray:
    """Indices sorted by score descending (best first)."""
    return np.argsort(-score_row, kind="stable")


def _select_anchor(scores: dict[str, np.ndarray], k: int, transpose: bool):
    """Pick anchor q so mod4 stands out:
      1. mod4 top1 == q                              (mod4 is correct)
      2. maximize #other methods covering q in top5  (example is not trivially hard)
      3. minimize #other methods that also top1-hit  (mod4 is uniquely best at rank1)
      4. larger mod4 score margin
    """
    def row(method, q):
        S = scores[method]
        return (S.T if transpose else S)[q]

    others = [m for m in scores if m != HERO and m != "mod4_vanilla"]
    best = None
    for q in range(k):
        hero_rank = _ranking(row(HERO, q)).tolist().index(q)
        hero_hit = int(hero_rank == 0)
        cover_near = others_top1 = 0
        for m in others:
            r = _ranking(row(m, q)).tolist().index(q)
            cover_near += int(1 <= r < TOPK)   # GT in top2..5 (close, but not rank1)
            others_top1 += int(r == 0)
        srt = np.sort(row(HERO, q))[::-1]
        margin = float(srt[0] - srt[1]) if len(srt) > 1 else 0.0
        # mod4 correct -> many others get GT into top2..5 (close) -> few others top1
        # (mod4 uniquely best) -> larger mod4 margin.
        key = (hero_hit, cover_near, -others_top1, margin)
        if best is None or key > best[0]:
            best = (key, q, hero_rank)
    return best[1], best[2]  # q, hero_rank


def _topk_record(scores: dict[str, np.ndarray], q: int, transpose: bool, k: int):
    """Per-method top5 record for anchor q."""
    rec = {}
    for m, S in scores.items():
        row = (S.T if transpose else S)[q]
        order = _ranking(row)
        gt_rank = int(order.tolist().index(q))
        top5 = order[:TOPK].tolist()
        rec[m] = {
            "top5_idx": top5,                       # pool positions, best first
            "top5_scores": [float(row[i]) for i in top5],
            "gt_rank": gt_rank,                      # 0 = top1 hit
            "gt_in_top5": bool(gt_rank < TOPK),
        }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--pool-seed", type=int, default=13)
    ap.add_argument("--pool-mode", choices=["smooth", "random"], default="smooth",
                    help="smooth: cleanest-looking (straightest) held-out pairs; random: uniform.")
    ap.add_argument("--pinned-case", default="out/selected_pairs.json",
                    help="If the file exists, reuse its exact pool_gidx and force the two "
                         "recorded anchors (reproduces the chosen figure). Use --no-pin to ignore.")
    ap.add_argument("--no-pin", action="store_true", help="Ignore --pinned-case; re-select fresh.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baseline-methods", default="dtw,hausdorff,frechet,sspd,t3s,traj2simvec,neutraj,trajgat,st2vec,trajcl,dan,attnmove")
    ap.add_argument("--skip-workers", action="store_true", help="reuse existing out/scores_*.npz")
    ap.add_argument("--out", default="out/case_study.pkl")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "out")
    os.makedirs(out_dir, exist_ok=True)
    pool_json = os.path.join(out_dir, "pool.json")

    # 1) pool
    all_pairs = load_pairs(registry.PAIRS_PKL)
    n = len(all_pairs)
    pinned = None
    pin_path = os.path.join(here, args.pinned_case)
    if not args.no_pin and os.path.exists(pin_path):
        pinned = json.load(open(pin_path))
        pool_gidx = list(pinned["pool_gidx"])
        print(f"[compute] PINNED case from {args.pinned_case}: pool_size={len(pool_gidx)}", flush=True)
    elif args.pool_mode == "smooth":
        pool_gidx = pool_smooth_indices(
            all_pairs, frac=args.frac, sample_ratio=registry.SAMPLE_RATIO,
            sample_seed=registry.SAMPLE_SEED, pool_seed=args.pool_seed,
        )
    else:
        pool_gidx = pool_global_indices(
            n, frac=args.frac, sample_ratio=registry.SAMPLE_RATIO,
            sample_seed=registry.SAMPLE_SEED, pool_seed=args.pool_seed,
        )
    json.dump({"pool_gidx": pool_gidx, "n_pairs": n, "frac": args.frac,
               "pool_seed": args.pool_seed, "pool_mode": args.pool_mode},
              open(pool_json, "w"), indent=2)
    k = len(pool_gidx)
    print(f"[compute] pool size={k} (frac={args.frac}, mode={args.pool_mode})", flush=True)

    # 2) workers
    sb = os.path.join(out_dir, "scores_baseline.npz")
    sm = os.path.join(out_dir, "scores_mod4.npz")
    if not args.skip_workers:
        _run([sys.executable, "worker_baseline.py", "--pool-json", pool_json, "--out", sb,
              "--device", args.device, "--methods", args.baseline_methods])
        _run([sys.executable, "worker_mod4.py", "--pool-json", pool_json, "--out", sm,
              "--device", args.device, "--modes", "rl_selector,match_vanilla"])

    # 3) merge score matrices
    scores: dict[str, np.ndarray] = {}
    for f in (sb, sm):
        d = np.load(f)
        for key in d.files:
            scores[key] = d[key].astype(np.float64)
    print(f"[compute] methods with scores: {sorted(scores)}", flush=True)

    # 4) clean coords of every pool pair (for plotting)
    pool_pre = [pre_lonlat(all_pairs[g]) for g in pool_gidx]
    pool_post = [post_lonlat(all_pairs[g]) for g in pool_gidx]

    # 5) per direction: select anchor + per-method top5
    result = {
        "meta": {
            "frac": args.frac, "pool_seed": args.pool_seed, "pool_size": k,
            "sample_ratio": registry.SAMPLE_RATIO, "sample_seed": registry.SAMPLE_SEED,
            "topk": TOPK, "method_order": METHOD_ORDER, "hero": HERO,
            "pairs_pkl": str(registry.PAIRS_PKL),
        },
        "pool_gidx": pool_gidx,
        "pool_pre_lonlat": pool_pre,
        "pool_post_lonlat": pool_post,
        "scores": {m: scores[m] for m in scores},
        "directions": {},
    }
    for name, transpose in (("pre_anchor", False), ("post_anchor", True)):
        if pinned is not None:
            g = int(pinned["directions"][name]["global_pair_idx"])
            q = pool_gidx.index(g)
            hero_rank = _ranking((scores[HERO].T if transpose else scores[HERO])[q]).tolist().index(q)
        else:
            q, hero_rank = _select_anchor(scores, k, transpose)
        rec = _topk_record(scores, q, transpose, k)
        cover = sum(1 for m, r in rec.items() if m not in (HERO, "mod4_vanilla") and r["gt_in_top5"])
        result["directions"][name] = {
            "anchor_pool_idx": int(q),
            "anchor_gidx": int(pool_gidx[q]),
            "hero_rank": int(hero_rank),
            "n_other_cover_top5": int(cover),
            "per_method": rec,
        }
        print(f"[compute] {name}: anchor pool#{q} (gidx={pool_gidx[q]}) "
              f"mod4_rank={hero_rank} others_cover_top5={cover}", flush=True)

    with open(os.path.join(here, args.out), "wb") as f:
        pickle.dump(result, f)
    print(f"[compute] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
