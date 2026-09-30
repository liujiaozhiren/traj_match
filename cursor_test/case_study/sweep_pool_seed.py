"""Try several pool seeds; score each on how compelling the case study is, in
BOTH directions: mod4 must top1-hit, many others cover GT in top5, few others
top1-hit. Prints a ranking so you can pick a --pool-seed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

import registry
from common import load_pairs, pool_global_indices, pool_smooth_indices

HERO = "mod4"
TOPK = 5
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def _ranking(rowv):
    return np.argsort(-rowv, kind="stable")


def _dir_stats(scores, k, transpose):
    others = [m for m in scores if m not in (HERO, "mod4_vanilla")]

    def row(m, q):
        S = scores[m]
        return (S.T if transpose else S)[q]

    best = None
    for q in range(k):
        hrank = _ranking(row(HERO, q)).tolist().index(q)
        if hrank != 0:
            continue
        ranks = [_ranking(row(m, q)).tolist().index(q) for m in others]
        cover_near = sum(int(1 <= r < TOPK) for r in ranks)
        otop1 = sum(int(r == 0) for r in ranks)
        key = (cover_near, -otop1)
        if best is None or key > best[0]:
            best = (key, q, cover_near, otop1)
    return best  # None if mod4 never top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--pool-mode", choices=["smooth", "random"], default="smooth")
    ap.add_argument("--seeds", default="0,1,2,3,7,11,13,21,42,99,123,2024")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--baseline-methods", default="dtw,hausdorff,frechet,sspd,t3s,traj2simvec,neutraj,st2vec,dan,attnmove")
    args = ap.parse_args()

    all_pairs = load_pairs(registry.PAIRS_PKL)
    n = len(all_pairs)
    rows = []
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        if args.pool_mode == "smooth":
            gidx = pool_smooth_indices(all_pairs, frac=args.frac, sample_ratio=registry.SAMPLE_RATIO,
                                       sample_seed=registry.SAMPLE_SEED, pool_seed=seed)
        else:
            gidx = pool_global_indices(n, frac=args.frac, sample_ratio=registry.SAMPLE_RATIO,
                                       sample_seed=registry.SAMPLE_SEED, pool_seed=seed)
        pj = os.path.join(OUT, f"pool_s{seed}.json")
        json.dump({"pool_gidx": gidx}, open(pj, "w"))
        sb = os.path.join(OUT, f"scores_baseline_s{seed}.npz")
        sm = os.path.join(OUT, f"scores_mod4_s{seed}.npz")
        subprocess.run([sys.executable, "worker_baseline.py", "--pool-json", pj, "--out", sb,
                        "--device", args.device, "--methods", args.baseline_methods], check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "worker_mod4.py", "--pool-json", pj, "--out", sm,
                        "--device", args.device, "--modes", "rl_selector"], check=True, cwd=HERE,
                       stdout=subprocess.DEVNULL)
        scores = {}
        for f in (sb, sm):
            d = np.load(f)
            for key in d.files:
                scores[key] = d[key].astype(np.float64)
        k = len(gidx)
        a = _dir_stats(scores, k, False)
        b = _dir_stats(scores, k, True)
        if a is None or b is None:
            print(f"seed={seed}: mod4 not top1 in some direction -> skip")
            continue
        total_cover = a[2] + b[2]
        total_otop1 = a[3] + b[3]
        rows.append((total_cover, -total_otop1, seed, a, b))
        print(f"seed={seed}: A(anchor#{a[1]} cover={a[2]} otop1={a[3]}) "
              f"B(anchor#{b[1]} cover={b[2]} otop1={b[3]}) total_cover={total_cover}", flush=True)

    rows.sort(reverse=True)
    print("\n=== ranking (best first) ===")
    for total_cover, neg_otop1, seed, a, b in rows[:8]:
        print(f"seed={seed} total_cover={total_cover} total_otop1={-neg_otop1}")
    # cleanup temp npz/json
    for f in os.listdir(OUT):
        if f.startswith(("pool_s", "scores_baseline_s", "scores_mod4_s")):
            os.remove(os.path.join(OUT, f))


if __name__ == "__main__":
    main()
