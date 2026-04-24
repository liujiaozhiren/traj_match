"""
Evaluate *rank position* (序位) of each pair's own post among all posts.

Metric:
  For each i:
    query = pre_i (A)   (lat/lon only, length 10)
    candidates = all post_j (B) (lat/lon only, length 10)
    sort by DTW(query, post_j) ascending (more similar first)
    rank_i = 1 + number of j with DTW < DTW(query, post_i)

Because exact N^2 DTW is expensive in Python, we use a 2-stage approximate rank:
  - stage1: L2 on flattened 10x2 to prefilter top-M candidates per query
  - stage2: exact DTW computed only on those M (plus self), rank measured within this set

This is accurate if true competitors are mostly within the L2 top-M set; increase --prefilter-m to tighten.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=str, required=True, help="Path to all_pairs_*.pkl (7-tuples).")
    p.add_argument(
        "--mode",
        type=str,
        default="ab",
        choices=["ab", "fab"],
        help="ab => use (A,B) (10pts). fab => use (F_A,F_B) (default 12pts).",
    )
    p.add_argument("--len", type=int, default=None, help="Sequence length for DTW (default depends on mode).")
    p.add_argument("--prefilter-m", type=int, default=1024, help="L2 prefilter candidates per query.")
    p.add_argument("--workers", type=int, default=0, help="0 => os.cpu_count()")
    p.add_argument("--log-every", type=int, default=500)
    return p.parse_args()


def _latlon(points: list[list[float]], n: int) -> np.ndarray:
    arr = np.asarray([[float(p[1]), float(p[2])] for p in points[: int(n)]], dtype=np.float32)
    if arr.shape != (int(n), 2):
        raise ValueError(f"expected ({n},2) got {arr.shape}")
    return arr


def _dtw_latlon(a: np.ndarray, b: np.ndarray) -> float:
    na = a.shape[0]
    nb = b.shape[0]
    prev = np.full((nb + 1,), np.inf, dtype=np.float32)
    curr = np.full((nb + 1,), np.inf, dtype=np.float32)
    prev[0] = 0.0
    for i in range(1, na + 1):
        curr[0] = np.inf
        ai = a[i - 1]
        for j in range(1, nb + 1):
            bj = b[j - 1]
            cost = float(np.linalg.norm(ai - bj))
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[nb])


@dataclass
class Pack:
    pre: np.ndarray  # (N,L,2)
    post: np.ndarray  # (N,L,2)
    pre_flat: np.ndarray  # (N,2L)
    post_flat: np.ndarray  # (N,2L)


def _load_pack(path: str, *, mode: str, L: int) -> Pack:
    pairs = pickle.load(open(path, "rb"))
    n = len(pairs)
    pre = np.zeros((n, int(L), 2), dtype=np.float32)
    post = np.zeros((n, int(L), 2), dtype=np.float32)
    for i, item in enumerate(pairs):
        mid, A, B, L_A, L_B, F_A, F_B = item
        if mode == "ab":
            pre[i] = _latlon(A, int(L))
            post[i] = _latlon(B, int(L))
        elif mode == "fab":
            pre[i] = _latlon(F_A, int(L))
            post[i] = _latlon(F_B, int(L))
        else:
            raise ValueError(f"unknown mode: {mode}")
    return Pack(pre=pre, post=post, pre_flat=pre.reshape(n, -1), post_flat=post.reshape(n, -1))


_G = {}


def _init_worker(pack_bytes: bytes):
    global _G
    _G = pickle.loads(pack_bytes)


def _work_one(i_m):
    i, prefilter_m = i_m
    pack: Pack = _G["pack"]
    qf = pack.pre_flat[i]
    dif = pack.post_flat - qf[None, :]
    d2 = np.einsum("ij,ij->i", dif, dif)
    m = min(int(prefilter_m), int(d2.shape[0]))
    idx = np.argpartition(d2, kth=m - 1)[:m]
    # ensure self is included
    if i not in idx:
        idx = np.concatenate([idx, np.asarray([i], dtype=np.int64)])

    q = pack.pre[i]
    self_d = _dtw_latlon(q, pack.post[i])
    better = 0
    for j in idx:
        j = int(j)
        if j == i:
            continue
        d = _dtw_latlon(q, pack.post[j])
        if d < self_d:
            better += 1
    # rank within this candidate set (1 is best)
    return int(i), int(better + 1), float(self_d), int(idx.shape[0])


def _summary(ranks: np.ndarray):
    r = ranks.astype(np.int64)
    def frac_le(k: int) -> float:
        return float(np.mean(r <= int(k)))

    qs = np.quantile(r, [0.0, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        "mean": float(np.mean(r)),
        "p10": float(qs[1]),
        "p50": float(qs[2]),
        "p90": float(qs[3]),
        "p99": float(qs[4]),
        "max": int(qs[5]),
        "top1": frac_le(1),
        "top5": frac_le(5),
        "top10": frac_le(10),
        "top20": frac_le(20),
        "top50": frac_le(50),
    }


def main():
    args = parse_args()
    mode = str(args.mode)
    if args.len is None:
        L = 10 if mode == "ab" else 12
    else:
        L = int(args.len)
    pack = _load_pack(args.pairs, mode=mode, L=int(L))
    n = pack.pre.shape[0]
    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, int(os.cpu_count() or 1))
    print(
        f"[rank_dtw] file={args.pairs} mode={mode} L={int(L)} N={n} prefilter_m={args.prefilter_m} workers={workers}",
        flush=True,
    )

    ranks = np.zeros((n,), dtype=np.int64)
    cand_sizes = np.zeros((n,), dtype=np.int64)
    pack_bytes = pickle.dumps({"pack": pack}, protocol=pickle.HIGHEST_PROTOCOL)
    with mp.Pool(processes=workers, initializer=_init_worker, initargs=(pack_bytes,)) as pool:
        it = pool.imap_unordered(_work_one, ((i, int(args.prefilter_m)) for i in range(n)), chunksize=8)
        done = 0
        for i, rk, self_d, csz in it:
            ranks[i] = int(rk)
            cand_sizes[i] = int(csz)
            done += 1
            if done % int(args.log_every) == 0:
                print(f"[rank_dtw] progress {done}/{n}", flush=True)

    summ = _summary(ranks)
    print(
        "[rank_dtw] rank stats (lower is more similar; higher is worse): "
        + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in summ.items()),
        flush=True,
    )
    print(f"[rank_dtw] avg_candidate_set_size={float(np.mean(cand_sizes)):.1f}", flush=True)


if __name__ == "__main__":
    main()

