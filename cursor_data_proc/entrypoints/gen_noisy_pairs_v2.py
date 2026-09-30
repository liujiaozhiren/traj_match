"""
Generate alternative noisy boundary points for (F_A, F_B) using DTW retrieval,
and evaluate how *poorly* each pair's own pre/post match (DTW on lat/lon).

Input:  cursor_mod_4/pipeline/all_pairs_v1.pkl  (list of 7-tuples)
Output: cursor_mod_4/pipeline/all_pairs_v2_<scheme>.pkl

Schemes for composing c1/c2 from the top-K nearest candidates:
  - mean:    trimmed mean of candidate LB head-2 points
  - far:     pick candidate whose LB head-2 is farthest from this pair's LB head-2
  - maxdtw:  among top-K, pick the candidate that maximizes DTW(new_pre, old_post)

Notes on practicality:
  Full pairwise DTW (N^2) is expensive in pure Python. This script uses a 2-stage
  retrieval: cheap L2 distance on flattened 10x2 lat/lon to prefilter top-M,
  then exact DTW on those M to select top-K.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="cursor_mod_4/pipeline/all_pairs_v1.pkl")
    p.add_argument("--output-dir", type=str, default="cursor_mod_4/pipeline")
    p.add_argument("--schemes", type=str, default="mean,far,maxdtw", help="Comma-separated: mean,far,maxdtw")
    p.add_argument("--max-pairs", type=int, default=None, help="Debug: only process first N pairs.")
    p.add_argument("--prefilter-m", type=int, default=64, help="Stage-1 candidates per query via L2.")
    p.add_argument("--topk", type=int, default=10, help="Final DTW top-K nearest candidates.")
    p.add_argument("--trim", type=int, default=2, help="For mean scheme: trim this many from each side.")
    p.add_argument("--workers", type=int, default=0, help="0 => use os.cpu_count(); 1 => single process.")
    p.add_argument("--log-every", type=int, default=200, help="Print progress every N pairs (per scheme).")
    return p.parse_args()


def _latlon_seq(points: list[list[float]], n: int) -> np.ndarray:
    pts = points[:n]
    arr = np.asarray([[float(p[1]), float(p[2])] for p in pts], dtype=np.float32)
    if arr.shape != (n, 2):
        raise ValueError(f"expected ({n},2) got {arr.shape}")
    return arr


def _dtw_latlon(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW with Euclidean cost on (lat,lon). a,b: (L,2)."""
    na = a.shape[0]
    nb = b.shape[0]
    # dp with rolling rows
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


def _prefilter_topm_l2(a_flat: np.ndarray, B_flat: np.ndarray, m: int) -> np.ndarray:
    """Return indices of top-m smallest L2 distances between a_flat and each row of B_flat."""
    # squared distances
    dif = B_flat - a_flat[None, :]
    d2 = np.einsum("ij,ij->i", dif, dif)
    mm = min(int(m), int(d2.shape[0]))
    idx = np.argpartition(d2, kth=mm - 1)[:mm]
    return idx


def _topk_by_dtw(a: np.ndarray, B: np.ndarray, cand_idx: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute DTW to candidates and return top-k (idx, dist) sorted by dist asc."""
    dists = []
    for ii in cand_idx:
        d = _dtw_latlon(a, B[ii])
        dists.append((int(ii), float(d)))
    dists.sort(key=lambda x: x[1])
    out = dists[: min(int(k), len(dists))]
    return np.asarray([x[0] for x in out], dtype=np.int64), np.asarray([x[1] for x in out], dtype=np.float32)


def _blend_pt(p: list[float], q: list[float], alpha: float) -> list[float]:
    """Linear blend p->q in (t,lat,lon)."""
    return [
        float(p[0]) * (1 - alpha) + float(q[0]) * alpha,
        float(p[1]) * (1 - alpha) + float(q[1]) * alpha,
        float(p[2]) * (1 - alpha) + float(q[2]) * alpha,
    ]


def _harmonize(d1: list[float], d2: list[float], c1: list[float], c2: list[float]) -> tuple[list[float], list[float]]:
    """
    Simple smoothing to avoid abrupt kink: pull d's slightly towards the midpoint path.
    Kept intentionally lightweight; can be replaced with a stricter rule later.
    """
    m1 = _blend_pt(d1, c1, 0.35)
    m2 = _blend_pt(d2, c2, 0.35)
    return m1, m2


@dataclass
class PairPack:
    all_pairs: list[tuple]
    A10: np.ndarray  # (N,10,2) query from L_A[:10]
    B10: np.ndarray  # (N,10,2) tail from L_B[-10:]
    A10_flat: np.ndarray  # (N,20)
    B10_flat: np.ndarray  # (N,20)
    LB_head2: list[tuple[list[float], list[float]]]  # per-pair (pt0, pt1) full 3D points
    LA_anchor: list[list[float]]  # per-pair L_A[7] full pt
    LB_anchor: list[list[float]]  # per-pair L_B[4] full pt (anchor after replacing first 4)
    FA: list[list[list[float]]]  # per-pair F_A points (len 12)
    FB: list[list[list[float]]]  # per-pair F_B points (len 12)


def _load_pack(path: str, max_pairs: int | None) -> PairPack:
    all_pairs = pickle.load(open(path, "rb"))
    if max_pairs is not None:
        all_pairs = all_pairs[: int(max_pairs)]
    n = len(all_pairs)
    A10 = np.zeros((n, 10, 2), dtype=np.float32)
    B10 = np.zeros((n, 10, 2), dtype=np.float32)
    LB_head2 = []
    LA_anchor = []
    LB_anchor = []
    FA = []
    FB = []
    for i, item in enumerate(all_pairs):
        mid, A, B, L_A, L_B, F_A, F_B = item
        A10[i] = _latlon_seq(L_A, 10)
        B10[i] = _latlon_seq(L_B[-10:], 10)
        LB_head2.append((copy.deepcopy(L_B[0]), copy.deepcopy(L_B[1])))
        LA_anchor.append(copy.deepcopy(L_A[7]))
        # after we replace first 4 points of FB, the next point is index 4
        LB_anchor.append(copy.deepcopy(L_B[4]))
        FA.append(copy.deepcopy(F_A))
        FB.append(copy.deepcopy(F_B))
    A10_flat = A10.reshape(n, -1)
    B10_flat = B10.reshape(n, -1)
    return PairPack(
        all_pairs=all_pairs,
        A10=A10,
        B10=B10,
        A10_flat=A10_flat,
        B10_flat=B10_flat,
        LB_head2=LB_head2,
        LA_anchor=LA_anchor,
        LB_anchor=LB_anchor,
        FA=FA,
        FB=FB,
    )


def _compose_c_mean(head2_pts: list[tuple[list[float], list[float]]], trim: int) -> tuple[list[float], list[float]]:
    """Trimmed mean over candidates for c1,c2 (t,lat,lon) separately."""
    p0 = np.asarray([[float(x[0][0]), float(x[0][1]), float(x[0][2])] for x in head2_pts], dtype=np.float32)
    p1 = np.asarray([[float(x[1][0]), float(x[1][1]), float(x[1][2])] for x in head2_pts], dtype=np.float32)
    # trim by lat (robust enough for now)
    def _trim_mean(P: np.ndarray) -> np.ndarray:
        if P.shape[0] <= 2 * trim:
            return P.mean(axis=0)
        order = np.argsort(P[:, 1])  # sort by lat
        keep = order[trim : P.shape[0] - trim]
        return P[keep].mean(axis=0)

    c1 = _trim_mean(p0)
    c2 = _trim_mean(p1)
    return [float(c1[0]), float(c1[1]), float(c1[2])], [float(c2[0]), float(c2[1]), float(c2[2])]


def _compose_c_far(self_head2: tuple[list[float], list[float]], cand_head2: list[tuple[list[float], list[float]]]) -> tuple[list[float], list[float]]:
    """Pick the candidate whose head2 (lat,lon) is farthest from this pair's LB head2."""
    s0, s1 = self_head2
    s = np.asarray([float(s0[1]), float(s0[2]), float(s1[1]), float(s1[2])], dtype=np.float32)
    best = None
    best_d = -1.0
    for (p0, p1) in cand_head2:
        v = np.asarray([float(p0[1]), float(p0[2]), float(p1[1]), float(p1[2])], dtype=np.float32)
        d = float(np.linalg.norm(v - s))
        if d > best_d:
            best_d = d
            best = (p0, p1)
    assert best is not None
    return copy.deepcopy(best[0]), copy.deepcopy(best[1])


def _build_fa_tail(LA7: list[float], c1: list[float], c2: list[float]) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Construct last 4 points [d1,d2,c1,c2] such that d1,d2 transition from LA[7] towards c1/c2.
    """
    d1 = _blend_pt(LA7, c1, 0.60)
    d2 = _blend_pt(LA7, c2, 0.85)
    d1, d2 = _harmonize(d1, d2, c1, c2)
    return d1, d2, c1, c2


def _build_fb_head(LB4: list[float], c1: list[float], c2: list[float]) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Construct first 4 points [c1,c2,d1,d2] such that d1,d2 transition from c2 towards the existing LB[4].
    """
    d1 = _blend_pt(c2, LB4, 0.35)
    d2 = _blend_pt(c2, LB4, 0.70)
    d1, d2 = _harmonize(d1, d2, c1, c2)
    return c1, c2, d1, d2


def _eval_self_match(pre10: np.ndarray, post10: np.ndarray) -> float:
    return _dtw_latlon(pre10, post10)


def _scheme_generate(pack: PairPack, scheme: str, *, prefilter_m: int, topk: int, trim: int) -> tuple[list[tuple], np.ndarray]:
    n = pack.A10.shape[0]
    out_pairs = []
    scores = np.zeros((n,), dtype=np.float32)

    for k in range(n):
        # query a = L_A[:10]
        a = pack.A10[k]
        a_flat = pack.A10_flat[k]

        # stage-1 prefilter using L2 to all B10 tails
        cand_idx = _prefilter_topm_l2(a_flat, pack.B10_flat, m=prefilter_m)
        top_idx, _ = _topk_by_dtw(a, pack.B10, cand_idx, k=topk)

        cand_head2 = [pack.LB_head2[int(ii)] for ii in top_idx]

        if scheme == "mean":
            c1, c2 = _compose_c_mean(cand_head2, trim=trim)
        elif scheme == "far":
            c1, c2 = _compose_c_far(pack.LB_head2[k], cand_head2)
        elif scheme == "maxdtw":
            # choose candidate among top-k that maximizes DTW(new_pre, old_post)
            best_score = -1.0
            best_c = None
            old_post = _latlon_seq(pack.FB[k][-10:], 10)
            for p0, p1 in cand_head2:
                cc1, cc2 = copy.deepcopy(p0), copy.deepcopy(p1)
                d1, d2, c1t, c2t = _build_fa_tail(pack.LA_anchor[k], cc1, cc2)
                FA_tmp = copy.deepcopy(pack.FA[k])
                FA_tmp[-4:] = [d1, d2, c1t, c2t]
                pre_tmp = _latlon_seq(FA_tmp[:10], 10)
                sc = _eval_self_match(pre_tmp, old_post)
                if sc > best_score:
                    best_score = sc
                    best_c = (cc1, cc2)
            assert best_c is not None
            c1, c2 = best_c
        else:
            raise ValueError(f"unknown scheme: {scheme}")

        # build new FA and new FB
        FA_new = copy.deepcopy(pack.FA[k])
        FB_new = copy.deepcopy(pack.FB[k])

        d1, d2, cc1, cc2 = _build_fa_tail(pack.LA_anchor[k], copy.deepcopy(c1), copy.deepcopy(c2))
        FA_new[-4:] = [d1, d2, cc1, cc2]

        # For FB, reuse the same c1/c2 selection logic but mirrored by running another retrieval:
        # query is this pair's L_B[-10:], candidates are other pairs' L_A[:10].
        bq = pack.B10[k]
        bq_flat = pack.B10_flat[k]
        cand_idx2 = _prefilter_topm_l2(bq_flat, pack.A10_flat, m=prefilter_m)
        top_idx2, _ = _topk_by_dtw(bq, pack.A10, cand_idx2, k=topk)
        cand_la_head2 = []
        for ii in top_idx2:
            # use that candidate's L_A last-2 points as "would-be" FB head-2 (mirror)
            mid, A, B, L_A, L_B, F_A, F_B = pack.all_pairs[int(ii)]
            cand_la_head2.append((copy.deepcopy(L_A[-2]), copy.deepcopy(L_A[-1])))

        if scheme == "mean":
            c1b, c2b = _compose_c_mean(cand_la_head2, trim=trim)
        elif scheme == "far":
            # far from this pair's LA tail-2 (proxy for boundary)
            mid, A, B, L_A_self, L_B_self, F_A, F_B = pack.all_pairs[k]
            self_ref = (copy.deepcopy(L_A_self[-2]), copy.deepcopy(L_A_self[-1]))
            c1b, c2b = _compose_c_far(self_ref, cand_la_head2)
        elif scheme == "maxdtw":
            best_score = -1.0
            best_c = None
            old_pre = _latlon_seq(pack.FA[k][:10], 10)
            for p0, p1 in cand_la_head2:
                cc1, cc2 = copy.deepcopy(p0), copy.deepcopy(p1)
                h1, h2, h3, h4 = _build_fb_head(pack.LB_anchor[k], cc1, cc2)
                FB_tmp = copy.deepcopy(pack.FB[k])
                FB_tmp[:4] = [h1, h2, h3, h4]
                post_tmp = _latlon_seq(FB_tmp[-10:], 10)
                sc = _eval_self_match(old_pre, post_tmp)
                if sc > best_score:
                    best_score = sc
                    best_c = (cc1, cc2)
            assert best_c is not None
            c1b, c2b = best_c
        else:
            raise ValueError(f"unknown scheme: {scheme}")

        h1, h2, h3, h4 = _build_fb_head(pack.LB_anchor[k], copy.deepcopy(c1b), copy.deepcopy(c2b))
        FB_new[:4] = [h1, h2, h3, h4]

        # derive new A/B as in the old logic
        A_new = copy.deepcopy(FA_new[:10])
        B_new = copy.deepcopy(FB_new[-10:])
        pre10 = _latlon_seq(A_new, 10)
        post10 = _latlon_seq(B_new, 10)
        scores[k] = _eval_self_match(pre10, post10)

        mid, A_old, B_old, L_A, L_B, F_A_old, F_B_old = pack.all_pairs[k]
        out_pairs.append((mid, A_new, B_new, L_A, L_B, FA_new, FB_new))

    return out_pairs, scores


def _print_stats(name: str, scores: np.ndarray):
    s = np.asarray(scores, dtype=np.float64)
    q = np.quantile(s, [0.0, 0.1, 0.5, 0.9, 1.0])
    print(
        f"[{name}] self_dtw(pre,post) mean={s.mean():.4f} std={s.std():.4f} "
        f"min={q[0]:.4f} p10={q[1]:.4f} p50={q[2]:.4f} p90={q[3]:.4f} max={q[4]:.4f}",
        flush=True,
    )


_G = {}


def _init_worker(pack_bytes: bytes):
    global _G
    _G = pickle.loads(pack_bytes)


def _work_one(task):
    scheme, k, prefilter_m, topk, trim = task
    pack: PairPack = _G["pack"]

    # ---------- FA side ----------
    a = pack.A10[k]
    a_flat = pack.A10_flat[k]
    cand_idx = _prefilter_topm_l2(a_flat, pack.B10_flat, m=prefilter_m)
    top_idx, _ = _topk_by_dtw(a, pack.B10, cand_idx, k=topk)
    cand_head2 = [pack.LB_head2[int(ii)] for ii in top_idx]

    if scheme == "mean":
        c1, c2 = _compose_c_mean(cand_head2, trim=trim)
    elif scheme == "far":
        c1, c2 = _compose_c_far(pack.LB_head2[k], cand_head2)
    elif scheme == "maxdtw":
        best_score = -1.0
        best_c = None
        old_post = _latlon_seq(pack.FB[k][-10:], 10)
        for p0, p1 in cand_head2:
            cc1, cc2 = copy.deepcopy(p0), copy.deepcopy(p1)
            d1, d2, c1t, c2t = _build_fa_tail(pack.LA_anchor[k], cc1, cc2)
            FA_tmp = copy.deepcopy(pack.FA[k])
            FA_tmp[-4:] = [d1, d2, c1t, c2t]
            pre_tmp = _latlon_seq(FA_tmp[:10], 10)
            sc = _eval_self_match(pre_tmp, old_post)
            if sc > best_score:
                best_score = sc
                best_c = (cc1, cc2)
        assert best_c is not None
        c1, c2 = best_c
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    FA_new = copy.deepcopy(pack.FA[k])
    FB_new = copy.deepcopy(pack.FB[k])
    d1, d2, cc1, cc2 = _build_fa_tail(pack.LA_anchor[k], copy.deepcopy(c1), copy.deepcopy(c2))
    FA_new[-4:] = [d1, d2, cc1, cc2]

    # ---------- FB side (mirrored retrieval) ----------
    bq = pack.B10[k]
    bq_flat = pack.B10_flat[k]
    cand_idx2 = _prefilter_topm_l2(bq_flat, pack.A10_flat, m=prefilter_m)
    top_idx2, _ = _topk_by_dtw(bq, pack.A10, cand_idx2, k=topk)
    cand_la_head2 = []
    for ii in top_idx2:
        mid, A, B, L_A, L_B, F_A, F_B = pack.all_pairs[int(ii)]
        cand_la_head2.append((copy.deepcopy(L_A[-2]), copy.deepcopy(L_A[-1])))

    if scheme == "mean":
        c1b, c2b = _compose_c_mean(cand_la_head2, trim=trim)
    elif scheme == "far":
        mid, A, B, L_A_self, L_B_self, F_A0, F_B0 = pack.all_pairs[k]
        self_ref = (copy.deepcopy(L_A_self[-2]), copy.deepcopy(L_A_self[-1]))
        c1b, c2b = _compose_c_far(self_ref, cand_la_head2)
    elif scheme == "maxdtw":
        best_score = -1.0
        best_c = None
        old_pre = _latlon_seq(pack.FA[k][:10], 10)
        for p0, p1 in cand_la_head2:
            cc1, cc2 = copy.deepcopy(p0), copy.deepcopy(p1)
            h1, h2, h3, h4 = _build_fb_head(pack.LB_anchor[k], cc1, cc2)
            FB_tmp = copy.deepcopy(pack.FB[k])
            FB_tmp[:4] = [h1, h2, h3, h4]
            post_tmp = _latlon_seq(FB_tmp[-10:], 10)
            sc = _eval_self_match(old_pre, post_tmp)
            if sc > best_score:
                best_score = sc
                best_c = (cc1, cc2)
        assert best_c is not None
        c1b, c2b = best_c
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    h1, h2, h3, h4 = _build_fb_head(pack.LB_anchor[k], copy.deepcopy(c1b), copy.deepcopy(c2b))
    FB_new[:4] = [h1, h2, h3, h4]

    # ---------- new A/B + score ----------
    A_new = copy.deepcopy(FA_new[:10])
    B_new = copy.deepcopy(FB_new[-10:])
    pre10 = _latlon_seq(A_new, 10)
    post10 = _latlon_seq(B_new, 10)
    score = _eval_self_match(pre10, post10)

    mid, A_old, B_old, L_A, L_B, F_A_old, F_B_old = pack.all_pairs[k]
    out = (mid, A_new, B_new, L_A, L_B, FA_new, FB_new)
    return k, out, float(score)


def main():
    args = parse_args()
    schemes = [s.strip() for s in str(args.schemes).split(",") if s.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    pack = _load_pack(args.input, args.max_pairs)
    print(f"[gen_v2] loaded pairs={len(pack.all_pairs)} from {args.input}", flush=True)
    workers = int(args.workers)
    if workers <= 0:
        workers = max(1, int(os.cpu_count() or 1))
    print(
        f"[gen_v2] schemes={schemes} prefilter_m={args.prefilter_m} topk={args.topk} workers={workers}",
        flush=True,
    )

    for scheme in schemes:
        n = len(pack.all_pairs)
        out_pairs = [None] * n
        scores = np.zeros((n,), dtype=np.float32)

        if workers == 1:
            global _G
            _G = {"pack": pack}
            for k in range(n):
                kk, out, sc = _work_one((scheme, k, int(args.prefilter_m), int(args.topk), int(args.trim)))
                out_pairs[int(kk)] = out
                scores[int(kk)] = float(sc)
                if (k + 1) % int(args.log_every) == 0:
                    print(f"[v2_{scheme}] progress {k+1}/{n}", flush=True)
        else:
            import multiprocessing as mp

            pack_bytes = pickle.dumps({"pack": pack}, protocol=pickle.HIGHEST_PROTOCOL)
            with mp.Pool(processes=workers, initializer=_init_worker, initargs=(pack_bytes,)) as pool:
                it = pool.imap_unordered(
                    _work_one,
                    ((scheme, k, int(args.prefilter_m), int(args.topk), int(args.trim)) for k in range(n)),
                    chunksize=16,
                )
                done = 0
                for kk, out, sc in it:
                    out_pairs[int(kk)] = out
                    scores[int(kk)] = float(sc)
                    done += 1
                    if done % int(args.log_every) == 0:
                        print(f"[v2_{scheme}] progress {done}/{n}", flush=True)

        out_path = os.path.join(args.output_dir, f"all_pairs_v2_{scheme}.pkl")
        pickle.dump(out_pairs, open(out_path, "wb"))
        _print_stats(f"v2_{scheme}", scores)
        print(f"[v2_{scheme}] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

