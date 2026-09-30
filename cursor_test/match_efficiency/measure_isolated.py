#!/usr/bin/env python3
"""Measure one method in an isolated process (exact RSS / VRAM)."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import psutil
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from match_efficiency.constants import HYDRA_TAIL  # noqa: E402
from match_efficiency.count_params import count_method_params  # noqa: E402
from match_efficiency.gallery import build_gallery_index, slice_gallery_index  # noqa: E402
from match_efficiency.retrieval import run_retrieval_median  # noqa: E402
from match_efficiency.traj_data import make_synthetic_pairs  # noqa: E402


def _rss_mib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def gallery_batch_for(n: int, device: torch.device) -> int:
    if device.type != "cuda":
        return n
    if n <= 500:
        return n
    if n <= 1000:
        return min(n, 512)
    return min(n, 256)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--n-gallery", type=int, required=True)
    p.add_argument("--n-queries", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hydra-tail", type=int, default=HYDRA_TAIL)
    args = p.parse_args()

    method = args.method.lower()
    n = int(args.n_gallery)
    device = torch.device(args.device)
    proc = psutil.Process(os.getpid())
    peak_rss = _rss_mib()

    def sample_rss() -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, _rss_mib())

    sample_rss()
    params = count_method_params(method)

    pairs = make_synthetic_pairs(n + int(args.n_queries) + 50, seed=int(args.seed))
    indices = list(range(n))
    gb = gallery_batch_for(n, device)

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    sample_rss()
    gal_full = build_gallery_index(
        method,
        pairs,
        indices,
        device=device,
        hydra_tail=int(args.hydra_tail),
        gallery_online=True,
    )
    sample_rss()
    gal = slice_gallery_index(gal_full, n)
    query_pairs = [pairs[n + i] for i in range(int(args.n_queries))]

    timing = run_retrieval_median(
        method,
        query_pairs,
        gal,
        pairs=pairs,
        device=device,
        gallery_batch=gb,
        hydra_tail=int(args.hydra_tail),
        gallery_online=True,
    )
    sample_rss()

    peak_vram = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    out = {
        "method": method,
        "n_gallery": n,
        "p_learnable": params["p_learnable"],
        "p_aux": params["p_aux"],
        "p_total": params["p_total"],
        "peak_rss_mib": peak_rss,
        "peak_vram_mib": peak_vram,
        "t_e2e": timing.t_e2e,
        "t_query": timing.t_query,
        "t_gallery": timing.t_gallery,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
