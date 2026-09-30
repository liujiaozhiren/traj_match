"""
Trajectory baseline on noisy resampled (lon, lat) pre/post: DTW, Hausdorff, discrete Fréchet, or SSPD.

Same valid split as match FT (sample_db) and protocol: ~15% random subset × many passes, metrics averaged.

Writes JSON under cursor_baseline/results/ by default.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.modules["cfg"] = importlib.import_module("cursor_baseline.model_lib.cfg")
import cfg  # noqa: E402

from cursor_baseline.traj_distances import pairwise_neg_distance_matrix  # noqa: E402
from cursor_baseline.lonlat_traj import stack_subsample_lonlat  # noqa: E402
from cursor_baseline.metrics import retrieval_metrics  # noqa: E402
from cursor_baseline.model_lib.proc_pairs import clean_pairs, proc_multi_single_traj  # noqa: E402
from cursor_baseline.model_lib.sample_db import sample_db  # noqa: E402


def _load_all_pairs(pairs_pkl: str | None = None):
    if pairs_pkl is not None:
        return pickle.load(open(pairs_pkl, "rb"))
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    return clean_pairs(mul_p + sing_p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(_ROOT / "cursor_baseline" / "results"),
        help="Directory for JSON results.",
    )
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-frac", type=float, default=0.15)
    p.add_argument("--valid-n-passes", type=int, default=30)
    p.add_argument("--valid-seed", type=int, default=None, help="Base seed for subsample+noise; default sample-seed.")
    p.add_argument("--n-points", type=int, default=12)
    p.add_argument("--noise-std-deg", type=float, default=5e-5, help="Gaussian std on lon/lat in degrees (each dim).")
    p.add_argument(
        "--metric",
        type=str,
        default="dtw",
        choices=("dtw", "hausdorff", "frechet", "sspd"),
        help="Pairwise distance: dtw | hausdorff | frechet (discrete) | sspd (symmetric segment-path mean).",
    )
    p.add_argument("--chunk-queries", type=int, default=32, help="Similarity matrix query chunk size.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--pairs-pkl",
        type=str,
        default=None,
        help="If set, load a direct clean_pairs list from this pickle (same 7-tuple format as match FT).",
    )
    p.add_argument(
        "--noise-blend-alpha",
        type=float,
        default=None,
        help="Optional [0,1]. Blend noisy resampled lon/lat with label windows (see cursor_mod_4.traj_noise_blend).",
    )
    args = p.parse_args()

    if args.noise_blend_alpha is not None:
        ba = float(args.noise_blend_alpha)
        if not (0.0 <= ba <= 1.0):
            raise SystemExit("--noise-blend-alpha must be in [0,1]")

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_pairs = _load_all_pairs(pairs_pkl=str(args.pairs_pkl) if args.pairs_pkl else None)
    valid_pairs = sample_db(all_pairs, ratio=float(args.sample_ratio), seed=int(args.sample_seed))
    n_valid = len(valid_pairs)
    k = max(1, min(n_valid, int(round(float(n_valid) * float(args.valid_sample_frac)))))
    n_passes = max(1, int(args.valid_n_passes))
    base_sub = int(args.valid_seed) if args.valid_seed is not None else int(args.sample_seed)

    sums = {key: 0.0 for key in ("acc", "hr5", "hr10", "hr20", "r5_at_10", "r10_at_20")}
    per_pass: list[dict] = []

    for rep in range(n_passes):
        g_cpu = torch.Generator(device="cpu")
        g_cpu.manual_seed(int(base_sub) + rep * 7919)
        samp = torch.randperm(n_valid, generator=g_cpu)[:k].long()

        noise_base = int(base_sub) * 10007 + rep * 1_000_003 + k * 17
        pre_t, post_t = stack_subsample_lonlat(
            valid_pairs,
            samp,
            n_points=int(args.n_points),
            noise_std_deg=float(args.noise_std_deg),
            noise_seed_base=noise_base,
            device=device,
            noise_blend_alpha=args.noise_blend_alpha,
        )

        scores = pairwise_neg_distance_matrix(
            pre_t,
            post_t,
            str(args.metric),
            chunk_queries=int(args.chunk_queries),
        )
        m = retrieval_metrics(scores)
        per_pass.append({"pass": rep, **m})
        for key in sums:
            sums[key] += m[key]

    mean = {key: sums[key] / n_passes for key in sums}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp_utc": ts,
        "method": f"{args.metric}_lonlat_noisy_resampled",
        "metric": str(args.metric),
        "n_valid": n_valid,
        "k_subsample": k,
        "valid_sample_frac": float(args.valid_sample_frac),
        "valid_n_passes": n_passes,
        "n_points": int(args.n_points),
        "noise_std_deg": float(args.noise_std_deg),
        "sample_ratio": float(args.sample_ratio),
        "sample_seed": int(args.sample_seed),
        "valid_seed_base": base_sub,
        "device": str(device),
        "metrics_mean_over_passes": mean,
        "per_pass": per_pass,
    }
    out_path = out_dir / f"{args.metric}_lonlat_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(
        f"[lonlat_metric_baseline] metric={args.metric} n_valid={n_valid} k={k} passes={n_passes} "
        f"mean acc={mean['acc']:.4f} HR5={mean['hr5']:.4f} HR10={mean['hr10']:.4f} HR20={mean['hr20']:.4f} "
        f"R5@10={mean['r5_at_10']:.4f} R10@20={mean['r10_at_20']:.4f}",
        flush=True,
    )
    print(f"[lonlat_metric_baseline] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
