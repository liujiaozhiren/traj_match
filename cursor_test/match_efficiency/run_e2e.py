#!/usr/bin/env python3
"""Run E2E match retrieval efficiency benchmarks (self-contained under cursor_test)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from match_efficiency.constants import (  # noqa: E402
    ALL_14_METHODS,
    DEFAULT_GALLERY_BATCH,
    DEFAULT_GALLERY_SIZE,
    HYDRA_TAIL,
    METHOD_NAMES,
)
from match_efficiency.count_params import count_method_params  # noqa: E402
from match_efficiency.gallery import build_gallery_index  # noqa: E402
from match_efficiency.retrieval import run_retrieval_median  # noqa: E402
from match_efficiency.traj_data import load_pairs_pkl, make_synthetic_pairs  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="E2E match retrieval efficiency (1 query vs N gallery).")
    p.add_argument("--methods", type=str, default=",".join(ALL_14_METHODS), help="Comma-separated method names.")
    p.add_argument("--gallery-size", type=int, default=DEFAULT_GALLERY_SIZE)
    p.add_argument("--gallery-batch", type=int, default=DEFAULT_GALLERY_BATCH)
    p.add_argument("--hydra-tail", type=int, default=HYDRA_TAIL)
    p.add_argument("--pairs-pkl", type=str, default=None, help="Optional pickle; else synthetic pairs.")
    p.add_argument("--synthetic-pairs", type=int, default=5000, help="Synthetic pool size if no pkl.")
    p.add_argument("--n-queries", type=int, default=5, help="Median over this many query runs.")
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-json", type=str, default="cursor_test/match_efficiency/out/e2e_results.json")
    p.add_argument("--count-params-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    device = torch.device(args.device)

    if args.count_params_only:
        rows = [count_method_params(m) for m in methods]
        print(json.dumps(rows, indent=2))
        return

    n_gal = int(args.gallery_size)
    n_need = n_gal + int(args.n_queries) + 10
    if args.pairs_pkl:
        pairs = load_pairs_pkl(args.pairs_pkl, max_pairs=max(n_need, n_gal + 100))
    else:
        pairs = make_synthetic_pairs(max(n_need, args.synthetic_pairs), seed=int(args.seed))

    if len(pairs) < n_gal + 1:
        raise SystemExit(f"need at least {n_gal + 1} pairs, got {len(pairs)}")

    gallery_indices = list(range(n_gal))
    query_indices = list(range(n_gal, min(n_gal + int(args.n_queries), len(pairs))))

    rows = []
    for method in methods:
        print(f"[e2e] method={method} N={n_gal} gallery_batch={args.gallery_batch} device={device}", flush=True)
        if method not in METHOD_NAMES and method not in ("dtw", "hausdorff", "frechet", "sspd"):
            print(f"  skip unknown method {method}", flush=True)
            continue

        params = count_method_params(method)
        gal = build_gallery_index(
            method,
            pairs,
            gallery_indices,
            device=device,
            hydra_tail=int(args.hydra_tail),
        )
        query_pairs = [pairs[i] for i in query_indices]
        timing = run_retrieval_median(
            method,
            query_pairs,
            gal,
            pairs=pairs,
            device=device,
            gallery_batch=int(args.gallery_batch),
            hydra_tail=int(args.hydra_tail),
        )
        row = {
            "method": method,
            "n_gallery": n_gal,
            "gallery_batch": int(args.gallery_batch),
            "hydra_tail": int(args.hydra_tail),
            "device": str(device),
            "p_total": params["p_total"],
            "p_learnable": params["p_learnable"],
            "p_aux": params["p_aux"],
            "s_index_elements": gal.storage_elements,
            "t_index_sec": gal.build_sec,
            **timing.as_dict(),
        }
        rows.append(row)
        print(
            f"  P={row['p_total']} T_e2e={row['t_e2e']:.4f}s "
            f"(q={row['t_query']:.4f} gal={row['t_gallery']:.4f}) "
            f"M_vram={row['peak_vram_mib']:.1f}MiB",
            flush=True,
        )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"gallery_size": n_gal, "n_queries": len(query_indices)}, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[e2e] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
