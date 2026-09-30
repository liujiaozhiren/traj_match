"""Generate the shared fixed candidate pool (global pair indices)."""

from __future__ import annotations

import argparse
import json
import os

import registry
from common import load_pairs, pool_global_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--pool-seed", type=int, default=42)
    ap.add_argument("--out", default="out/pool.json")
    args = ap.parse_args()

    n = len(load_pairs(registry.PAIRS_PKL))
    gidx = pool_global_indices(
        n,
        frac=args.frac,
        sample_ratio=registry.SAMPLE_RATIO,
        sample_seed=registry.SAMPLE_SEED,
        pool_seed=args.pool_seed,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(
        {"pool_gidx": gidx, "n_pairs": n, "frac": args.frac, "pool_seed": args.pool_seed},
        open(args.out, "w"),
        indent=2,
    )
    print(f"n_pairs={n} pool_size={len(gidx)} -> {args.out}")
    print("first 5:", gidx[:5])


if __name__ == "__main__":
    main()
