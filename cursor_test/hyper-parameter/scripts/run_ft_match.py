"""
Run 8/4 match FT with hydra-only cache (meters). Shenzhen v2 pairs pkl only.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import os
import pickle

import torch

import cfg
from diffusion.fixlendiff84 import FixLenDiff84
from ft_match.hydra_cache import build_hydra_only_cache
from ft_match.train import run_ft_match_84


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-pkl", type=str, default=cfg.pairs_pkl)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--cache", type=str, default=None)
    p.add_argument("--regenerate-cache", action="store_true")
    p.add_argument("--hydra-tail", type=int, default=None)
    p.add_argument("--cache-gen-batch", type=int, default=16)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--rl-use", type=int, default=None)
    p.add_argument("--match-dim", type=int, default=None)
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--early-stop", type=int, default=10)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-frac", type=float, default=0.15)
    p.add_argument("--valid-sample-seed", type=int, default=None)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--match-scheme", type=int, default=4, choices=[3, 4])
    p.add_argument("--train-loss-mode", type=str, default="bb_ce", choices=["bce", "bb_ce"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--diff-pre-ckpt", type=str, required=True)
    p.add_argument("--diff-post-ckpt", type=str, required=True)
    p.add_argument("--skip-train", action="store_true", help="Only build cache if needed; skip match FT")
    return p.parse_args()


def _load_pairs_pkl(path: str):
    raw = pickle.load(open(path, "rb"))
    if not isinstance(raw, list):
        raise TypeError(f"--pairs-pkl must be a list, got {type(raw).__name__}")
    if raw and (not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7):
        raise TypeError(f"--pairs-pkl items must be 7-tuples; first item len={len(raw[0]) if raw else 'n/a'}")
    return raw


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device

    hydra_tail = int(args.hydra_tail if args.hydra_tail is not None else cfg.hydra_tail)
    rl_use = int(args.rl_use if args.rl_use is not None else cfg.rl_use)
    match_dim = int(args.match_dim if args.match_dim is not None else cfg.match_dim)
    pairs_pkl = os.path.abspath(args.pairs_pkl)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    cache_path = args.cache or os.path.join(out_dir, f"hydra_only_cache_tail{hydra_tail}.pt")

    if args.regenerate_cache or not os.path.exists(cache_path):
        print(f"[run_ft_match] loading pairs: {pairs_pkl}", flush=True)
        pairs = _load_pairs_pkl(pairs_pkl)
        if args.max_pairs is not None:
            pairs = pairs[: int(args.max_pairs)]
        print(f"[run_ft_match] pairs loaded: {len(pairs)}", flush=True)
        pre = FixLenDiff84(device=cfg.device, lr=cfg.diff_lr)
        post = FixLenDiff84(device=cfg.device, lr=cfg.diff_lr)
        pre.unet.load_state_dict(torch.load(args.diff_pre_ckpt, map_location=cfg.device), strict=True)
        post.unet.load_state_dict(torch.load(args.diff_post_ckpt, map_location=cfg.device), strict=True)
        cache = build_hydra_only_cache(
            pairs=pairs,
            pre_diff=pre,
            post_diff=post,
            hydra_tail=hydra_tail,
            batch=int(args.cache_gen_batch),
            device=cfg.device,
        )
        cache.save(cache_path)
        print(f"[run_ft_match] cache saved -> {cache_path}", flush=True)

    if args.skip_train:
        print("[run_ft_match] skip-train set; done.", flush=True)
        return

    res = run_ft_match_84(
        output_dir=out_dir,
        cache_path=cache_path,
        train_batch=int(args.train_batch),
        valid_batch=int(args.valid_batch),
        lr=float(args.lr),
        max_epochs=int(args.max_epochs),
        early_stop=int(args.early_stop),
        rl_use=rl_use,
        sample_ratio=float(args.sample_ratio),
        sample_seed=int(args.sample_seed),
        valid_sample_frac=float(args.valid_sample_frac),
        valid_sample_seed=args.valid_sample_seed,
        valid_n_passes=int(args.valid_n_passes),
        match_scheme=int(args.match_scheme),
        pairs_pkl=pairs_pkl,
        max_pairs=args.max_pairs,
        train_loss_mode=str(args.train_loss_mode),
        match_dim=match_dim,
    )
    print(
        f"[run_ft_match] done best_ep={res.best_epoch} best_acc={res.best_acc:.4f} best_ckpt={res.best_ckpt}",
        flush=True,
    )
    if res.posthoc_valid:
        ph_path = os.path.join(out_dir, "posthoc_valid_best.json")
        with open(ph_path, "w", encoding="utf-8") as f:
            json.dump(res.posthoc_valid, f, indent=2)


if __name__ == "__main__":
    main()
