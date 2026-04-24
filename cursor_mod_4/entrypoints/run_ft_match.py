"""
Run 8/4 match FT with hydra-only cache (meters).

Steps:
1) load Shenzhen pairs from traj_cmb.pkl (clean_pairs)
2) if needed, generate hydra-only cache from diffusion 8/4 models
3) train match using original info (computed on-the-fly) + cached hydra tails
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_rp = str(_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

sys.modules["cfg"] = importlib.import_module("cursor_mod_4.cfg")
import cfg  # noqa: E402

from cursor_mod_4.diffusion.fixlendiff84 import FixLenDiff84  # noqa: E402
from cursor_mod_4.ft_match.hydra_cache import build_hydra_only_cache, HydraOnlyCache  # noqa: E402
from cursor_mod_4.ft_match.train import run_ft_match_84  # noqa: E402
from cursor_mod_4.proc import clean_pairs, proc_multi_single_traj  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--cache", type=str, default=None)
    p.add_argument("--regenerate-cache", action="store_true")
    p.add_argument("--hydra-tail", type=int, default=64)
    p.add_argument("--cache-gen-batch", type=int, default=256)
    p.add_argument("--max-pairs", type=int, default=None, help="Debug: limit total pairs for cache/training")
    p.add_argument("--max-raw-trajs", type=int, default=None, help="Debug: limit number of raw single trajectories processed")
    p.add_argument("--rl-use", type=int, default=16)
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--early-stop", type=int, default=15)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument(
        "--valid-sample-frac",
        type=float,
        default=0.15,
        help="Each epoch: randomly sample this fraction of the valid set (new sample every epoch), build k×k retrieval.",
    )
    p.add_argument(
        "--valid-sample-seed",
        type=int,
        default=None,
        help="RNG seed for valid subsample; default is sample-seed + epoch (reproducible per epoch).",
    )
    p.add_argument(
        "--valid-n-passes",
        type=int,
        default=3,
        help="Each epoch: repeat valid on independent random 15%% (or --valid-sample-frac) subsets this many times; average metrics + valid_loss_ce.",
    )
    p.add_argument("--match-scheme", type=int, default=0, choices=[0, 1, 2, 3, 4])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--diff-pre-ckpt", type=str, required=True)
    p.add_argument("--diff-post-ckpt", type=str, required=True)
    return p.parse_args()


def _load_all_pairs(max_raw_trajs: int | None):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    if max_raw_trajs is not None:
        single_traj_cmb = single_traj_cmb[: int(max_raw_trajs)]
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    return clean_pairs(mul_p + sing_p)


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    cache_path = args.cache or os.path.join(out_dir, "hydra_only_cache_84.pt")

    if args.regenerate_cache or not os.path.exists(cache_path):
        print("[run_ft_match_84] loading pairs from traj_cmb.pkl ...", flush=True)
        pairs = _load_all_pairs(args.max_raw_trajs)
        print(f"[run_ft_match_84] pairs loaded: {len(pairs)}", flush=True)
        if args.max_pairs is not None:
            pairs = pairs[: int(args.max_pairs)]
            print(f"[run_ft_match_84] pairs truncated to: {len(pairs)}", flush=True)
        pre = FixLenDiff84(device=cfg.device, lr=cfg.diff_lr)
        post = FixLenDiff84(device=cfg.device, lr=cfg.diff_lr)
        print("[run_ft_match_84] loading diffusion ckpts ...", flush=True)
        pre.unet.load_state_dict(torch.load(args.diff_pre_ckpt, map_location=cfg.device), strict=True)
        post.unet.load_state_dict(torch.load(args.diff_post_ckpt, map_location=cfg.device), strict=True)
        print("[run_ft_match_84] generating hydra-only cache ...", flush=True)
        cache = build_hydra_only_cache(
            pairs=pairs,
            pre_diff=pre,
            post_diff=post,
            hydra_tail=int(args.hydra_tail),
            batch=int(args.cache_gen_batch),
            device=cfg.device,
        )
        print(f"[run_ft_match_84] saving cache -> {cache_path}", flush=True)
        cache.save(cache_path)
        print("[run_ft_match_84] cache saved.", flush=True)

    res = run_ft_match_84(
        output_dir=out_dir,
        cache_path=cache_path,
        train_batch=int(args.train_batch),
        valid_batch=int(args.valid_batch),
        lr=float(args.lr),
        max_epochs=int(args.max_epochs),
        early_stop=int(args.early_stop),
        rl_use=int(args.rl_use),
        sample_ratio=float(args.sample_ratio),
        sample_seed=int(args.sample_seed),
        valid_sample_frac=float(args.valid_sample_frac),
        valid_sample_seed=args.valid_sample_seed,
        valid_n_passes=int(args.valid_n_passes),
        match_scheme=int(args.match_scheme),
    )
    print(
        f"[run_ft_match_84] done scheme={int(args.match_scheme)} best_ep={res.best_epoch} best_acc={res.best_acc:.4f} "
        f"best_HR5={res.best_hr5:.4f} best_HR10={res.best_hr10:.4f} best_HR20={res.best_hr20:.4f} "
        f"best_R5@10={res.best_r5_at_10:.4f} best_R10@20={res.best_r10_at_20:.4f} "
        f"best_valid_loss_ce={res.best_valid_loss_ce:.6f} "
        f"best_ckpt={res.best_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()

