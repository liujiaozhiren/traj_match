#!/usr/bin/env python3
"""
Run one (sweep, tier) hyper experiment: train missing stages + valid grid.

All logic self-contained under scripts/; reads pre-trained ★ artifacts from registry/baseline_star.json.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import cfg
from hyper_config import SWEEP_ORDER, tier_config, tier_out_dir

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parents[2]


def _abspath(rel: str) -> str:
    p = Path(rel)
    return str(p if p.is_absolute() else _REPO / rel)


def _load_baseline() -> dict:
    path = _SCRIPTS / "registry" / "baseline_star.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run(cmd: list[str], *, log_path: str | None = None):
    print("[run_tier] $", " ".join(cmd), flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write("\n=== " + " ".join(cmd) + " ===\n")
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
    else:
        proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _pick_best(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def _match_ft_batch_args(values: dict, *, need_cache: bool) -> list[str]:
    """Smaller train/valid batches for long hydra/k sequences to avoid OOM on A40.
    Cache generation only runs the diffusion model (small), so use a large batch
    to keep the GPU busy and cut wall-clock time."""
    k = int(values["k_select"])
    tail = int(values["hydra_tail"])
    train_b, valid_b, cache_b = 32, 10, 128
    if k >= 32:
        train_b, valid_b = 4, 2
    elif k >= 24:
        train_b, valid_b = 8, 4
    if tail >= 128:
        train_b = min(train_b, 8)
        valid_b = min(valid_b, 4)
        cache_b = 96
    elif tail >= 96:
        train_b = min(train_b, 16)
        valid_b = min(valid_b, 6)
        cache_b = 112
    out = [
        "--train-batch",
        str(train_b),
        "--valid-batch",
        str(valid_b),
    ]
    if need_cache:
        out += ["--cache-gen-batch", str(cache_b)]
    return out


def _apply_cfg_values(values: dict):
    cfg.apply_hyper(
        T=int(values["T"]),
        unet_dim=int(values["unet_dim"]),
        res_blocks=int(values["num_res_blocks"]),
        lam_x0=float(values["lambda_x0"]),
        tail=int(values["hydra_tail"]),
        k=int(values["k_select"]),
        m_dim=int(values["match_dim"]),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", type=str, required=True, choices=SWEEP_ORDER)
    p.add_argument("--tier", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--out-root", type=str, default=cfg.hyper_out_root)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--pairs-pkl", type=str, default=None)
    p.add_argument("--valid-only", action="store_true", help="Skip training; only run eval grid")
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def main():
    args = parse_args()
    tc = tier_config(args.sweep, args.tier)
    baseline = _load_baseline()
    pairs_pkl = _abspath(args.pairs_pkl or baseline["pairs_pkl"])
    out_dir = tier_out_dir(tc.sweep_id, tc.tier, args.out_root)
    os.makedirs(out_dir, exist_ok=True)
    log = os.path.join(out_dir, "run.log")

    _apply_cfg_values(tc.values)
    if args.device:
        cfg.device = args.device

    # Propagate per-tier hyper-parameters to child processes (diff FT, cache/match,
    # rl, eval) via env vars; cfg.py reads HYPER_* at import time. Critical for the
    # arch/schedule sweeps (T, unet_dim, num_res_blocks) which have no CLI flags.
    os.environ["HYPER_T"] = str(int(tc.values["T"]))
    os.environ["HYPER_UNET_DIM"] = str(int(tc.values["unet_dim"]))
    os.environ["HYPER_RES_BLOCKS"] = str(int(tc.values["num_res_blocks"]))
    os.environ["HYPER_HYDRA_TAIL"] = str(int(tc.values["hydra_tail"]))
    os.environ["HYPER_RL_USE"] = str(int(tc.values["k_select"]))
    os.environ["HYPER_MATCH_DIM"] = str(int(tc.values["match_dim"]))
    os.environ["HYPER_LAMBDA_X0"] = str(float(tc.values["lambda_x0"]))

    art = {k: _abspath(v) for k, v in baseline["artifacts"].items()}
    meta_path = os.path.join(out_dir, "tier_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "sweep": tc.sweep_id,
                "tier": tc.tier,
                "is_star": tc.is_star,
                "hyper": tc.values,
                "train_stages": tc.train_stages,
                "pairs_pkl": pairs_pkl,
                "out_dir": out_dir,
            },
            f,
            indent=2,
        )

    py = args.python
    scripts = str(_SCRIPTS)

    diff_pre = art["diff_pre"]
    diff_post = art["diff_post"]
    cache_path = art["cache"]
    match_ckpt = art["match"]
    selector_ckpt = art["rl_selector"]
    rl_lr = float(tc.values["rl_lr"])

    # --- train stages ---
    if not args.valid_only and not tc.is_star:
        if "diff" in tc.train_stages:
            diff_out = os.path.join(out_dir, "diff_ft")
            arch_matches_baseline = (
                int(tc.values["unet_dim"]) == 512 and int(tc.values["num_res_blocks"]) == 2
            )
            if tc.diff_init == "baseline":
                # Continue fine-tuning from baseline's own fine-tuned diff ckpts
                # (used when the pretrain ckpt is unavailable and arch matches baseline).
                ckpt_pre = art["diff_pre"]
                ckpt_post = art["diff_post"]
            elif (
                arch_matches_baseline
                and os.path.isfile(art["diff_pretrain_pre"])
                and os.path.isfile(art["diff_pretrain_post"])
            ):
                ckpt_pre = art["diff_pretrain_pre"]
                ckpt_post = art["diff_pretrain_post"]
            else:
                # Arch differs from baseline (or pretrain ckpt missing) -> train from scratch.
                ckpt_pre = None
                ckpt_post = None
            from_scratch = not (ckpt_pre and ckpt_post)
            # 1e-6 suits continued fine-tuning; from-scratch needs a higher lr.
            diff_lr = 1e-4 if from_scratch else cfg.diff_lr
            cmd = [
                py,
                os.path.join(scripts, "ft_diffusion_on_match_pairs.py"),
                "--pairs-pkl",
                pairs_pkl,
                "--out-dir",
                diff_out,
                "--epochs",
                "300",
                "--batch",
                "32",
                "--lr",
                str(diff_lr),
                "--sample-ratio",
                "0.1",
                "--sample-seed",
                "42",
                "--lambda-x0",
                str(tc.values["lambda_x0"]),
            ]
            if not from_scratch:
                cmd += ["--ckpt-pre", ckpt_pre, "--ckpt-post", ckpt_post]
            else:
                cmd.append("--from-scratch")
            if args.device:
                cmd += ["--device", args.device]
            _run(cmd, log_path=log)
            diff_pre = _pick_best(os.path.join(diff_out, "best_pre*.pth")) or os.path.join(diff_out, "best_pre.pth")
            diff_post = _pick_best(os.path.join(diff_out, "best_post*.pth")) or os.path.join(diff_out, "best_post.pth")

        need_cache = "cache" in tc.train_stages
        # Reuse a prebuilt master hydra cache instead of regenerating per tier.
        # The cache stores `hydra_tail` i.i.d. candidate tails from the SAME (fixed
        # baseline) diffusion model; downstream match slices [:, :rl_use] and RL
        # slices [:, :min(hydra_tail, n_candidates)]. So a 128-candidate master is a
        # valid tier-K cache for any K<=128 — just pass --hydra-tail K. Saves 4
        # multi-hour cache regenerations for the hydra_tail sweep.
        if need_cache and tc.cache_master:
            master = _abspath(tc.cache_master)
            if os.path.isfile(master):
                print(f"[run_tier] reuse master cache (skip regenerate): {master}", flush=True)
                need_cache = False
                cache_path = master
            else:
                print(f"[run_tier] master cache missing, will regenerate: {master}", flush=True)
        need_match = "match" in tc.train_stages
        if need_cache or need_match:
            match_out = os.path.join(out_dir, "match_ft")
            existing_match = _pick_best(os.path.join(match_out, "best_match*.pth"))
            skip_match_run = bool(need_match and existing_match and not need_cache)

            if skip_match_run:
                print(f"[run_tier] skip match train; reuse {existing_match}", flush=True)
                match_ckpt = existing_match
            else:
                cmd = [
                    py,
                    os.path.join(scripts, "run_ft_match.py"),
                    "--pairs-pkl",
                    pairs_pkl,
                    "--output-dir",
                    match_out,
                    "--diff-pre-ckpt",
                    diff_pre,
                    "--diff-post-ckpt",
                    diff_post,
                    "--hydra-tail",
                    str(int(tc.values["hydra_tail"])),
                    "--rl-use",
                    str(int(tc.values["k_select"])),
                    "--match-dim",
                    str(int(tc.values["match_dim"])),
                    "--train-loss-mode",
                    "bb_ce",
                ]
                if need_cache:
                    cmd.append("--regenerate-cache")
                else:
                    cmd += ["--cache", cache_path]
                if not need_match:
                    cmd.append("--skip-train")
                cmd += _match_ft_batch_args(tc.values, need_cache=need_cache)
                if args.device:
                    cmd += ["--device", args.device]
                _run(cmd, log_path=log)
                if need_match:
                    match_ckpt = _pick_best(os.path.join(match_out, "best_match*.pth"))
                    if not match_ckpt:
                        raise SystemExit(f"no match ckpt under {match_out}")

            if need_cache:
                local_cache = os.path.join(match_out, f"hydra_only_cache_tail{int(tc.values['hydra_tail'])}.pt")
                if os.path.isfile(local_cache):
                    cache_path = local_cache
                else:
                    raise SystemExit(f"expected regenerated cache missing: {local_cache}")
            # else: reuse external baseline cache (e.g. registry/hydra_only_cache_84.pt)

        if "rl" in tc.train_stages:
            rl_out = os.path.join(out_dir, "rl")
            cmd = [
                py,
                os.path.join(scripts, "run_rl_tail_select.py"),
                "--output-dir",
                rl_out,
                "--pairs-pkl",
                pairs_pkl,
                "--cache",
                cache_path,
                "--match-ckpt",
                match_ckpt,
                "--match-dim",
                str(int(tc.values["match_dim"])),
                "--hydra-tail",
                str(int(tc.values["hydra_tail"])),
                "--k-select",
                str(int(tc.values["k_select"])),
                "--lr",
                str(rl_lr),
                "--train-reward-mode",
                "bb_ce",
                "--fixed-valid-seed",
                "42",
            ]
            if args.device:
                cmd += ["--device", args.device]
            _run(cmd, log_path=log)
            selector_ckpt = _pick_best(os.path.join(rl_out, "best_selector*.pt"))
            if not selector_ckpt:
                raise SystemExit(f"no selector ckpt under {rl_out}")

    elif tc.is_star:
        print(f"[run_tier] ★ tier — reuse baseline artifacts (train skipped)", flush=True)

    if tc.skip_eval:
        print(f"[run_tier] skip eval grid (match-only sweep)", flush=True)
        print(f"[run_tier] done out_dir={out_dir}", flush=True)
        return

    # --- valid ---
    grid_path = os.path.join(out_dir, "grid.json")
    cmd = [
        py,
        os.path.join(scripts, "eval_match_rl_noise_blend_grid.py"),
        "--pairs-pkl",
        pairs_pkl,
        "--cache",
        cache_path,
        "--match-ckpt",
        match_ckpt,
        "--selector-ckpt",
        selector_ckpt,
        "--hydra-tail",
        str(int(tc.values["hydra_tail"])),
        "--k-select",
        str(int(tc.values["k_select"])),
        "--match-dim",
        str(int(tc.values["match_dim"])),
        "--alphas",
        "1.0",
        "--valid-sample-fracs",
        "0.15,0.10,0.05,0.02",
        "--output-json",
        grid_path,
    ]
    if args.device:
        cmd += ["--device", args.device]
    _run(cmd, log_path=log)
    print(f"[run_tier] done out_dir={out_dir} grid={grid_path}", flush=True)


if __name__ == "__main__":
    main()
