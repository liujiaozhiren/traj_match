"""
Hyper-parameter sweep definitions (8 sweeps × 5 tiers). See ../README.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STAR = {
    "T": 200,
    "unet_dim": 512,
    "num_res_blocks": 2,
    "lambda_x0": 0.0,
    "hydra_tail": 64,
    "k_select": 16,
    "match_dim": 128,
    "rl_lr": 1e-4,
}

SWEEP_ORDER = [
    "08_rl_lr",
    "06_k_select",
    "07_match_dim",
    "05_hydra_tail",
    "04_lambda_x0",
    "01_diffusion_timestamp",
    "02_unet_dim",
    "03_num_res_blocks",
]

SWEEPS: dict[str, dict[str, Any]] = {
    "08_rl_lr": {
        "param": "rl_lr",
        "tiers": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
        "star_tier": 3,
        "train": ["rl"],
    },
    "06_k_select": {
        "param": "k_select",
        "tiers": [4, 8, 16, 24, 32],
        "star_tier": 3,
        "train": ["match", "rl"],
    },
    "07_match_dim": {
        "param": "match_dim",
        "tiers": [64, 96, 128, 192, 256],
        "star_tier": 3,
        "train": ["match"],
        "skip_eval": True,
    },
    "05_hydra_tail": {
        "param": "hydra_tail",
        "tiers": [32, 48, 64, 96, 128],
        "star_tier": 3,
        "train": ["cache", "match", "rl"],
        # hydra_tail = number of i.i.d. candidate tails sampled from the (fixed
        # baseline) diffusion model. match uses cache[:, :rl_use], RL uses
        # cache[:, :hydra_tail], both prefixes -> one 128-candidate master cache
        # serves every tier (each just passes --hydra-tail K). Skips 4 separate
        # multi-hour cache regenerations.
        "cache_master": "cursor_test/hyper-parameter/scripts/registry/hydra_master_tail128.pt",
    },
    "04_lambda_x0": {
        "param": "lambda_x0",
        "tiers": [0.0, 0.05, 0.1, 0.25, 0.5],
        "star_tier": 1,
        # lambda_x0 is a DIFFUSION-loss weight (loss = eps_mse + lambda*x0_recon_mse).
        # Its effect is read directly from the diffusion validation RMSE (logged each
        # epoch + in best_*.pth filenames), so we stop at diff and skip the expensive
        # cache/match/rl/eval chain.
        "train": ["diff"],
        "skip_eval": True,
        # Shenzhen diffusion pretrain ckpt is missing on disk; continue
        # fine-tuning from the baseline's own fine-tuned diff ckpts instead.
        "diff_init": "baseline",
    },
    "01_diffusion_timestamp": {
        "param": "T",
        "tiers": [25, 50, 100, 200, 400],
        "star_tier": 4,
        # T (noise-schedule length) is a diffusion hyperparameter; its effect is read
        # from the diffusion validation RMSE -> diff-only, skip cache/match/rl/eval.
        "train": ["diff"],
        "skip_eval": True,
        # Only the noise schedule length (T) changes; UNet arch is identical to
        # baseline, so continue fine-tuning from baseline diff ckpts (pretrain ckpt
        # is missing on disk).
        "diff_init": "baseline",
    },
    "02_unet_dim": {
        "param": "unet_dim",
        "tiers": [128, 256, 384, 512, 768],
        "star_tier": 4,
        # unet_dim changes the diffusion UNet architecture; compare via diffusion
        # validation RMSE -> diff-only (arch differs from baseline -> trains from scratch).
        "train": ["diff"],
        "skip_eval": True,
    },
    "03_num_res_blocks": {
        "param": "num_res_blocks",
        "tiers": [1, 2, 3, 4, 5],
        "star_tier": 2,
        # num_res_blocks changes the diffusion UNet architecture; compare via diffusion
        # validation RMSE -> diff-only (arch differs from baseline -> trains from scratch).
        "train": ["diff"],
        "skip_eval": True,
    },
}


@dataclass(frozen=True)
class TierConfig:
    sweep_id: str
    tier: int
    values: dict[str, Any]
    is_star: bool
    train_stages: list[str]
    skip_eval: bool
    diff_init: str
    cache_master: str | None = None


def tier_config(sweep_id: str, tier: int) -> TierConfig:
    if sweep_id not in SWEEPS:
        raise KeyError(f"unknown sweep {sweep_id!r}")
    spec = SWEEPS[sweep_id]
    if not (1 <= tier <= 5):
        raise ValueError(f"tier must be 1..5, got {tier}")
    values = dict(STAR)
    param = spec["param"]
    val = spec["tiers"][tier - 1]
    values[param] = val
    if param == "k_select":
        values["rl_use"] = val
    is_star = tier == int(spec["star_tier"])
    train = [] if is_star else list(spec["train"])
    return TierConfig(
        sweep_id=sweep_id,
        tier=tier,
        values=values,
        is_star=is_star,
        train_stages=train,
        skip_eval=bool(spec.get("skip_eval")),
        diff_init=str(spec.get("diff_init", "pretrain")),
        cache_master=spec.get("cache_master"),
    )


def tier_out_dir(sweep_id: str, tier: int, out_root: str) -> str:
    import os

    spec = SWEEPS[sweep_id]
    param = spec["param"]
    val = spec["tiers"][tier - 1]
    if isinstance(val, float) and val < 0.01:
        tag = f"{val:.0e}".replace("+", "")
    else:
        tag = str(val).replace(".", "p")
    return os.path.join(out_root, f"sweep_{sweep_id}", f"tier_{tier}_{param}_{tag}")
