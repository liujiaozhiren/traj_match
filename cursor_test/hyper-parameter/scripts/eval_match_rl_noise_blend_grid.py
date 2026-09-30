"""
Valid-only grid: noise–label blend α × valid subsample frac × {match_vanilla | rl_selector}.

- **match_vanilla**: frozen MatchModel, first ``--k-select`` hydra tails (same as RL training baseline).
- **rl_selector**: same frozen match + trained ``DeepLinUCBSelector`` checkpoint.

Requires ``HydraOnlyCache`` built on the **same** ordered ``--pairs-pkl`` list used for train/valid split
(``split_train_valid_indices`` with ``--sample-ratio`` / ``--sample-seed``).

Example (深圳 v2 pkl + 与该 pkl 对齐的 cache)::

    conda run --no-capture-output -n traj_match python cursor_mod_4/entrypoints/eval_match_rl_noise_blend_grid.py \\
      --pairs-pkl cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl \\
      --cache /path/to/hydra_only_cache_84.pt \\
      --match-ckpt /path/to/best_match.pth \\
      --selector-ckpt /path/to/best_selector.pt \\
      --output-json cursor_mod_run_chengdu/match_rl_noise_blend_eval.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401 — local scripts path
import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cfg

from ft_match.hydra_cache import HydraOnlyCache  # noqa: E402
from ft_match.pair_split import split_train_valid_indices  # noqa: E402
from geo_coords import match_traj_pts_to_xy_m  # noqa: E402
from match import MatchModel  # noqa: E402
from match.matcher import MatchInputs  # noqa: E402
from rl.linucb_selector import DeepLinUCBSelector, SelectorCheckpoint  # noqa: E402
from traj_noise_blend import NOISE_BLEND_LEVEL_ALPHAS, assert_pair_ts_labelfuse_aligned  # noqa: E402

def _pair_to_info_xy8(item) -> tuple[torch.Tensor, torch.Tensor]:
    _, A, B, _, _, _, _ = item
    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    Brev = list(reversed(B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    return torch.tensor(np.asarray(A_xy, dtype=np.float32)), torch.tensor(np.asarray(B_xy, dtype=np.float32))


def _pair_to_info_xy8_blend(
    item,
    noise_blend_alpha: float,
    *,
    check_label_ts: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if noise_blend_alpha >= 1.0 - 1e-12:
        return _pair_to_info_xy8(item)
    assert_pair_ts_labelfuse_aligned(item, check_label_ts_match=check_label_ts)
    a = float(noise_blend_alpha)
    _, A, B, L_A, L_B, _, _ = item
    A_noisy = np.asarray(match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len], dtype=np.float32)
    Brev = list(reversed(B))
    B_noisy = np.asarray(match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len], dtype=np.float32)
    L_A_xy = np.asarray(match_traj_pts_to_xy_m(L_A[: len(A)])[: cfg.diff_pre_len], dtype=np.float32)
    n_b = len(B)
    L_B_tail = L_B[len(L_B) - n_b :]
    L_B_xy = np.asarray(match_traj_pts_to_xy_m(list(reversed(L_B_tail)))[: cfg.diff_pre_len], dtype=np.float32)
    ip = a * A_noisy + (1.0 - a) * L_A_xy
    io = a * B_noisy + (1.0 - a) * L_B_xy
    return torch.from_numpy(ip), torch.from_numpy(io)


def _load_pairs_pkl(path: str, max_pairs: int | None):
    raw = pickle.load(open(path, "rb"))
    if not isinstance(raw, list):
        raise TypeError(f"--pairs-pkl must be a list, got {type(raw).__name__}")
    if raw and (not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7):
        raise TypeError("pairs items must be 7-tuples (mid,A,B,L_A,L_B,F_A,F_B)")
    if max_pairs is not None:
        raw = raw[: int(max_pairs)]
    return raw


def _hr_at(scores: torch.Tensor, k: int) -> float:
    b, n = scores.shape
    if b == 0 or n == 0:
        return 0.0
    kk = min(int(k), n)
    topk_idx = torch.topk(scores, k=kk, dim=1).indices
    correct = torch.arange(b, device=scores.device, dtype=topk_idx.dtype).unsqueeze(1)
    hits = (topk_idx == correct).any(dim=1).float()
    return float(hits.mean().item())


@torch.no_grad()
def _match_logits(match: MatchModel, info_pre, info_post, hydra_pre, hydra_post) -> torch.Tensor:
    x = MatchInputs(graph=None, info_pre=info_pre, info_post=info_post, hydra_pre=hydra_pre, hydra_post=hydra_post)
    return match.match(x, pairs=True).view(-1)


def _build_selector(ckpt_path: str, device: str) -> DeepLinUCBSelector:
    ck = SelectorCheckpoint.load(ckpt_path)
    meta = ck.meta if isinstance(ck.meta, dict) else {}
    sel = DeepLinUCBSelector(
        k_select=int(meta.get("k_select", 16)),
        rep_dim=int(meta.get("rep_dim", 32)),
        alpha=float(meta.get("alpha", 1.0)),
        lambda_=float(meta.get("lambda_", 1.0)),
        joint_select=bool(meta.get("joint_select", False)),
        device=device,
        lr=1e-4,
    )
    sel.load_state(ck)
    sel.freeze(True)
    return sel


@torch.no_grad()
def _eval_one_cell(
    *,
    mode: str,
    frac: float,
    noise_blend_alpha: float,
    match: MatchModel,
    selector: DeepLinUCBSelector | None,
    valid_pairs: list,
    valid_gidx: list[int],
    valid_info_pre: torch.Tensor,
    valid_info_post: torch.Tensor,
    hydra_pre_all: torch.Tensor,
    hydra_post_all: torch.Tensor,
    N: int,
    k_select: int,
    valid_batch: int,
    n_passes: int,
    cell_base_seed: int,
) -> dict[str, float]:
    n_valid = len(valid_pairs)
    vg = torch.tensor(valid_gidx, dtype=torch.long, device="cpu")
    valid_hydra_pre = hydra_pre_all[vg]
    valid_hydra_post = hydra_post_all[vg]

    k = max(1, min(n_valid, int(round(float(n_valid) * float(frac)))))
    acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = vloss_s = 0.0

    for rep in range(int(n_passes)):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(cell_base_seed) + rep * 7919)
        perm = torch.randperm(n_valid, generator=g)[:k]
        samp = perm.long()

        post_info = valid_info_post[samp].to(cfg.device)
        post_hydra = valid_hydra_post[samp].to(cfg.device)

        scores = torch.zeros((k, k), device=cfg.device)
        num = 0
        n_batches = (k + int(valid_batch) - 1) // int(valid_batch)
        for bi in range(n_batches):
            i0 = bi * int(valid_batch)
            i1 = min(i0 + int(valid_batch), k)
            b = i1 - i0
            row_idx = samp[i0:i1]
            q_info_pre = valid_info_pre[row_idx].to(cfg.device)
            q_hydra_pre = valid_hydra_pre[row_idx].to(cfg.device)

            B1, B2 = q_info_pre.size(0), post_info.size(0)
            i_pre = q_info_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
            i_post = post_info.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
            h_pre = q_hydra_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)
            h_post = post_hydra.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)

            ksel = int(k_select)
            if mode == "match_vanilla":
                h_pre_use = h_pre[:, :ksel]
                h_post_use = h_post[:, :ksel]
            elif mode == "rl_selector":
                if selector is None:
                    raise ValueError("rl_selector mode requires selector")
                h_pre_use, h_post_use, _, _ = selector.choose(i_pre, i_post, h_pre, h_post)
            else:
                raise ValueError(f"unknown mode {mode!r}")

            logits = _match_logits(match, i_pre, i_post, h_pre_use, h_post_use).view(B1, B2)
            scores[num : num + b] = logits
            num += b

        pred = scores.argmax(dim=1)
        gt = torch.arange(k, device=cfg.device)
        acc = float((pred == gt).float().mean().item())
        hr5 = _hr_at(scores, 5)
        hr10 = _hr_at(scores, 10)
        hr20 = _hr_at(scores, 20)
        r5_at_10 = hr5 / hr10 if hr10 > 1e-12 else 0.0
        r10_at_20 = hr10 / hr20 if hr20 > 1e-12 else 0.0
        vloss = float(F.cross_entropy(scores, gt.long()).item())

        acc_s += acc
        hr5_s += hr5
        hr10_s += hr10
        hr20_s += hr20
        r51_s += r5_at_10
        r102_s += r10_at_20
        vloss_s += vloss

    return {
        "k": float(k),
        "n_valid": float(n_valid),
        "acc": acc_s / n_passes,
        "hr5": hr5_s / n_passes,
        "hr10": hr10_s / n_passes,
        "hr20": hr20_s / n_passes,
        "r5_at_10": r51_s / n_passes,
        "r10_at_20": r102_s / n_passes,
        "valid_loss_ce": vloss_s / n_passes,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-pkl", type=str, required=True)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--match-ckpt", type=str, required=True, help="Frozen match weights (vanilla path; RL path unless --match-ckpt-rl)")
    p.add_argument(
        "--match-ckpt-rl",
        type=str,
        default=None,
        help="Optional: separate frozen match for RL branch if selector was trained with different match ckpt.",
    )
    p.add_argument("--match-scheme", type=int, default=4, choices=[3, 4])
    p.add_argument("--match-dim", type=int, default=None)
    p.add_argument("--selector-ckpt", type=str, default=None, help="Required for rl_selector rows")
    p.add_argument("--hydra-tail", type=int, default=64)
    p.add_argument("--k-select", type=int, default=16, help="Vanilla: use first k tails; RL meta may override from ckpt")
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-fracs", type=str, default="0.15,0.10,0.05,0.02")
    p.add_argument("--alphas", type=str, default=",".join(str(a) for a in NOISE_BLEND_LEVEL_ALPHAS))
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--eval-base-seed", type=int, default=42)
    p.add_argument("--valid-batch", type=int, default=10)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--blend-check-label-ts", action="store_true")
    p.add_argument(
        "--modes",
        type=str,
        default="match_vanilla,rl_selector",
        help="Comma-separated: match_vanilla, rl_selector",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device

    all_pairs = _load_pairs_pkl(str(args.pairs_pkl), args.max_pairs)
    train_pairs, valid_pairs, _train_gidx, valid_gidx = split_train_valid_indices(
        all_pairs, float(args.sample_ratio), int(args.sample_seed)
    )
    del train_pairs

    cache = HydraOnlyCache.load(str(args.cache))
    N = min(int(args.hydra_tail), int(cache.hydra_pre.shape[1]))
    hydra_pre_all = cache.hydra_pre[:, :N]
    hydra_post_all = cache.hydra_post[:, :N]
    n_pairs_total = len(all_pairs)
    if int(hydra_pre_all.shape[0]) != int(n_pairs_total):
        raise SystemExit(
            f"Hydra cache rows ({hydra_pre_all.shape[0]}) != pairs list length ({n_pairs_total}). "
            "Rebuild --cache with the same --pairs-pkl (and --max-pairs) as training."
        )

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    for m in modes:
        if m not in ("match_vanilla", "rl_selector"):
            raise SystemExit(f"unknown mode {m!r}")
    if "rl_selector" in modes and not args.selector_ckpt:
        raise SystemExit("--selector-ckpt is required when modes includes rl_selector")

    alphas = [float(x) for x in str(args.alphas).split(",") if x.strip()]
    fracs = [float(x) for x in str(args.valid_sample_fracs).split(",") if x.strip()]

    m_dim = int(args.match_dim if getattr(args, "match_dim", None) is not None else cfg.match_dim)
    match_v = MatchModel(dim=m_dim, scheme=int(args.match_scheme)).to(cfg.device)
    match_v.load_state_dict(torch.load(str(args.match_ckpt), map_location=cfg.device), strict=True)
    match_v.eval()
    match_v.freeze(True)

    match_rl = match_v
    if args.match_ckpt_rl:
        match_rl = MatchModel(dim=m_dim, scheme=int(args.match_scheme)).to(cfg.device)
        match_rl.load_state_dict(torch.load(str(args.match_ckpt_rl), map_location=cfg.device), strict=True)
        match_rl.eval()
        match_rl.freeze(True)

    selector = _build_selector(str(args.selector_ckpt), cfg.device) if args.selector_ckpt else None
    k_sel_meta = int(selector.k_select) if selector is not None else int(args.k_select)

    meta = {
        "pairs_pkl": os.path.abspath(str(args.pairs_pkl)),
        "cache": os.path.abspath(str(args.cache)),
        "match_ckpt": os.path.abspath(str(args.match_ckpt)),
        "match_ckpt_rl": os.path.abspath(str(args.match_ckpt_rl)) if args.match_ckpt_rl else None,
        "selector_ckpt": os.path.abspath(str(args.selector_ckpt)) if args.selector_ckpt else None,
        "match_scheme": int(args.match_scheme),
        "sample_ratio": float(args.sample_ratio),
        "sample_seed": int(args.sample_seed),
        "n_train": len(all_pairs) - len(valid_pairs),
        "n_valid": len(valid_pairs),
        "hydra_tail_used": int(N),
        "k_select_arg": int(args.k_select),
        "k_select_effective": k_sel_meta,
        "alphas": alphas,
        "fracs": fracs,
        "valid_n_passes": int(args.valid_n_passes),
        "eval_base_seed": int(args.eval_base_seed),
        "modes": modes,
        "blend_check_label_ts": bool(args.blend_check_label_ts),
    }
    print(f"[match_rl_noise_blend] meta={json.dumps(meta, indent=2)}", flush=True)

    rows: list[dict] = []
    cell_i = 0
    total = len(alphas) * len(fracs) * len(modes)
    for ai, alpha in enumerate(alphas):
        n_valid = len(valid_pairs)
        valid_info_pre = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
        valid_info_post = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
        for i in range(n_valid):
            if alpha >= 1.0 - 1e-12:
                ip, io = _pair_to_info_xy8(valid_pairs[i])
            else:
                ip, io = _pair_to_info_xy8_blend(
                    valid_pairs[i], float(alpha), check_label_ts=bool(args.blend_check_label_ts)
                )
            valid_info_pre[i] = ip
            valid_info_post[i] = io

        for fi, frac in enumerate(fracs):
            for mode in modes:
                cell_i += 1
                mdl = match_v if mode == "match_vanilla" else match_rl
                sel = None if mode == "match_vanilla" else selector
                cell_seed = int(args.eval_base_seed) + ai * 100_003 + fi * 17_011 + (0 if mode == "match_vanilla" else 90_001)
                m = _eval_one_cell(
                    mode=mode,
                    frac=float(frac),
                    noise_blend_alpha=float(alpha),
                    match=mdl,
                    selector=sel,
                    valid_pairs=valid_pairs,
                    valid_gidx=valid_gidx,
                    valid_info_pre=valid_info_pre,
                    valid_info_post=valid_info_post,
                    hydra_pre_all=hydra_pre_all,
                    hydra_post_all=hydra_post_all,
                    N=int(N),
                    k_select=k_sel_meta,
                    valid_batch=int(args.valid_batch),
                    n_passes=int(args.valid_n_passes),
                    cell_base_seed=cell_seed,
                )
                row = {
                    "method": mode,
                    "noise_blend_alpha": float(alpha),
                    "valid_sample_frac": float(frac),
                    "cell_base_seed": int(cell_seed),
                    **m,
                }
                rows.append(row)
                print(
                    f"[match_rl_noise_blend] ({cell_i}/{total}) {mode} α={alpha:g} frac={frac:g} "
                    f"k={int(m['k'])}/{int(m['n_valid'])} acc={m['acc']:.4f} HR5={m['hr5']:.4f} "
                    f"HR10={m['hr10']:.4f} HR20={m['hr20']:.4f} CE={m['valid_loss_ce']:.4f}",
                    flush=True,
                )

    out = {"meta": meta, "rows": rows}
    os.makedirs(os.path.dirname(os.path.abspath(str(args.output_json))) or ".", exist_ok=True)
    with open(str(args.output_json), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[match_rl_noise_blend] wrote {args.output_json} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
