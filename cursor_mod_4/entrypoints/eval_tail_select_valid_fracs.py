"""
Evaluate k×k retrieval metrics on multiple valid sampling fractions.

Two modes:
  1) With rank selector (`--rank-ckpt`): select top-k tails via JointTailRankSelector, then score with frozen MatchModel.
  2) Baseline only (`--rank-ckpt` omitted): use vanilla first-k tails for all queries.

Metrics reported per fraction (averaged over `--valid-n-passes`):
  - acc, HR5/10/20, R5@10, R10@20, valid_loss_ce
"""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
_rp = str(_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

sys.modules["cfg"] = importlib.import_module("cursor_mod_4.cfg")
import cfg  # noqa: E402

from cursor_mod_4.ft_match.hydra_cache import HydraOnlyCache  # noqa: E402
from cursor_mod_4.geo_coords import match_traj_pts_to_xy_m  # noqa: E402
from cursor_mod_4.match import MatchModel  # noqa: E402
from cursor_mod_4.proc import clean_pairs, proc_multi_single_traj  # noqa: E402
from cursor_mod_4.rl.joint_tail_rank_selector import JointTailRankSelector, RankSelectorCheckpoint  # noqa: E402
from cursor_mod_4.sample_db import sample_db  # noqa: E402


def _pair_to_info_xy8(item) -> tuple[torch.Tensor, torch.Tensor]:
    _, A, B, _, _, _, _ = item
    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    Brev = list(reversed(B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    return torch.tensor(A_xy, dtype=torch.float32), torch.tensor(B_xy, dtype=torch.float32)


def _load_pairs(sample_ratio: float, sample_seed: int):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    valid = sample_db(all_pairs, ratio=sample_ratio, seed=sample_seed)
    train = [p for p in all_pairs if p not in valid]
    return train, valid


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
    from cursor_mod_4.match.matcher import MatchInputs

    x = MatchInputs(graph=None, info_pre=info_pre, info_post=info_post, hydra_pre=hydra_pre, hydra_post=hydra_post)
    return match.match(x, pairs=True).view(-1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--match-ckpt", type=str, required=True)
    p.add_argument("--match-scheme", type=int, default=4, choices=[0, 1, 2, 3, 4])
    p.add_argument("--k-select", type=int, default=16)
    p.add_argument("--hydra-tail", type=int, default=64)
    p.add_argument("--valid-batch", type=int, default=10)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--fixed-valid-seed", type=int, default=123)
    p.add_argument("--valid-sample-fracs", type=str, default="0.02,0.05,0.10")
    p.add_argument("--device", type=str, default=None)

    # rank selector (optional)
    p.add_argument("--rank-ckpt", type=str, default=None)
    p.add_argument("--rep-dim", type=int, default=64)
    p.add_argument("--tail-mode", type=str, default="prepost_sep_fuse", choices=["pre_only", "prepost_sep_fuse"])
    return p.parse_args()


@torch.no_grad()
def _eval_one_frac(
    *,
    frac: float,
    match: MatchModel,
    rank: JointTailRankSelector | None,
    valid_pairs,
    hydra_pre_all: torch.Tensor,
    hydra_post_all: torch.Tensor,
    N: int,
    valid_batch: int,
    n_passes: int,
    fixed_seed: int,
    k_select: int,
) -> dict[str, float]:
    n_valid = len(valid_pairs)
    valid_info_pre = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
    valid_info_post = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
    for i in range(n_valid):
        ip, io = _pair_to_info_xy8(valid_pairs[i])
        valid_info_pre[i] = ip
        valid_info_post[i] = io

    valid_hydra_pre = hydra_pre_all[:n_valid]
    valid_hydra_post = hydra_post_all[:n_valid]

    k = max(1, min(n_valid, int(round(float(n_valid) * float(frac)))))
    base_seed = int(fixed_seed)

    acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = vloss_s = 0.0
    for rep in range(int(n_passes)):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(base_seed) + rep * 7919)
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
            q_info_pre = valid_info_pre[samp[i0:i1]].to(cfg.device)
            q_hydra_pre = valid_hydra_pre[samp[i0:i1]].to(cfg.device)

            B1, B2 = q_info_pre.size(0), post_info.size(0)
            i_pre = q_info_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
            i_post = post_info.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
            h_pre = q_hydra_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)
            h_post = post_hydra.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)

            ksel = int(k_select)
            if rank is None:
                h_pre_use = h_pre[:, :ksel]
                h_post_use = h_post[:, :ksel]
            else:
                h_pre_use, h_post_use, _ = rank.choose_topk(i_pre, i_post, h_pre, h_post, k_select=ksel)

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
        "acc": acc_s / n_passes,
        "HR5": hr5_s / n_passes,
        "HR10": hr10_s / n_passes,
        "HR20": hr20_s / n_passes,
        "R5@10": r51_s / n_passes,
        "R10@20": r102_s / n_passes,
        "valid_loss_ce": vloss_s / n_passes,
    }


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device

    _train_pairs, valid_pairs = _load_pairs(float(args.sample_ratio), int(args.sample_seed))

    cache = HydraOnlyCache.load(args.cache)
    N = min(int(args.hydra_tail), int(cache.hydra_pre.shape[1]))
    hydra_pre_all = cache.hydra_pre[:, :N]
    hydra_post_all = cache.hydra_post[:, :N]

    match = MatchModel(dim=128, scheme=int(args.match_scheme)).to(cfg.device)
    match.load_state_dict(torch.load(args.match_ckpt, map_location=cfg.device), strict=True)
    match.eval()
    match.freeze(True)

    rank = None
    if args.rank_ckpt is not None:
        rank = JointTailRankSelector(rep_dim=int(args.rep_dim), lr=1e-4, device=cfg.device, tail_mode=str(args.tail_mode))
        rank.load_state(RankSelectorCheckpoint.load(args.rank_ckpt))
        rank.eval()

    fracs = [float(x) for x in str(args.valid_sample_fracs).split(",") if x.strip()]
    print(
        f"[eval_valid_fracs] mode={'rank' if rank is not None else 'vanilla'} "
        f"fracs={fracs} n_passes={int(args.valid_n_passes)} fixed_seed={int(args.fixed_valid_seed)}",
        flush=True,
    )
    for frac in fracs:
        m = _eval_one_frac(
            frac=float(frac),
            match=match,
            rank=rank,
            valid_pairs=valid_pairs,
            hydra_pre_all=hydra_pre_all,
            hydra_post_all=hydra_post_all,
            N=int(N),
            valid_batch=int(args.valid_batch),
            n_passes=int(args.valid_n_passes),
            fixed_seed=int(args.fixed_valid_seed),
            k_select=int(args.k_select),
        )
        print(
            f"[eval_valid_fracs] frac={float(frac):.4f} k={int(m['k'])} "
            f"valid_loss_ce={m['valid_loss_ce']:.6f} acc={m['acc']:.4f} "
            f"HR5={m['HR5']:.4f} HR10={m['HR10']:.4f} HR20={m['HR20']:.4f} "
            f"R5@10={m['R5@10']:.4f} R10@20={m['R10@20']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

