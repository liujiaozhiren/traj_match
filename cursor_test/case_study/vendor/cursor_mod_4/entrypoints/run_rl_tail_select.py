"""
RL (bandit) tail selection entrypoint for cursor_mod_4.

Goal:
  - Load a trained MatchModel checkpoint (e.g. scheme4) and freeze it.
  - Train a DeepLinUCBSelector to pick k_select tails out of hydra_tail candidates (default 16 out of 64).
  - Save selector checkpoint separately (decoupled from match scheme/weights).

Match fine-tuning may use `--train-loss-mode bb_ce` on scheme 3/4. This script loads frozen Match weights.
`--train-reward-mode bb_ce` uses the same B×B batch CE (baseline vs selected tails) as the **positive**
selector reward; negatives still use the original BCE-based advantage. Default remains `bce`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
_rp = str(_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

sys.modules["cfg"] = importlib.import_module("cursor_mod_4.cfg")
import cfg  # noqa: E402

from cursor_mod_4.geo_coords import match_traj_pts_to_xy_m  # noqa: E402
from cursor_mod_4.ft_match.hydra_cache import HydraOnlyCache  # noqa: E402
from cursor_mod_4.ft_match.pair_split import split_train_valid_indices  # noqa: E402
from cursor_mod_4.match import MatchModel, build_match_inputs  # noqa: E402
from cursor_mod_4.proc import clean_pairs, proc_multi_single_traj  # noqa: E402
from cursor_mod_4.rl.linucb_selector import DeepLinUCBSelector, SelectorCheckpoint, bce_loss_each  # noqa: E402


class PairIdxDataset(Dataset):
    """Returns (global_cache_idx, pair) so `HydraOnlyCache` rows align with the pair list used at cache build."""

    def __init__(self, pairs, global_indices: list[int] | None = None, *, max_len: int | None = None):
        self.pairs = list(pairs)
        if global_indices is None:
            self.gidx = list(range(len(self.pairs)))
        else:
            self.gidx = [int(x) for x in global_indices]
            if len(self.gidx) != len(self.pairs):
                raise ValueError("pairs and global_indices length mismatch")
        if max_len is not None:
            self.pairs = self.pairs[: int(max_len)]
            self.gidx = self.gidx[: int(max_len)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int):
        return self.gidx[i], self.pairs[i]


def _pair_to_info_xy8(item) -> tuple[torch.Tensor, torch.Tensor]:
    _, A, B, _, _, _, _ = item
    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    Brev = list(reversed(B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    return (
        torch.tensor(A_xy, dtype=torch.float32),  # (8,2)
        torch.tensor(B_xy, dtype=torch.float32),
    )


def _collate_idx_info(batch):
    idxs, pre, post = [], [], []
    for idx, item in batch:
        ip, io = _pair_to_info_xy8(item)
        idxs.append(int(idx))
        pre.append(ip)
        post.append(io)
    return torch.tensor(idxs, dtype=torch.long), torch.stack(pre, 0), torch.stack(post, 0)


def _load_pairs_traj_cmb(sample_ratio: float, sample_seed: int):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    return split_train_valid_indices(all_pairs, sample_ratio, sample_seed)


def _load_pairs_pkl(pkl_path: str, sample_ratio: float, sample_seed: int, *, max_pairs: int | None = None):
    raw = pickle.load(open(pkl_path, "rb"))
    if not isinstance(raw, list):
        raise TypeError(f"pairs_pkl must be a list, got {type(raw).__name__}")
    if raw and (not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7):
        raise TypeError(f"pairs_pkl items must be 7-tuples; first len={len(raw[0]) if raw else 'n/a'}")
    if max_pairs is not None:
        raw = raw[: int(max_pairs)]
    return split_train_valid_indices(raw, sample_ratio, sample_seed)


def _derangement_indices(n: int, device: torch.device):
    if n == 1:
        raise ValueError("batch size=1 无法构造负样本（derangement）。")
    idx = torch.arange(n, device=device)
    shift = torch.randint(1, n, (1,), device=device).item()
    return (idx + shift) % n


def _hr_at(scores: torch.Tensor, k: int) -> float:
    """
    Hit rate @k: each row's ground-truth column index is i; success if col i is in top-k by score.
    scores: (B, B) with correct label on diagonal indices 0..B-1.
    """
    b, n = scores.shape
    if b == 0 or n == 0:
        return 0.0
    kk = min(int(k), n)
    topk_idx = torch.topk(scores, k=kk, dim=1).indices
    correct = torch.arange(b, device=scores.device, dtype=topk_idx.dtype).unsqueeze(1)
    hits = (topk_idx == correct).any(dim=1).float()
    return float(hits.mean().item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--pairs-pkl",
        type=str,
        default=None,
        help="If set, load pairs from this pickle (7-tuples) and split by index like run_ft_match_84; "
        "must match the pair order used to build --cache.",
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Truncate pairs list (after load) before train/valid split; must match cache build.",
    )
    p.add_argument("--cache", type=str, required=True, help="HydraOnlyCache path (must contain hydra_tail candidates)")
    p.add_argument("--match-ckpt", type=str, required=True, help="Trained match checkpoint (scheme-dependent)")
    p.add_argument("--match-scheme", type=int, default=4, choices=[0, 1, 2, 3, 4])
    p.add_argument("--k-select", type=int, default=16, help="How many tails to select from hydra_tail candidates")
    p.add_argument("--hydra-tail", type=int, default=64, help="How many candidates exist in cache (use first this many)")
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--early-stop", type=int, default=15)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-frac", type=float, default=0.15)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument(
        "--fixed-valid-seed",
        type=int,
        default=None,
        help="If set, use a fixed RNG seed for valid subsampling across all epochs (stabilizes valid metrics).",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    p.add_argument("--rep-dim", type=int, default=32)
    p.add_argument(
        "--joint-select",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, selector chooses the SAME candidate indices for pre/post (idx_pre==idx_post).",
    )
    p.add_argument(
        "--reward-zscore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If enabled, z-score normalize per-arm reward within each batch before update.",
    )
    p.add_argument(
        "--reward-clip",
        type=float,
        default=3.0,
        help="Clip normalized per-arm reward to [-reward-clip, reward-clip].",
    )
    p.add_argument(
        "--train-reward-mode",
        type=str,
        default="bce",
        choices=["bce", "bb_ce"],
        help="bce: per-arm BCE advantage (default). bb_ce: positive batch reward from (CE_base - CE_sel) on B×B "
        "retrieval (scheme 3/4 only); negatives still use BCE advantage.",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--selector-ckpt", type=str, default=None, help="Optional: resume selector checkpoint")
    return p.parse_args()


@torch.no_grad()
def _match_logits(match: MatchModel, info_pre, info_post, hydra_pre, hydra_post) -> torch.Tensor:
    # scheme3/4 path: match.match expects MatchInputs. We'll use the internal tuple directly.
    from cursor_mod_4.match.matcher import MatchInputs

    x = MatchInputs(graph=None, info_pre=info_pre, info_post=info_post, hydra_pre=hydra_pre, hydra_post=hydra_post)
    return match.match(x, pairs=True).view(-1)


@torch.no_grad()
def _bb_retrieval_ce(
    match: MatchModel,
    scheme: int,
    info_pre: torch.Tensor,
    info_post: torch.Tensor,
    hydra_pre: torch.Tensor,
    hydra_post: torch.Tensor,
) -> torch.Tensor:
    """Same B×B construction as FT `bb_ce` / valid: CE(logits, arange(B))."""
    B = int(info_pre.size(0))
    x = build_match_inputs(
        scheme=int(scheme),
        trajA_head=info_pre,
        trajB_head=info_post,
        trajA_tail=hydra_pre,
        trajB_tail=hydra_post,
        pair=True,
        valid=True,
    )
    logits = match.match(x, pairs=True).view(B, B).float()
    return F.cross_entropy(logits, torch.arange(B, device=logits.device, dtype=torch.long))


@torch.no_grad()
def _bce_k1_each_arm(
    *,
    match: MatchModel,
    info_pre: torch.Tensor,  # (B, L, 2)
    info_post: torch.Tensor,  # (B, L, 2)
    hydra_pre: torch.Tensor,  # (B, K, L2, 2)
    hydra_post: torch.Tensor,  # (B, K, L2, 2)
    label01: float,
) -> torch.Tensor:
    """
    Compute BCE-with-logits loss for each arm using k=1 (single tail per arm).
    Returns: (B*K,) float tensor on cfg.device.
    """
    B, K, L2, Dc = hydra_pre.shape
    assert hydra_post.shape == (B, K, L2, Dc)
    i_pre = info_pre.repeat_interleave(K, dim=0)
    i_post = info_post.repeat_interleave(K, dim=0)
    h_pre = hydra_pre.contiguous().view(B * K, 1, L2, Dc)
    h_post = hydra_post.contiguous().view(B * K, 1, L2, Dc)
    logits = _match_logits(match, i_pre, i_post, h_pre, h_post).view(-1)
    labels = torch.full((B * K,), float(label01), device=logits.device, dtype=torch.float32)
    return bce_loss_each(logits, labels)


def _normalize_reward(r: torch.Tensor, *, zscore: bool, clip: float) -> torch.Tensor:
    """
    r: (B*k,) reward tensor on cfg.device.
    Returns: normalized reward tensor same shape/device.
    """
    out = r
    if zscore:
        mu = out.mean()
        var = (out - mu).pow(2).mean()
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        out = (out - mu) / std
    c = float(clip)
    if c > 0:
        out = torch.clamp(out, -c, c)
    return out


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device

    trm = str(args.train_reward_mode).lower()
    if trm not in ("bce", "bb_ce"):
        raise ValueError(f"train_reward_mode must be bce or bb_ce, got {args.train_reward_mode!r}")
    if trm == "bb_ce" and int(args.match_scheme) not in (3, 4):
        raise ValueError("train_reward_mode=bb_ce requires match_scheme 3 or 4 (tensor B×B path).")

    os.makedirs(args.output_dir, exist_ok=True)

    # data (global cache indices align with HydraOnlyCache row order = full pair list before split)
    if args.pairs_pkl:
        train_pairs, valid_pairs, train_gidx, valid_gidx = _load_pairs_pkl(
            str(args.pairs_pkl), float(args.sample_ratio), int(args.sample_seed), max_pairs=args.max_pairs
        )
        print(f"[rl_tail_select] pairs source=pkl: {os.path.abspath(args.pairs_pkl)}", flush=True)
    else:
        train_pairs, valid_pairs, train_gidx, valid_gidx = _load_pairs_traj_cmb(float(args.sample_ratio), int(args.sample_seed))
    train_ds = PairIdxDataset(train_pairs, train_gidx)
    valid_ds = PairIdxDataset(valid_pairs, valid_gidx)
    train_loader = DataLoader(train_ds, batch_size=int(args.train_batch), shuffle=True, num_workers=0, pin_memory=False, collate_fn=_collate_idx_info)

    # cache (CPU)
    cache = HydraOnlyCache.load(args.cache)
    N = min(int(args.hydra_tail), int(cache.hydra_pre.shape[1]))
    # keep all N candidates (for RL), do NOT slice to first k_select
    hydra_pre_all = cache.hydra_pre[:, :N]  # (Npairs, N, 4, 2)
    hydra_post_all = cache.hydra_post[:, :N]

    n_pairs_total = len(train_pairs) + len(valid_pairs)
    if int(hydra_pre_all.shape[0]) != int(n_pairs_total):
        raise ValueError(
            f"Hydra cache rows ({hydra_pre_all.shape[0]}) != train+valid pair count ({n_pairs_total}); "
            "rebuild cache or pass matching --pairs-pkl/--max-pairs as cache build."
        )

    # match model (frozen)
    match = MatchModel(dim=128, scheme=int(args.match_scheme)).to(cfg.device)
    sd = torch.load(args.match_ckpt, map_location=cfg.device)
    match.load_state_dict(sd, strict=True)
    match.eval()
    match.freeze(True)

    # selector (trainable)
    selector = DeepLinUCBSelector(
        k_select=int(args.k_select),
        rep_dim=int(args.rep_dim),
        alpha=float(args.alpha),
        lambda_=float(args.lambda_),
        joint_select=bool(args.joint_select),
        device=cfg.device,
        lr=float(args.lr),
    )
    if args.selector_ckpt is not None:
        ck = SelectorCheckpoint.load(args.selector_ckpt)
        selector.load_state(ck)

    best_acc = -1.0
    best_ckpt = None
    best_ep = None
    stall = 0
    # best metrics (RL selector)
    best_hr5 = best_hr10 = best_hr20 = 0.0
    best_r5_at_10 = best_r10_at_20 = 0.0
    best_valid_loss_ce = float("inf")
    # vanilla16 metrics at the same best_ep (for direct comparison)
    best_acc_vanilla16 = 0.0
    best_hr5_vanilla16 = best_hr10_vanilla16 = best_hr20_vanilla16 = 0.0
    best_r5_at_10_vanilla16 = best_r10_at_20_vanilla16 = 0.0
    best_valid_loss_ce_vanilla16 = float("inf")

    print(
        f"[rl_tail_select] train_reward_mode={trm} match_scheme={int(args.match_scheme)} "
        f"train_pairs={len(train_pairs)} valid_pairs={len(valid_pairs)} cache_rows={int(hydra_pre_all.shape[0])}",
        flush=True,
    )

    for ep in range(1, int(args.max_epochs) + 1):
        selector.freeze(False)
        losses = []
        # epoch-level diagnostics (BCE + advantage)
        pos_base_bce_s = 0.0
        neg_base_bce_s = 0.0
        pos_sel_bce_s = 0.0
        neg_sel_bce_s = 0.0
        adv_pos_s = 0.0
        adv_neg_s = 0.0
        adv_pos_sq_s = 0.0
        adv_neg_sq_s = 0.0
        adv_pos_poscnt = 0
        adv_neg_poscnt = 0
        n_pos = 0
        n_neg = 0
        pos_bb_base_s = 0.0
        pos_bb_sel_s = 0.0
        pos_bb_delta_s = 0.0
        bb_pos_n = 0
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[rl_tail_select] train ep{ep}/{args.max_epochs}", mininterval=0.5)
        for idxs, info_pre, info_post in pbar:
            B = info_pre.size(0)
            if B <= 1:
                continue
            info_pre = info_pre.to(cfg.device)
            info_post = info_post.to(cfg.device)
            hydra_pre = hydra_pre_all[idxs].to(cfg.device)
            hydra_post = hydra_post_all[idxs].to(cfg.device)

            # neg pairs by derangement on post side
            perm = _derangement_indices(B, device=info_pre.device)
            neg_info_pre = info_pre
            neg_info_post = info_post[perm]
            neg_hydra_pre = hydra_pre
            neg_hydra_post = hydra_post[perm]

            label_pos = torch.ones(B, device=cfg.device)
            label_neg = torch.zeros(B, device=cfg.device)

            # baseline: first k tails
            ksel = int(args.k_select)
            hydra_pre_base = hydra_pre[:, :ksel]
            hydra_post_base = hydra_post[:, :ksel]
            neg_hydra_pre_base = neg_hydra_pre[:, :ksel]
            neg_hydra_post_base = neg_hydra_post[:, :ksel]

            with torch.no_grad():
                logits_pos_base = _match_logits(match, info_pre, info_post, hydra_pre_base, hydra_post_base)
                logits_neg_base = _match_logits(match, neg_info_pre, neg_info_post, neg_hydra_pre_base, neg_hydra_post_base)
                loss_pos_base_each = bce_loss_each(logits_pos_base, label_pos)  # (B,)
                loss_neg_base_each = bce_loss_each(logits_neg_base, label_neg)

            # choose + reward + update for pos
            selector.freeze(False)
            hydra_pre_sel, hydra_post_sel, _, _ = selector.choose(info_pre, info_post, hydra_pre, hydra_post)
            with torch.no_grad():
                logits_pos_sel = _match_logits(match, info_pre, info_post, hydra_pre_sel, hydra_post_sel)
                loss_pos_sel_each = bce_loss_each(logits_pos_sel, label_pos)  # (B,)

                if trm == "bb_ce":
                    ce_base = _bb_retrieval_ce(
                        match, int(args.match_scheme), info_pre, info_post, hydra_pre_base, hydra_post_base
                    )
                    ce_sel = _bb_retrieval_ce(
                        match, int(args.match_scheme), info_pre, info_post, hydra_pre_sel, hydra_post_sel
                    )
                    delta = ce_base - ce_sel
                    r_val = float(delta) / float(B * ksel)
                    r_pos_arm = torch.full((B * ksel,), r_val, device=cfg.device, dtype=torch.float32)
                    adv_pos = r_pos_arm.view(B, ksel).mean(dim=1)
                    bb_pos_n += 1
                    pos_bb_base_s += float(ce_base.item())
                    pos_bb_sel_s += float(ce_sel.item())
                    pos_bb_delta_s += float(delta.item())
                else:
                    loss_pos_base_k1 = _bce_k1_each_arm(
                        match=match,
                        info_pre=info_pre,
                        info_post=info_post,
                        hydra_pre=hydra_pre_base,
                        hydra_post=hydra_post_base,
                        label01=1.0,
                    ).view(B, ksel)  # (B,k)
                    base_pos_ref = loss_pos_base_k1.mean(dim=1, keepdim=True)  # (B,1)
                    loss_pos_sel_k1 = _bce_k1_each_arm(
                        match=match,
                        info_pre=info_pre,
                        info_post=info_post,
                        hydra_pre=hydra_pre_sel,
                        hydra_post=hydra_post_sel,
                        label01=1.0,
                    ).view(B, -1)  # (B,k)
                    r_pos_arm = (base_pos_ref - loss_pos_sel_k1).reshape(-1)  # (B*k,)

                    # For logging: sample-level advantage (mean over selected arms)
                    adv_pos = r_pos_arm.view(B, -1).mean(dim=1)  # (B,)
            r_pos_arm = _normalize_reward(r_pos_arm, zscore=bool(args.reward_zscore), clip=float(args.reward_clip))
            selector.update_batch(r_pos_arm)

            # choose + reward + update for neg
            hydra_pre_sel_n, hydra_post_sel_n, _, _ = selector.choose(neg_info_pre, neg_info_post, neg_hydra_pre, neg_hydra_post)
            with torch.no_grad():
                logits_neg_sel = _match_logits(match, neg_info_pre, neg_info_post, hydra_pre_sel_n, hydra_post_sel_n)
                loss_neg_sel_each = bce_loss_each(logits_neg_sel, label_neg)  # (B,)

                ksel = int(args.k_select)
                loss_neg_base_k1 = _bce_k1_each_arm(
                    match=match,
                    info_pre=neg_info_pre,
                    info_post=neg_info_post,
                    hydra_pre=neg_hydra_pre_base,
                    hydra_post=neg_hydra_post_base,
                    label01=0.0,
                ).view(B, ksel)
                base_neg_ref = loss_neg_base_k1.mean(dim=1, keepdim=True)
                loss_neg_sel_k1 = _bce_k1_each_arm(
                    match=match,
                    info_pre=neg_info_pre,
                    info_post=neg_info_post,
                    hydra_pre=hydra_pre_sel_n,
                    hydra_post=hydra_post_sel_n,
                    label01=0.0,
                ).view(B, -1)
                r_neg_arm = (base_neg_ref - loss_neg_sel_k1).reshape(-1)
                adv_neg = r_neg_arm.view(B, -1).mean(dim=1)
            r_neg_arm = _normalize_reward(r_neg_arm, zscore=bool(args.reward_zscore), clip=float(args.reward_clip))
            selector.update_batch(r_neg_arm)

            # reporting loss: average selected loss
            with torch.no_grad():
                loss_batch = 0.5 * (loss_pos_sel_each.mean() + loss_neg_sel_each.mean())
            losses.append(float(loss_batch.item()))
            # accumulate epoch diagnostics
            with torch.no_grad():
                pos_base_bce_s += float(loss_pos_base_each.sum().item())
                neg_base_bce_s += float(loss_neg_base_each.sum().item())
                pos_sel_bce_s += float(loss_pos_sel_each.sum().item())
                neg_sel_bce_s += float(loss_neg_sel_each.sum().item())

                adv_pos_s += float(adv_pos.sum().item())
                adv_neg_s += float(adv_neg.sum().item())
                adv_pos_sq_s += float((adv_pos * adv_pos).sum().item())
                adv_neg_sq_s += float((adv_neg * adv_neg).sum().item())
                adv_pos_poscnt += int((adv_pos > 0).sum().item())
                adv_neg_poscnt += int((adv_neg > 0).sum().item())
                n_pos += int(adv_pos.numel())
                n_neg += int(adv_neg.numel())
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        pbar.close()

        # quick valid: k×k retrieval on subset, using selector for every pair (expanded), same semantics as ft_match
        selector.freeze(True)
        with torch.no_grad():
            n_valid = len(valid_ds)
            valid_info_pre = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
            valid_info_post = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
            for i in range(n_valid):
                ip, io = _pair_to_info_xy8(valid_pairs[i])
                valid_info_pre[i] = ip
                valid_info_post[i] = io
            vg = torch.tensor(valid_gidx, dtype=torch.long, device="cpu")
            valid_hydra_pre = hydra_pre_all[vg]
            valid_hydra_post = hydra_post_all[vg]

            k = max(1, min(n_valid, int(round(float(n_valid) * float(args.valid_sample_frac)))))
            n_passes = max(1, int(args.valid_n_passes))
            base_seed = int(args.fixed_valid_seed) if args.fixed_valid_seed is not None else (int(args.sample_seed) + int(ep))

            acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = vloss_s = 0.0
            acc_base_s = hr5b_s = hr10b_s = hr20b_s = r51b_s = r102b_s = vlossb_s = 0.0
            for rep in range(n_passes):
                g = torch.Generator(device="cpu")
                g.manual_seed(int(base_seed) + rep * 7919)
                perm = torch.randperm(n_valid, generator=g)[:k]
                samp = perm.long()

                post_info = valid_info_post[samp].to(cfg.device)
                post_hydra = valid_hydra_post[samp].to(cfg.device)

                scores = torch.zeros((k, k), device=cfg.device)
                scores_base = torch.zeros((k, k), device=cfg.device)
                num = 0
                n_batches = (k + int(args.valid_batch) - 1) // int(args.valid_batch)
                for bi in range(n_batches):
                    i0 = bi * int(args.valid_batch)
                    i1 = min(i0 + int(args.valid_batch), k)
                    b = i1 - i0
                    row_idx = samp[i0:i1]
                    q_info_pre = valid_info_pre[row_idx].to(cfg.device)
                    q_hydra_pre = valid_hydra_pre[row_idx].to(cfg.device)

                    # expand to b*k pairs
                    B1, B2 = q_info_pre.size(0), post_info.size(0)
                    i_pre = q_info_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
                    i_post = post_info.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
                    h_pre = q_hydra_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)
                    h_post = post_hydra.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)

                    # vanilla baseline: first k-select tails
                    ksel = int(args.k_select)
                    h_pre_base = h_pre[:, :ksel]
                    h_post_base = h_post[:, :ksel]
                    logits_base = _match_logits(match, i_pre, i_post, h_pre_base, h_post_base).view(B1, B2)

                    h_pre_sel, h_post_sel, _, _ = selector.choose(i_pre, i_post, h_pre, h_post)
                    logits = _match_logits(match, i_pre, i_post, h_pre_sel, h_post_sel).view(B1, B2)
                    scores[num : num + b] = logits
                    scores_base[num : num + b] = logits_base
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
                pred_base = scores_base.argmax(dim=1)
                acc_base = float((pred_base == gt).float().mean().item())
                hr5b = _hr_at(scores_base, 5)
                hr10b = _hr_at(scores_base, 10)
                hr20b = _hr_at(scores_base, 20)
                r5b_at_10 = hr5b / hr10b if hr10b > 1e-12 else 0.0
                r10b_at_20 = hr10b / hr20b if hr20b > 1e-12 else 0.0
                vlossb = float(F.cross_entropy(scores_base, gt.long()).item())
                acc_base_s += acc_base
                hr5b_s += hr5b
                hr10b_s += hr10b
                hr20b_s += hr20b
                r51b_s += r5b_at_10
                r102b_s += r10b_at_20
                vlossb_s += vlossb

            acc = acc_s / n_passes
            hr5 = hr5_s / n_passes
            hr10 = hr10_s / n_passes
            hr20 = hr20_s / n_passes
            r5_at_10 = r51_s / n_passes
            r10_at_20 = r102_s / n_passes
            valid_loss_ce = vloss_s / n_passes

            acc_base = acc_base_s / n_passes
            hr5_base = hr5b_s / n_passes
            hr10_base = hr10b_s / n_passes
            hr20_base = hr20b_s / n_passes
            r5_at_10_base = r51b_s / n_passes
            r10_at_20_base = r102b_s / n_passes
            valid_loss_ce_base = vlossb_s / n_passes

        # epoch diagnostics (means/std)
        denom_pos = max(1, n_pos)
        denom_neg = max(1, n_neg)
        pos_base_bce = pos_base_bce_s / denom_pos
        neg_base_bce = neg_base_bce_s / denom_neg
        pos_sel_bce = pos_sel_bce_s / denom_pos
        neg_sel_bce = neg_sel_bce_s / denom_neg
        adv_pos_mean = adv_pos_s / denom_pos
        adv_neg_mean = adv_neg_s / denom_neg
        adv_pos_var = max(0.0, adv_pos_sq_s / denom_pos - adv_pos_mean * adv_pos_mean)
        adv_neg_var = max(0.0, adv_neg_sq_s / denom_neg - adv_neg_mean * adv_neg_mean)
        adv_pos_std = adv_pos_var ** 0.5
        adv_neg_std = adv_neg_var ** 0.5
        adv_pos_posrate = adv_pos_poscnt / denom_pos
        adv_neg_posrate = adv_neg_poscnt / denom_neg

        bb_suffix = ""
        if trm == "bb_ce" and bb_pos_n > 0:
            bn = float(bb_pos_n)
            bb_suffix = (
                f" posBB_CE(base/sel/delta)={pos_bb_base_s/bn:.4f}/{pos_bb_sel_s/bn:.4f}/{pos_bb_delta_s/bn:.4f}"
            )

        if acc > best_acc:
            best_acc = acc
            best_ep = ep
            stall = 0
            best_hr5 = hr5
            best_hr10 = hr10
            best_hr20 = hr20
            best_r5_at_10 = r5_at_10
            best_r10_at_20 = r10_at_20
            best_valid_loss_ce = valid_loss_ce

            best_acc_vanilla16 = acc_base
            best_hr5_vanilla16 = hr5_base
            best_hr10_vanilla16 = hr10_base
            best_hr20_vanilla16 = hr20_base
            best_r5_at_10_vanilla16 = r5_at_10_base
            best_r10_at_20_vanilla16 = r10_at_20_base
            best_valid_loss_ce_vanilla16 = valid_loss_ce_base
            best_ckpt = os.path.join(args.output_dir, f"best_selector_k{int(args.k_select)}_acc{best_acc:.4f}_ep{ep}.pt")
            selector.state().save(best_ckpt)
        else:
            stall += 1
            if stall >= int(args.early_stop):
                break

        print(
            f"[rl_tail_select] ep{ep}/{args.max_epochs} train_loss_mean={sum(losses)/max(1,len(losses)):.6f} "
            f"valid_loss_ce={valid_loss_ce:.6f} valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
            f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f} "
            f"vanilla16_loss_ce={valid_loss_ce_base:.6f} vanilla16_acc={acc_base:.4f} vanilla16_HR5={hr5_base:.4f} vanilla16_HR10={hr10_base:.4f} vanilla16_HR20={hr20_base:.4f} "
            f"vanilla16_R5@10={r5_at_10_base:.4f} vanilla16_R10@20={r10_at_20_base:.4f} "
            f"best_acc={best_acc:.4f} stall={stall}/{args.early_stop} "
            f"posBCE(base/sel)={pos_base_bce:.4f}/{pos_sel_bce:.4f} negBCE(base/sel)={neg_base_bce:.4f}/{neg_sel_bce:.4f} "
            f"adv_pos(mean±std,pos%)={adv_pos_mean:.4f}±{adv_pos_std:.4f},{adv_pos_posrate*100:.1f}% "
            f"adv_neg(mean±std,pos%)={adv_neg_mean:.4f}±{adv_neg_std:.4f},{adv_neg_posrate*100:.1f}% "
            f"best_selector_ckpt={best_ckpt}"
            f"{bb_suffix}",
            flush=True,
        )

    print(
        f"[rl_tail_select] done best_ep={best_ep} "
        f"best_valid_loss_ce={best_valid_loss_ce:.6f} best_acc={best_acc:.4f} "
        f"best_HR5={best_hr5:.4f} best_HR10={best_hr10:.4f} best_HR20={best_hr20:.4f} "
        f"best_R5@10={best_r5_at_10:.4f} best_R10@20={best_r10_at_20:.4f} "
        f"best_vanilla16_loss_ce={best_valid_loss_ce_vanilla16:.6f} best_vanilla16_acc={best_acc_vanilla16:.4f} "
        f"best_vanilla16_HR5={best_hr5_vanilla16:.4f} best_vanilla16_HR10={best_hr10_vanilla16:.4f} best_vanilla16_HR20={best_hr20_vanilla16:.4f} "
        f"best_vanilla16_R5@10={best_r5_at_10_vanilla16:.4f} best_vanilla16_R10@20={best_r10_at_20_vanilla16:.4f} "
        f"best_selector_ckpt={best_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()

