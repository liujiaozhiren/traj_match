from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import cfg
from geo_coords import match_traj_pts_to_xy_m
from ft_match.hydra_cache import HydraOnlyCache
from ft_match.pair_split import split_train_valid_indices
from match import MatchModel, build_match_inputs
from match.matcher import pair_loss


class PairIdxDataset(Dataset):
    """
    Returns (global_cache_idx, pair). `global_cache_idx` indexes rows in HydraOnlyCache
    built from the same `all_pairs` list in stable order (row i <-> all_pairs[i]).
    """

    def __init__(self, pairs: Sequence[tuple], global_indices: Sequence[int], *, max_len: int | None = None):
        self.pairs = list(pairs)
        self.gidx = [int(x) for x in global_indices]
        if len(self.pairs) != len(self.gidx):
            raise ValueError("pairs and global_indices length mismatch")
        if max_len is not None:
            self.pairs = self.pairs[: int(max_len)]
            self.gidx = self.gidx[: int(max_len)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        return self.gidx[i], self.pairs[i]


def _pair_to_info_xy8(item) -> tuple[torch.Tensor, torch.Tensor]:
    _, A, B, _, _, _, _ = item
    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    Brev = list(reversed(B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    info_pre = torch.tensor(A_xy, dtype=torch.float32)  # (8,2)
    info_post = torch.tensor(B_xy, dtype=torch.float32)
    return info_pre, info_post


def _collate_idx_info(batch):
    idxs = []
    infos_pre = []
    infos_post = []
    for idx, item in batch:
        ip, io = _pair_to_info_xy8(item)
        idxs.append(int(idx))
        infos_pre.append(ip)
        infos_post.append(io)
    return (
        torch.tensor(idxs, dtype=torch.long),
        torch.stack(infos_pre, dim=0),
        torch.stack(infos_post, dim=0),
    )


def _load_pairs_pkl(pkl_path: str, sample_ratio: float, sample_seed: int, *, max_pairs: int | None = None):
    raw = pickle.load(open(pkl_path, "rb"))
    if not isinstance(raw, list):
        raise TypeError(f"pairs_pkl must be a list, got {type(raw).__name__}")
    if raw and (not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7):
        raise TypeError(f"pairs_pkl items must be 7-tuples; first len={len(raw[0]) if raw else 'n/a'}")
    if max_pairs is not None:
        raw = raw[: int(max_pairs)]
    return split_train_valid_indices(raw, sample_ratio, sample_seed)


def _require_pairs_pkl(pairs_pkl: str | None) -> str:
    path = pairs_pkl or cfg.pairs_pkl
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"pairs pkl not found: {path}")
    return path
    raw = pickle.load(open(pkl_path, "rb"))
    if not isinstance(raw, list):
        raise TypeError(f"pairs_pkl must be a list, got {type(raw).__name__}")
    if raw and (not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7):
        raise TypeError(f"pairs_pkl items must be 7-tuples; first len={len(raw[0]) if raw else 'n/a'}")
    if max_pairs is not None:
        raw = raw[: int(max_pairs)]
    return split_train_valid_indices(raw, sample_ratio, sample_seed)


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


@dataclass
class TrainResult:
    best_acc: float
    best_ckpt: str | None
    best_hr5: float
    best_hr10: float
    best_hr20: float
    best_r5_at_10: float
    best_r10_at_20: float
    best_valid_loss_ce: float
    best_epoch: int | None
    # After training: best checkpoint re-evaluated at smaller valid pools (same valid split, new frac).
    posthoc_valid: list[dict] | None = None


@torch.no_grad()
def _evaluate_valid_retrieval(
    match: torch.nn.Module,
    *,
    valid_info_pre: torch.Tensor,
    valid_info_post: torch.Tensor,
    valid_hydra_pre: torch.Tensor,
    valid_hydra_post: torch.Tensor,
    n_valid: int,
    valid_batch: int,
    valid_sample_frac: float,
    valid_n_passes: int,
    base_seed: int,
    match_scheme: int,
    desc: str,
    show_pbar: bool,
) -> dict:
    k = max(1, min(n_valid, int(round(float(n_valid) * float(valid_sample_frac)))))
    n_passes = max(1, int(valid_n_passes))
    acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = vloss_s = 0.0
    for rep in range(n_passes):
        g = torch.Generator(device="cpu")
        g.manual_seed(int(base_seed) + rep * 7919)
        perm = torch.randperm(n_valid, generator=g)[:k]
        samp = perm.long()

        post_info = valid_info_post[samp].to(cfg.device)
        post_hydra = valid_hydra_post[samp].to(cfg.device)

        scores = torch.zeros((k, k), device=cfg.device)
        num = 0
        n_batches = (k + valid_batch - 1) // valid_batch
        inner = range(n_batches)
        if show_pbar:
            inner = tqdm(inner, desc=f"{desc} pass{rep+1}/{n_passes} k={k}/{n_valid}", mininterval=0.5)
        for bi in inner:
            i0 = bi * valid_batch
            i1 = min(i0 + valid_batch, k)
            b = i1 - i0
            row_idx = samp[i0:i1]
            q_info_pre = valid_info_pre[row_idx].to(cfg.device, non_blocking=True)
            q_hydra_pre = valid_hydra_pre[row_idx].to(cfg.device, non_blocking=True)
            x = build_match_inputs(
                scheme=int(match_scheme),
                trajA_head=q_info_pre,
                trajB_head=post_info,
                trajA_tail=q_hydra_pre,
                trajB_tail=post_hydra,
                pair=True,
                valid=True,
            )
            ret = match.match(x, pairs=True).view(b, k)
            scores[num : num + b] = ret
            num += b
            if show_pbar:
                inner.set_postfix(rows=f"{num}/{k}")
        if show_pbar and hasattr(inner, "close"):
            inner.close()

        pred = scores.argmax(dim=1)
        gt = torch.arange(k, device=cfg.device)
        acc_p = float((pred == gt).float().mean().item())
        hr5_p = _hr_at(scores, 5)
        hr10_p = _hr_at(scores, 10)
        hr20_p = _hr_at(scores, 20)
        r5_at_10_p = hr5_p / hr10_p if hr10_p > 1e-12 else 0.0
        r10_at_20_p = hr10_p / hr20_p if hr20_p > 1e-12 else 0.0
        vloss_p = float(F.cross_entropy(scores, gt.long()).item())

        acc_s += acc_p
        hr5_s += hr5_p
        hr10_s += hr10_p
        hr20_s += hr20_p
        r51_s += r5_at_10_p
        r102_s += r10_at_20_p
        vloss_s += vloss_p

    return {
        "valid_sample_frac": float(valid_sample_frac),
        "k": int(k),
        "n_valid": int(n_valid),
        "acc": acc_s / n_passes,
        "hr5": hr5_s / n_passes,
        "hr10": hr10_s / n_passes,
        "hr20": hr20_s / n_passes,
        "r5_at_10": r51_s / n_passes,
        "r10_at_20": r102_s / n_passes,
        "valid_loss_ce": vloss_s / n_passes,
    }


def run_ft_match_84(
    *,
    output_dir: str,
    cache_path: str,
    train_batch: int,
    valid_batch: int,
    lr: float,
    max_epochs: int,
    early_stop: int,
    rl_use: int,
    sample_ratio: float,
    sample_seed: int,
    valid_sample_frac: float = 0.15,
    valid_sample_seed: int | None = None,
    valid_n_passes: int = 3,
    match_scheme: int = 0,
    pairs_pkl: str | None = None,
    max_pairs: int | None = None,
    train_loss_mode: Literal["bce", "bb_ce"] = "bce",
    match_dim: int | None = None,
) -> TrainResult:
    os.makedirs(output_dir, exist_ok=True)

    tlm = str(train_loss_mode).lower()
    if tlm not in ("bce", "bb_ce"):
        raise ValueError(f"train_loss_mode must be 'bce' or 'bb_ce', got {train_loss_mode!r}")
    if tlm == "bb_ce" and int(match_scheme) not in (3, 4):
        raise ValueError(
            "train_loss_mode='bb_ce' is only defined for match_scheme 3 or 4 (tensor B×B retrieval). "
            "Schemes 0/1/2 use graph + pair BCE labels; keep train_loss_mode='bce'."
        )

    print(
        f"[ft_match_84] train_loss_mode={tlm} match_scheme={int(match_scheme)}",
        flush=True,
    )

    if pairs_pkl:
        pkl = _require_pairs_pkl(pairs_pkl)
        train_pairs, valid_pairs, train_gidx, valid_gidx = _load_pairs_pkl(
            pkl, sample_ratio, sample_seed, max_pairs=max_pairs
        )
    else:
        pkl = _require_pairs_pkl(None)
        train_pairs, valid_pairs, train_gidx, valid_gidx = _load_pairs_pkl(
            pkl, sample_ratio, sample_seed, max_pairs=max_pairs
        )
    train_ds = PairIdxDataset(train_pairs, train_gidx)
    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, num_workers=0, pin_memory=False, collate_fn=_collate_idx_info)

    cache = HydraOnlyCache.load(cache_path)
    hydra_pre_all = cache.hydra_pre[:, :rl_use]  # (N, rl_use, k, 2)
    hydra_post_all = cache.hydra_post[:, :rl_use]

    n_pairs_total = len(train_pairs) + len(valid_pairs)
    if int(hydra_pre_all.shape[0]) != int(n_pairs_total):
        raise ValueError(
            f"Hydra cache rows ({hydra_pre_all.shape[0]}) != train+valid pair count ({n_pairs_total}); "
            "rebuild cache with the same --pairs-pkl and --max-pairs as this run."
        )
    print(f"[ft_match_84] pairs source=pkl: {os.path.abspath(pkl)}", flush=True)

    n_valid = len(valid_pairs)
    valid_info_pre = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
    valid_info_post = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
    for i in range(n_valid):
        ip, io = _pair_to_info_xy8(valid_pairs[i])
        valid_info_pre[i] = ip
        valid_info_post[i] = io
    vg = torch.tensor(valid_gidx, dtype=torch.long, device="cpu")
    valid_hydra_pre = hydra_pre_all[vg]
    valid_hydra_post = hydra_post_all[vg]

    m_dim = int(match_dim if match_dim is not None else cfg.match_dim)
    match = MatchModel(dim=m_dim, scheme=int(match_scheme)).to(cfg.device)
    optim = torch.optim.Adam(match.parameters(), lr=lr)

    best_acc = -1.0
    best_ckpt = None
    best_epoch = None
    best_hr5 = best_hr10 = best_hr20 = 0.0
    best_r5_at_10 = best_r10_at_20 = 0.0
    best_valid_loss_ce = float("inf")
    stall = 0
    n_passes = max(1, int(valid_n_passes))

    for ep in range(1, max_epochs + 1):
        match.train()
        losses = []
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[ft_match_84] train ep{ep}/{max_epochs}", mininterval=0.5)
        for idxs, info_pre, info_post in pbar:
            # info_*: (B,8,2) meters
            info_pre = info_pre.to(cfg.device, non_blocking=True)
            info_post = info_post.to(cfg.device, non_blocking=True)
            # idxs: global row indices into hydra cache (same order as full pair list used at cache build)
            hydra_pre = hydra_pre_all[idxs].to(cfg.device, non_blocking=True)  # (B,rl_use,4,2)
            hydra_post = hydra_post_all[idxs].to(cfg.device, non_blocking=True)
            B = int(info_pre.shape[0])

            if tlm == "bb_ce":
                if B < 2:
                    continue
                x = build_match_inputs(
                    scheme=int(match_scheme),
                    trajA_head=info_pre,
                    trajB_head=info_post,
                    trajA_tail=hydra_pre,
                    trajB_tail=hydra_post,
                    pair=True,
                    valid=True,
                )
                logits = match.match(x, pairs=True).view(B, B).float()
                loss = F.cross_entropy(logits, torch.arange(B, device=cfg.device, dtype=torch.long))
            else:
                x, y = build_match_inputs(
                    scheme=int(match_scheme),
                    trajA_head=info_pre,
                    trajB_head=info_post,
                    trajA_tail=hydra_pre,
                    trajB_tail=hydra_post,
                    pair=True,
                    valid=False,
                )
                ret = match.match(x, pairs=True).view(-1)
                loss = pair_loss(ret, y).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        pbar.close()

        match.eval()
        with torch.no_grad():
            base_seed = int(valid_sample_seed) if valid_sample_seed is not None else (int(sample_seed) + int(ep))
            vm = _evaluate_valid_retrieval(
                match,
                valid_info_pre=valid_info_pre,
                valid_info_post=valid_info_post,
                valid_hydra_pre=valid_hydra_pre,
                valid_hydra_post=valid_hydra_post,
                n_valid=n_valid,
                valid_batch=valid_batch,
                valid_sample_frac=float(valid_sample_frac),
                valid_n_passes=valid_n_passes,
                base_seed=base_seed,
                match_scheme=int(match_scheme),
                desc=f"[ft_match_84] valid ep{ep}/{max_epochs}",
                show_pbar=True,
            )
        acc = float(vm["acc"])
        hr5 = float(vm["hr5"])
        hr10 = float(vm["hr10"])
        hr20 = float(vm["hr20"])
        r5_at_10 = float(vm["r5_at_10"])
        r10_at_20 = float(vm["r10_at_20"])
        valid_loss = float(vm["valid_loss_ce"])
        k = int(vm["k"])

        if acc > best_acc:
            best_acc = acc
            stall = 0
            best_epoch = ep
            best_hr5 = hr5
            best_hr10 = hr10
            best_hr20 = hr20
            best_r5_at_10 = r5_at_10
            best_r10_at_20 = r10_at_20
            best_valid_loss_ce = valid_loss
            best_ckpt = os.path.join(output_dir, f"best_match_s{int(match_scheme)}_ep{ep}_acc{best_acc:.4f}.pth")
            torch.save(match.state_dict(), best_ckpt)
        else:
            stall += 1
            if stall >= early_stop:
                break

        print(
            f"[ft_match_84] ep{ep}/{max_epochs} train_loss_mean={sum(losses)/max(1,len(losses)):.6f} "
            f"valid_loss_ce={valid_loss:.6f} (mean over {n_passes} passes @ k={k}/{n_valid} {100.0*k/n_valid:.1f}%) "
            f"valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
            f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f} "
            f"best_acc={best_acc:.4f} stall={stall}/{early_stop}",
            flush=True,
        )

    posthoc_valid: list[dict] = []
    if best_ckpt is not None and os.path.isfile(best_ckpt):
        match.load_state_dict(torch.load(best_ckpt, map_location=cfg.device))
        match.eval()
        with torch.no_grad():
            for fi, frac in enumerate((0.15, 0.10, 0.05, 0.02)):
                pm = _evaluate_valid_retrieval(
                    match,
                    valid_info_pre=valid_info_pre,
                    valid_info_post=valid_info_post,
                    valid_hydra_pre=valid_hydra_pre,
                    valid_hydra_post=valid_hydra_post,
                    n_valid=n_valid,
                    valid_batch=valid_batch,
                    valid_sample_frac=float(frac),
                    valid_n_passes=valid_n_passes,
                    base_seed=int(sample_seed) + 951_000 + fi * 97,
                    match_scheme=int(match_scheme),
                    desc=f"[ft_match_84] posthoc_best frac={frac}",
                    show_pbar=False,
                )
                posthoc_valid.append(pm)
                print(
                    f"[ft_match_84] posthoc_best ep={best_epoch} ckpt={best_ckpt} valid_sample_frac={frac} "
                    f"k={pm['k']}/{n_valid} ({100.0 * float(pm['k']) / max(1, n_valid):.1f}%) "
                    f"valid_loss_ce={pm['valid_loss_ce']:.6f} (mean over {n_passes} passes) "
                    f"acc={pm['acc']:.4f} HR5={pm['hr5']:.4f} HR10={pm['hr10']:.4f} HR20={pm['hr20']:.4f} "
                    f"R5@10={pm['r5_at_10']:.4f} R10@20={pm['r10_at_20']:.4f}",
                    flush=True,
                )
    else:
        print("[ft_match_84] posthoc_best skipped (no checkpoint was saved).", flush=True)

    return TrainResult(
        best_acc=best_acc,
        best_ckpt=best_ckpt,
        best_hr5=best_hr5,
        best_hr10=best_hr10,
        best_hr20=best_hr20,
        best_r5_at_10=best_r5_at_10,
        best_r10_at_20=best_r10_at_20,
        best_valid_loss_ce=best_valid_loss_ce,
        best_epoch=best_epoch,
        posthoc_valid=posthoc_valid if posthoc_valid else None,
    )

