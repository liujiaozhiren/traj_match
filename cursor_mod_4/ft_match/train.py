from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import cfg
from cursor_mod_4.geo_coords import match_traj_pts_to_xy_m
from cursor_mod_4.ft_match.hydra_cache import HydraOnlyCache
from cursor_mod_4.match import MatchModel, build_match_inputs
from cursor_mod_4.match.matcher import pair_loss
from cursor_mod_4.proc import clean_pairs, proc_multi_single_traj
from cursor_mod_4.sample_db import sample_db


class PairIdxDataset(Dataset):
    def __init__(self, pairs: Sequence[tuple], *, max_len: int | None = None):
        self.pairs = list(pairs)
        if max_len is not None:
            self.pairs = self.pairs[: int(max_len)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        return i, self.pairs[i]


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


def _load_pairs(sample_ratio: float, sample_seed: int):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    valid = sample_db(all_pairs, ratio=sample_ratio, seed=sample_seed)
    train = [p for p in all_pairs if p not in valid]
    return train, valid


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
) -> TrainResult:
    os.makedirs(output_dir, exist_ok=True)

    train_pairs, valid_pairs = _load_pairs(sample_ratio, sample_seed)
    train_ds = PairIdxDataset(train_pairs)
    valid_ds = PairIdxDataset(valid_pairs)
    train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True, num_workers=0, pin_memory=False, collate_fn=_collate_idx_info)

    cache = HydraOnlyCache.load(cache_path)
    hydra_pre_all = cache.hydra_pre[:, :rl_use]  # (N, rl_use, k, 2)
    hydra_post_all = cache.hydra_post[:, :rl_use]

    match = MatchModel(dim=128, scheme=int(match_scheme)).to(cfg.device)
    optim = torch.optim.Adam(match.parameters(), lr=lr)

    best_acc = -1.0
    best_ckpt = None
    best_epoch = None
    best_hr5 = best_hr10 = best_hr20 = 0.0
    best_r5_at_10 = best_r10_at_20 = 0.0
    best_valid_loss_ce = float("inf")
    stall = 0

    for ep in range(1, max_epochs + 1):
        match.train()
        losses = []
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[ft_match_84] train ep{ep}/{max_epochs}", mininterval=0.5)
        for idxs, info_pre, info_post in pbar:
            # info_*: (B,8,2) meters
            info_pre = info_pre.to(cfg.device, non_blocking=True)
            info_post = info_post.to(cfg.device, non_blocking=True)
            hydra_pre = hydra_pre_all[idxs].to(cfg.device, non_blocking=True)   # (B,rl_use,4,2)
            hydra_post = hydra_post_all[idxs].to(cfg.device, non_blocking=True)

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

        # valid: several passes of random ~frac*|valid| subset, k×k retrieval; average metrics + row-wise CE loss
        match.eval()
        with torch.no_grad():
            n_valid = len(valid_ds)
            valid_info_pre = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
            valid_info_post = torch.zeros((n_valid, cfg.diff_pre_len, 2), dtype=torch.float32)
            for i in range(n_valid):
                ip, io = _pair_to_info_xy8(valid_pairs[i])
                valid_info_pre[i] = ip
                valid_info_post[i] = io
            valid_hydra_pre = hydra_pre_all[:n_valid]
            valid_hydra_post = hydra_post_all[:n_valid]

            k = max(1, min(n_valid, int(round(float(n_valid) * float(valid_sample_frac)))))
            n_passes = max(1, int(valid_n_passes))
            base_seed = int(valid_sample_seed) if valid_sample_seed is not None else (int(sample_seed) + int(ep))

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
                vbar = tqdm(
                    range(n_batches),
                    desc=f"[ft_match_84] valid ep{ep}/{max_epochs} pass{rep+1}/{n_passes} k={k}/{n_valid}",
                    mininterval=0.5,
                )
                for bi in vbar:
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
                    vbar.set_postfix(rows=f"{num}/{k}")
                vbar.close()

                pred = scores.argmax(dim=1)
                gt = torch.arange(k, device=cfg.device)
                acc_p = float((pred == gt).float().mean().item())
                hr5_p = _hr_at(scores, 5)
                hr10_p = _hr_at(scores, 10)
                hr20_p = _hr_at(scores, 20)
                r5_at_10_p = hr5_p / hr10_p if hr10_p > 1e-12 else 0.0
                r10_at_20_p = hr10_p / hr20_p if hr20_p > 1e-12 else 0.0
                # row-wise softmax CE vs diagonal label (same decision rule as acc; comparable scale across k)
                vloss_p = float(F.cross_entropy(scores, gt.long()).item())

                acc_s += acc_p
                hr5_s += hr5_p
                hr10_s += hr10_p
                hr20_s += hr20_p
                r51_s += r5_at_10_p
                r102_s += r10_at_20_p
                vloss_s += vloss_p

            acc = acc_s / n_passes
            hr5 = hr5_s / n_passes
            hr10 = hr10_s / n_passes
            hr20 = hr20_s / n_passes
            r5_at_10 = r51_s / n_passes
            r10_at_20 = r102_s / n_passes
            valid_loss = vloss_s / n_passes

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
    )

