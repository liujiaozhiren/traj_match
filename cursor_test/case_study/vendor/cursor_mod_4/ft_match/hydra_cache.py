from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import cfg
from cursor_mod_4.geo_coords import match_traj_pts_to_xy_m
from cursor_mod_4.diffusion.fixlendiff84 import FixLenDiff84


class PairIndexDataset(Dataset):
    def __init__(self, pairs: Sequence[tuple]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return idx, self.pairs[idx]


def _pair_to_head_xy8(item) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      head_pre:  (2, k0=8)
      head_post: (2, k0=8)  where post uses reverse(B)
    """
    _, A, B, _, _, _, _ = item
    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    Brev = list(reversed(B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    head_pre = torch.tensor(A_xy, dtype=torch.float32).t().contiguous()
    head_post = torch.tensor(B_xy, dtype=torch.float32).t().contiguous()
    return head_pre, head_post


def _collate_idx_heads(batch):
    idxs = []
    heads_pre = []
    heads_post = []
    for idx, item in batch:
        hp, ho = _pair_to_head_xy8(item)
        idxs.append(int(idx))
        heads_pre.append(hp)
        heads_post.append(ho)
    idxs_t = torch.tensor(idxs, dtype=torch.long)
    return idxs_t, torch.stack(heads_pre, dim=0), torch.stack(heads_post, dim=0)


@dataclass
class HydraOnlyCache:
    pair_ids: torch.Tensor  # (N,) long, usually 0..N-1
    hydra_pre: torch.Tensor  # (N, hydra_tail, k, 2) on CPU
    hydra_post: torch.Tensor  # (N, hydra_tail, k, 2) on CPU

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "pair_ids": self.pair_ids,
                "hydra_pre": self.hydra_pre,
                "hydra_post": self.hydra_post,
                "meta": {
                    "diff_pre_len": int(cfg.diff_pre_len),
                    "diff_infer_len": int(cfg.diff_infer_len),
                    "hydra_tail": int(self.hydra_pre.shape[1]),
                },
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "HydraOnlyCache":
        d = torch.load(path, map_location="cpu")
        return HydraOnlyCache(
            pair_ids=d["pair_ids"].long().contiguous(),
            hydra_pre=d["hydra_pre"].float().contiguous(),
            hydra_post=d["hydra_post"].float().contiguous(),
        )


@torch.no_grad()
def build_hydra_only_cache(
    *,
    pairs: Sequence[tuple],
    pre_diff: FixLenDiff84,
    post_diff: FixLenDiff84,
    hydra_tail: int,
    batch: int,
    device: str,
) -> HydraOnlyCache:
    """
    Generate hydra tails for every pair in `pairs`.
    Cache stores ONLY hydra tails + pair_ids (order indices).
    """
    ds = PairIndexDataset(pairs)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False, collate_fn=_collate_idx_heads)
    n = len(ds)
    out_pre = torch.empty((n, hydra_tail, cfg.diff_infer_len, 2), dtype=torch.float32, device="cpu")
    out_post = torch.empty((n, hydra_tail, cfg.diff_infer_len, 2), dtype=torch.float32, device="cpu")
    pair_ids = torch.arange(n, dtype=torch.long)

    pre_diff.to(device)
    post_diff.to(device)
    pre_diff.unet.eval()
    post_diff.unet.eval()

    pbar = tqdm(loader, total=len(loader), desc=f"[hydra_cache_84] sampling (N={n}, tail={hydra_tail})", mininterval=0.5)
    for idxs, head_pre, head_post in pbar:
        head_pre = head_pre.to(device)
        head_post = head_post.to(device)
        hyd_pre = pre_diff.infer_from_noise(head_pre, num=hydra_tail)  # (B, n, 2, k)
        hyd_post = post_diff.infer_from_noise(head_post, num=hydra_tail)
        # to (B,n,k,2)
        hyd_pre = hyd_pre.permute(0, 1, 3, 2).contiguous().cpu()
        hyd_post = hyd_post.permute(0, 1, 3, 2).contiguous().cpu()
        out_pre[idxs] = hyd_pre
        out_post[idxs] = hyd_post
        pbar.set_postfix(done=int(idxs.max().item()) + 1)
    pbar.close()

    return HydraOnlyCache(pair_ids=pair_ids, hydra_pre=out_pre, hydra_post=out_post)

