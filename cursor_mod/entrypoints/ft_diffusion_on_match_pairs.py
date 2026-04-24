"""
Fine-tune diffusion (pre/post) on Shenzhen match pairs in **meters**.

Supervision (per pair):
- pre:  info = noisy A head (k0=cfg.diff_pre_len points), label = clean L_A tail last k points (k=cfg.diff_infer_len)
- post: info = noisy reverse(B) head, label = clean reverse(L_B) tail last k points

We train FixLenDiff by concatenating [info, label] along time and calling `learn()`,
which splits by cfg.diff_pre_len internally.

Checkpointing:
- latest_pre.pth / latest_post.pth + latest_epoch.txt saved every --save-every epochs
- best_pre_ep{ep}_rmse{...}m.pth and best_post_ep{ep}_rmse{...}m.pth saved on improvement
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repo root on sys.path
_ROOT = Path(__file__).resolve().parents[2]
_rp = str(_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

# Shadow cfg for this process
sys.modules["cfg"] = importlib.import_module("cursor_mod.cfg")
import cfg  # noqa: E402

from cursor_mod.geo_coords import match_traj_pts_to_xy_m, rmse_l2_m  # noqa: E402
from my_diffusion.fixLdiff.model import FixLenDiff  # noqa: E402
from new_test.proc import clean_pairs, proc_multi_single_traj  # noqa: E402


class PairDiffDataset(Dataset):
    def __init__(self, pairs: Sequence[tuple], *, max_pairs: int | None = None):
        self.pairs = list(pairs)
        if max_pairs is not None:
            self.pairs = self.pairs[: int(max_pairs)]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def _pair_to_meters_tensors(item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      info_pre_xy:  (2,k0)
      label_pre_xy: (2,k)
      info_post_xy: (2,k0)
      label_post_xy:(2,k)
    """
    _, A, B, L_A, L_B, _, _ = item

    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    LA_xy = match_traj_pts_to_xy_m(L_A)
    lab_pre = LA_xy[-cfg.diff_infer_len :]

    Brev = list(reversed(B))
    LBrev = list(reversed(L_B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    LB_xy = match_traj_pts_to_xy_m(LBrev)
    lab_post = LB_xy[-cfg.diff_infer_len :]

    info_pre = torch.tensor(A_xy, dtype=torch.float32).t().contiguous()
    label_pre = torch.tensor(lab_pre, dtype=torch.float32).t().contiguous()
    info_post = torch.tensor(B_xy, dtype=torch.float32).t().contiguous()
    label_post = torch.tensor(lab_post, dtype=torch.float32).t().contiguous()
    return info_pre, label_pre, info_post, label_post


def _collate(batch):
    infos_pre, labels_pre, infos_post, labels_post = [], [], [], []
    for item in batch:
        ip, lp, io, lo = _pair_to_meters_tensors(item)
        infos_pre.append(ip)
        labels_pre.append(lp)
        infos_post.append(io)
        labels_post.append(lo)
    return (
        torch.stack(infos_pre, dim=0),
        torch.stack(labels_pre, dim=0),
        torch.stack(infos_post, dim=0),
        torch.stack(labels_post, dim=0),
    )


def _load_pairs(max_pairs: int, valid_ratio: float, seed: int):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    all_pairs = all_pairs[: int(max_pairs)]
    n_valid = max(1, int(len(all_pairs) * float(valid_ratio)))
    valid = all_pairs[:n_valid]
    train = all_pairs[n_valid:]
    return train, valid


def _save_state_dict_only(path: str, sd: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(sd, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-pre", type=str, required=True)
    p.add_argument("--ckpt-post", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-pairs", type=int, default=20000)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--diff-pre-post",
        type=str,
        default=None,
        choices=["a_pre_only", "a_post_only", "b_alternate"],
        help="If omitted: default C (joint train both, best saved separately).",
    )
    p.add_argument("--seed", type=int, default=20250803)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lambda-x0", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    cfg.diff_lr = float(args.lr)
    cfg.lambda_x0 = float(args.lambda_x0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # build models
    pre = FixLenDiff(device=cfg.device, lr=cfg.diff_lr)
    post = FixLenDiff(device=cfg.device, lr=cfg.diff_lr)

    latest_pre = os.path.join(out_dir, "latest_pre.pth")
    latest_post = os.path.join(out_dir, "latest_post.pth")
    latest_epoch = os.path.join(out_dir, "latest_epoch.txt")

    if args.resume and os.path.exists(latest_pre) and os.path.exists(latest_post):
        pre.unet.load_state_dict(torch.load(latest_pre, map_location=cfg.device), strict=True)
        post.unet.load_state_dict(torch.load(latest_post, map_location=cfg.device), strict=True)
        start_epoch = 1
        if os.path.exists(latest_epoch):
            try:
                start_epoch = int(open(latest_epoch, "r").read().strip()) + 1
            except Exception:
                start_epoch = 1
    else:
        pre.unet.load_state_dict(torch.load(args.ckpt_pre, map_location=cfg.device), strict=True)
        post.unet.load_state_dict(torch.load(args.ckpt_post, map_location=cfg.device), strict=True)
        start_epoch = 1

    train_pairs, valid_pairs = _load_pairs(args.max_pairs, args.valid_ratio, args.seed)
    train_loader = DataLoader(
        PairDiffDataset(train_pairs),
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate,
        drop_last=False,
    )
    valid_loader = DataLoader(
        PairDiffDataset(valid_pairs),
        batch_size=args.eval_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate,
        drop_last=False,
    )

    best_pre = float("inf")
    best_post = float("inf")

    schedule = args.diff_pre_post or "C_joint"

    for ep in range(start_epoch, int(args.epochs) + 1):
        pre.unet.train()
        post.unet.train()

        loss_list = []
        for info_pre, lab_pre, info_post, lab_post in train_loader:
            info_pre = info_pre.to(cfg.device)
            lab_pre = lab_pre.to(cfg.device)
            info_post = info_post.to(cfg.device)
            lab_post = lab_post.to(cfg.device)

            # build input_trajs as (B,2,k0+k)
            x_pre = torch.cat([info_pre, lab_pre], dim=2)
            x_post = torch.cat([info_post, lab_post], dim=2)

            do_pre = schedule in ("C_joint", "a_pre_only") or (schedule == "b_alternate" and ep % 2 == 1)
            do_post = schedule in ("C_joint", "a_post_only") or (schedule == "b_alternate" and ep % 2 == 0)

            l_pre = l_post = 0.0
            r_pre = r_post = 0.0
            if do_pre:
                l_pre, r_pre = pre.learn(x_pre, lam=cfg.lambda_x0)
            if do_post:
                l_post, r_post = post.learn(x_post, lam=cfg.lambda_x0)
            loss_list.append((float(l_pre) + float(l_post)) / max(1, int(do_pre) + int(do_post)))

        # eval
        pre.unet.eval()
        post.unet.eval()
        with torch.no_grad():
            errs_pre = []
            errs_post = []
            for info_pre, lab_pre, info_post, lab_post in valid_loader:
                info_pre = info_pre.to(cfg.device)
                lab_pre = lab_pre.to(cfg.device)
                info_post = info_post.to(cfg.device)
                lab_post = lab_post.to(cfg.device)

                pred_pre = pre.infer_from_noise(info_pre)
                pred_post = post.infer_from_noise(info_post)

                # per batch scalar rmse (meters)
                errs_pre.append(rmse_l2_m(pred_pre, lab_pre).detach().float().cpu())
                errs_post.append(rmse_l2_m(pred_post, lab_post).detach().float().cpu())

            rmse_pre_m = float(torch.stack(errs_pre).mean().item())
            rmse_post_m = float(torch.stack(errs_post).mean().item())

        # save latest
        if ep % int(args.save_every) == 0:
            _save_state_dict_only(latest_pre, pre.unet.state_dict())
            _save_state_dict_only(latest_post, post.unet.state_dict())
            with open(latest_epoch, "w") as f:
                f.write(str(ep))

        # save best separately (default C behavior)
        if rmse_pre_m < best_pre:
            best_pre = rmse_pre_m
            fn = os.path.join(out_dir, f"best_pre_ep{ep}_rmse{best_pre:.3f}m.pth")
            _save_state_dict_only(fn, pre.unet.state_dict())
        if rmse_post_m < best_post:
            best_post = rmse_post_m
            fn = os.path.join(out_dir, f"best_post_ep{ep}_rmse{best_post:.3f}m.pth")
            _save_state_dict_only(fn, post.unet.state_dict())

        train_loss_mean = sum(loss_list) / max(1, len(loss_list))
        print(
            f"[ft_diff_pairs_m] ep{ep}/{args.epochs} schedule={schedule} "
            f"train_loss_mean={train_loss_mean:.6f} "
            f"valid(pre) rmse_m={rmse_pre_m:.3f} | valid(post) rmse_m={rmse_post_m:.3f} "
            f"| best_pre={best_pre:.3f} best_post={best_post:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

