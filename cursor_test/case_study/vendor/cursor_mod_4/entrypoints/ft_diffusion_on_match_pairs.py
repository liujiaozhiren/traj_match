"""
FT diffusion for cursor_mod_4: diff_pre_len=8, diff_infer_len=4 (meters).

Data sources:
  - Default: `cfg.con_file` -> traj_cmb.pkl -> proc_multi_single_traj -> clean_pairs.
  - `--pairs-pkl`: pickle list of 7-tuples (same as `clean_pairs` output / v1 / v2 maxdtw):
        (mid, A, B, L_A, L_B, F_A, F_B)
    Train/valid split for `--pairs-pkl` matches `run_ft_match_84` (same `sample_ratio` / `sample_seed`
    on the `[:max_pairs]` prefix). `--valid-ratio` is only used for the traj_cmb path.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[2]
_rp = str(_ROOT)
if _rp not in sys.path:
    sys.path.insert(0, _rp)

sys.modules["cfg"] = importlib.import_module("cursor_mod_4.cfg")
import cfg  # noqa: E402

from cursor_mod_4.geo_coords import match_traj_pts_to_xy_m, rmse_l2_m  # noqa: E402
from cursor_mod_4.diffusion.fixlendiff84 import FixLenDiff84  # noqa: E402
from cursor_mod_4.proc import clean_pairs, proc_multi_single_traj  # noqa: E402
from cursor_mod_4.ft_match.pair_split import split_train_valid_indices  # noqa: E402


class PairDiffDataset(Dataset):
    def __init__(self, pairs, max_pairs=None):
        self.pairs = list(pairs)
        if max_pairs is not None:
            self.pairs = self.pairs[: int(max_pairs)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def _pair_to_meters_tensors(item):
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
    ips, lps, ios, los = [], [], [], []
    for item in batch:
        ip, lp, io, lo = _pair_to_meters_tensors(item)
        ips.append(ip)
        lps.append(lp)
        ios.append(io)
        los.append(lo)
    return torch.stack(ips), torch.stack(lps), torch.stack(ios), torch.stack(los)

def _shuffle_split_pairs(all_pairs: list, max_pairs: int, valid_ratio: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    all_pairs = all_pairs[: int(max_pairs)]
    if len(all_pairs) == 0:
        raise ValueError("no pairs after shuffle/cap (check max-pairs or input file)")
    n_valid = max(1, int(len(all_pairs) * float(valid_ratio)))
    if n_valid >= len(all_pairs):
        n_valid = max(1, len(all_pairs) - 1)
    valid = all_pairs[:n_valid]
    train = all_pairs[n_valid:]
    if len(train) == 0:
        raise ValueError("train split empty; reduce --valid-ratio or increase pairs")
    return train, valid


def _load_pairs(max_pairs: int, valid_ratio: float, seed: int):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    return _shuffle_split_pairs(all_pairs, max_pairs, valid_ratio, seed)


def _load_pairs_from_pkl(pkl_path: str, max_pairs: int, sample_ratio: float, sample_seed: int):
    """Same list order + split as `run_ft_match_84` with `pairs_pkl` / `max_pairs` / `sample_ratio` / `sample_seed`."""
    path = os.path.abspath(pkl_path)
    raw = pickle.load(open(path, "rb"))
    if not isinstance(raw, list):
        raise ValueError(f"--pairs-pkl must unpickle to a list, got {type(raw)}")
    if raw and (not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7):
        raise ValueError(
            f"expected 7-tuple pairs (mid,A,B,L_A,L_B,F_A,F_B), first item len={len(raw[0]) if raw else 'n/a'}"
        )
    if max_pairs is not None:
        raw = raw[: int(max_pairs)]
    train_pairs, valid_pairs, _, _ = split_train_valid_indices(raw, float(sample_ratio), int(sample_seed))
    return train_pairs, valid_pairs


def _save_sd(path: str, sd: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(sd, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pairs-pkl",
        type=str,
        default=None,
        help="Optional. Pickle list of 7-tuples (clean_pairs / v1 / v2 maxdtw). If omitted, load cfg.con_file.",
    )
    p.add_argument("--ckpt-pre", type=str, default=None)
    p.add_argument("--ckpt-post", type=str, default=None)
    p.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not load --ckpt-pre/post; keep random init (ignored if --resume finds latest_*.pth).",
    )
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-pairs", type=int, default=20000)
    p.add_argument(
        "--valid-ratio",
        type=float,
        default=0.05,
        help="Only for traj_cmb path (no --pairs-pkl). Shuffle-split valid fraction.",
    )
    p.add_argument(
        "--sample-ratio",
        type=float,
        default=0.1,
        help="Only with --pairs-pkl: validation fraction (same as run_ft_match --sample-ratio). Train is 1-this.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Only with --pairs-pkl: RNG seed for the index split (same as run_ft_match --sample-seed).",
    )
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=20250803)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--lambda-x0", type=float, default=0.0)
    p.add_argument("--log-file", type=str, default=None, help="Default: <out-dir>/train.log")
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=10,
        help="Stop if neither valid rmse_pre nor rmse_post beats its running best for this many epochs. 0 disables.",
    )
    return p.parse_args()

def _setup_run_logging(log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    cfg.diff_lr = float(args.lr)
    cfg.lambda_x0 = float(args.lambda_x0)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log_path = args.log_file or os.path.join(out_dir, "train.log")
    _setup_run_logging(log_path)

    pre = FixLenDiff84(device=cfg.device, lr=cfg.diff_lr)
    post = FixLenDiff84(device=cfg.device, lr=cfg.diff_lr)

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
        logging.info("[ft_diff_pairs_84m] resumed from %s", out_dir)
    else:
        if args.from_scratch:
            start_epoch = 1
            logging.info("[ft_diff_pairs_84m] init from scratch (no ckpt load)")
        else:
            if not args.ckpt_pre or not args.ckpt_post:
                raise SystemExit(
                    "Provide both --ckpt-pre and --ckpt-post, or pass --from-scratch "
                    "(or --resume with existing latest_pre.pth/latest_post.pth in --out-dir)."
                )
            pre.unet.load_state_dict(torch.load(args.ckpt_pre, map_location=cfg.device), strict=False)
            post.unet.load_state_dict(torch.load(args.ckpt_post, map_location=cfg.device), strict=False)
            start_epoch = 1
            logging.info("[ft_diff_pairs_84m] loaded ckpt pre=%s post=%s", args.ckpt_pre, args.ckpt_post)

    run_meta = {
        "pairs_pkl": os.path.abspath(args.pairs_pkl) if args.pairs_pkl else None,
        "con_file": getattr(cfg, "con_file", None),
        "out_dir": os.path.abspath(out_dir),
        "log_file": os.path.abspath(log_path),
        "from_scratch": bool(args.from_scratch),
        "resume": bool(args.resume),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "eval_batch": int(args.eval_batch),
        "lr": float(args.lr),
        "max_pairs": int(args.max_pairs),
        "valid_ratio": float(args.valid_ratio),
        "seed": int(args.seed),
        "lambda_x0": float(args.lambda_x0),
        "early_stop_patience": int(args.early_stop_patience),
    }
    if args.pairs_pkl:
        run_meta["split"] = "match_aligned"
        run_meta["sample_ratio"] = float(args.sample_ratio)
        run_meta["sample_seed"] = int(args.sample_seed)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    if args.pairs_pkl:
        train_pairs, valid_pairs = _load_pairs_from_pkl(
            args.pairs_pkl, int(args.max_pairs), float(args.sample_ratio), int(args.sample_seed)
        )
        logging.info(
            "[ft_diff_pairs_84m] pairs from %s split=match_aligned sample_ratio=%s sample_seed=%s | train=%d valid=%d",
            os.path.abspath(args.pairs_pkl),
            args.sample_ratio,
            args.sample_seed,
            len(train_pairs),
            len(valid_pairs),
        )
    else:
        train_pairs, valid_pairs = _load_pairs(args.max_pairs, args.valid_ratio, args.seed)
        logging.info(
            "[ft_diff_pairs_84m] pairs from traj_cmb pipeline %s | train=%d valid=%d",
            cfg.con_file,
            len(train_pairs),
            len(valid_pairs),
        )
    train_loader = DataLoader(PairDiffDataset(train_pairs), batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=_collate)
    valid_loader = DataLoader(PairDiffDataset(valid_pairs), batch_size=args.eval_batch, shuffle=False, num_workers=0, collate_fn=_collate)

    best_pre = float("inf")
    best_post = float("inf")
    patience = int(args.early_stop_patience)
    patience_ctr = 0
    for ep in range(start_epoch, int(args.epochs) + 1):
        pre.unet.train()
        post.unet.train()
        loss_list = []
        for info_pre, lab_pre, info_post, lab_post in train_loader:
            info_pre = info_pre.to(cfg.device)
            lab_pre = lab_pre.to(cfg.device)
            info_post = info_post.to(cfg.device)
            lab_post = lab_post.to(cfg.device)
            x_pre = torch.cat([info_pre, lab_pre], dim=2)
            x_post = torch.cat([info_post, lab_post], dim=2)
            l_pre, _ = pre.learn(x_pre, lam=cfg.lambda_x0)
            l_post, _ = post.learn(x_post, lam=cfg.lambda_x0)
            loss_list.append((float(l_pre) + float(l_post)) / 2.0)

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
                errs_pre.append(rmse_l2_m(pred_pre, lab_pre).detach().float().cpu())
                errs_post.append(rmse_l2_m(pred_post, lab_post).detach().float().cpu())
            rmse_pre_m = float(torch.stack(errs_pre).mean().item())
            rmse_post_m = float(torch.stack(errs_post).mean().item())

        improved_valid = (rmse_pre_m < best_pre) or (rmse_post_m < best_post)

        if ep % int(args.save_every) == 0:
            _save_sd(latest_pre, pre.unet.state_dict())
            _save_sd(latest_post, post.unet.state_dict())
            with open(latest_epoch, "w") as f:
                f.write(str(ep))

        if rmse_pre_m < best_pre:
            best_pre = rmse_pre_m
            sd_pre = pre.unet.state_dict()
            _save_sd(os.path.join(out_dir, f"best_pre_ep{ep}_rmse{best_pre:.3f}m.pth"), sd_pre)
            _save_sd(os.path.join(out_dir, "best_pre.pth"), sd_pre)
        if rmse_post_m < best_post:
            best_post = rmse_post_m
            sd_post = post.unet.state_dict()
            _save_sd(os.path.join(out_dir, f"best_post_ep{ep}_rmse{best_post:.3f}m.pth"), sd_post)
            _save_sd(os.path.join(out_dir, "best_post.pth"), sd_post)

        train_loss_mean = sum(loss_list) / max(1, len(loss_list))
        combined = (rmse_pre_m + rmse_post_m) / 2.0
        logging.info(
            "[ft_diff_pairs_84m] ep%d/%d train_loss_mean=%.6f valid(pre) rmse_m=%.3f | valid(post) rmse_m=%.3f "
            "| valid_combined=%.3f | best_pre=%.3f best_post=%.3f | es_improved=%s",
            ep,
            int(args.epochs),
            train_loss_mean,
            rmse_pre_m,
            rmse_post_m,
            combined,
            best_pre,
            best_post,
            improved_valid,
        )

        if patience > 0:
            if improved_valid:
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    logging.info(
                        "[ft_diff_pairs_84m] early_stop ep=%d patience=%d best_pre=%.3f best_post=%.3f",
                        ep,
                        patience,
                        best_pre,
                        best_post,
                    )
                    _save_sd(latest_pre, pre.unet.state_dict())
                    _save_sd(latest_post, post.unet.state_dict())
                    with open(latest_epoch, "w") as f:
                        f.write(str(ep))
                    with open(os.path.join(out_dir, "early_stop.json"), "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "epoch": ep,
                                "patience": patience,
                                "best_pre_rmse_m": best_pre,
                                "best_post_rmse_m": best_post,
                                "last_rmse_pre_m": rmse_pre_m,
                                "last_rmse_post_m": rmse_post_m,
                            },
                            f,
                            indent=2,
                        )
                    break

    with open(os.path.join(out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_pre_rmse_m": best_pre, "best_post_rmse_m": best_post}, f, indent=2)
    logging.info(
        "[ft_diff_pairs_84m] finished best_pre_rmse_m=%.3f best_post_rmse_m=%.3f -> %s %s",
        best_pre,
        best_post,
        os.path.join(out_dir, "best_pre.pth"),
        os.path.join(out_dir, "best_post.pth"),
    )


if __name__ == "__main__":
    main()

