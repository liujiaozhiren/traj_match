"""
Train **TAT 讨论用 baseline**（两段 pre/post 轨迹；见 ``tat_traj/model.py`` 设计说明）。

- **数据 / valid / post-eval**：与 ``train_dan_traj.py`` **同一套**（import 共享逻辑），便于和 DAN 横比。
- **模型**：``TATTrajTwoSegmentBaseline``（当前 phase 0：batch 内 ``(B,B)`` + 行 CE；非 TAT 原文 bi-level）。

Example::

    conda run -n traj_match python cursor_baseline/train_tat_traj.py \\
        --pairs-pkl cursor_mod_run_chengdu/data/all_pairs_v2_maxdtw.pkl \\
        --output-dir cursor_baseline/results/tat_chengdu_v2_run1 \\
        --train-batch 64 --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.modules["cfg"] = importlib.import_module("cursor_baseline.model_lib.cfg")
import cfg  # noqa: E402

from cursor_baseline.tat_traj.model import TATTrajTwoSegmentBaseline  # noqa: E402
from cursor_baseline.train_dan_traj import (  # noqa: E402
    TrainResult,
    PairListDataset,
    batch_standardize_lonlat_in_batch,
    collate_lonlat_noisy,
    dan_valid_metrics,
    post_eval_sample_fracs,
    _load_pairs_from_con_file,
    _load_pairs_from_pickle_list,
)


def run_tat(
    *,
    output_dir: str,
    pairs_pkl: str | None,
    train_batch: int,
    lr: float,
    max_epochs: int,
    early_stop: int,
    sample_ratio: float,
    sample_seed: int,
    rnn_hidden: int,
    embed_dim: int,
    rnn_layers: int,
    mlp_hidden: int,
    n_points: int,
    noise_std_deg: float,
    valid_sample_frac: float,
    valid_n_passes: int,
    valid_sample_seed: int | None,
    use_graph_update: bool,
    input_batch_standardize: bool = True,
) -> tuple[TrainResult, list]:
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(cfg.device)
    cpu = torch.device("cpu")

    if pairs_pkl:
        train_pairs, valid_pairs = _load_pairs_from_pickle_list(pairs_pkl, sample_ratio, sample_seed)
    else:
        train_pairs, valid_pairs = _load_pairs_from_con_file(sample_ratio, sample_seed)

    train_ds = PairListDataset(train_pairs)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_batch,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
        collate_fn=lambda batch: batch,
    )

    n_valid = len(valid_pairs)
    if n_valid == 0:
        raise RuntimeError("empty valid set")

    model = TATTrajTwoSegmentBaseline(
        rnn_hidden=rnn_hidden,
        embed_dim=embed_dim,
        rnn_layers=rnn_layers,
        mlp_hidden=mlp_hidden,
        use_graph_update=use_graph_update,
    ).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    use_cuda = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_acc = -1.0
    best_hr5 = best_hr10 = best_hr20 = best_r5_at_10 = best_r10_at_20 = -1.0
    best_ckpt = None
    stall = 0
    step_noise = 0
    tag = "tat_traj_baseline"
    last_epoch = 0

    print(
        f"[{tag}] NOTE: phase-0 = same train/valid protocol as train_dan_traj; "
        f"see tat_traj/model.py for TAT vs this collapse + loss TODOs.",
        flush=True,
    )
    print(
        f"[{tag}] input_batch_standardize={'ON' if input_batch_standardize else 'OFF'} "
        f"(same as train_dan_traj; --no-input-batch-standardize to disable)",
        flush=True,
    )

    for ep in range(1, max_epochs + 1):
        last_epoch = ep
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[{tag}] train ep{ep}/{max_epochs}", mininterval=0.5)
        for batch in pbar:
            B = len(batch)
            if B < 2:
                continue
            step_noise += 1
            noise_base = (ep * 1_000_003 + step_noise * 17 + sample_seed * 31) & ((1 << 31) - 1)
            pre, post = collate_lonlat_noisy(batch, n_points, noise_std_deg, noise_base, cpu)
            pre = pre.to(dev, non_blocking=True)
            post = post.to(dev, non_blocking=True)
            pre = batch_standardize_lonlat_in_batch(pre, enabled=input_batch_standardize)
            post = batch_standardize_lonlat_in_batch(post, enabled=input_batch_standardize)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                logits = model(pre, post)
            with torch.cuda.amp.autocast(enabled=False):
                loss = F.cross_entropy(logits.float(), torch.arange(B, device=dev))
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        pbar.close()

        model.eval()
        base_seed = int(valid_sample_seed) if valid_sample_seed is not None else (int(sample_seed) + int(ep))
        acc, hr5, hr10, hr20, r5_at_10, r10_at_20, valid_loss, k = dan_valid_metrics(
            model,
            valid_pairs,
            dev=dev,
            cpu=cpu,
            use_cuda=use_cuda,
            n_valid=n_valid,
            valid_sample_frac=valid_sample_frac,
            valid_n_passes=valid_n_passes,
            base_seed=base_seed,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            input_batch_standardize=input_batch_standardize,
        )
        n_passes = max(1, int(valid_n_passes))

        if hr5 > best_hr5:
            best_hr5 = hr5
        if hr10 > best_hr10:
            best_hr10 = hr10
        if hr20 > best_hr20:
            best_hr20 = hr20
        if r5_at_10 > best_r5_at_10:
            best_r5_at_10 = r5_at_10
        if r10_at_20 > best_r10_at_20:
            best_r10_at_20 = r10_at_20

        if acc > best_acc:
            best_acc = acc
            stall = 0
            best_ckpt = os.path.join(output_dir, f"best_{tag}_ep{ep}_acc{best_acc:.4f}.pth")
            torch.save(model.state_dict(), best_ckpt)
        else:
            stall += 1
            if stall >= early_stop:
                print(f"[{tag}] early stop at ep{ep}", flush=True)
                break

        print(
            f"[{tag}] ep{ep}/{max_epochs} train_loss_mean={sum(losses)/max(1,len(losses)):.6f} "
            f"valid_loss_ce={valid_loss:.6f} (mean over {n_passes} passes @ k={k}/{n_valid}) "
            f"valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
            f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f} best_acc={best_acc:.4f} stall={stall}/{early_stop}",
            flush=True,
        )

    return (
        TrainResult(
            best_acc=best_acc,
            best_hr5=best_hr5,
            best_hr10=best_hr10,
            best_hr20=best_hr20,
            best_r5_at_10=best_r5_at_10,
            best_r10_at_20=best_r10_at_20,
            best_ckpt=best_ckpt,
            last_epoch=last_epoch,
        ),
        valid_pairs,
    )


def parse_args():
    p = argparse.ArgumentParser(description="TAT two-segment traj baseline (see tat_traj/model.py).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--pairs-pkl",
        type=str,
        default=None,
        help="Optional. Same 7-tuple pickle as train_dan_traj.",
    )
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--early-stop", type=int, default=10)
    p.add_argument("--rnn-hidden", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--n-points", type=int, default=12)
    p.add_argument("--noise-std-deg", type=float, default=5e-5)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-frac", type=float, default=0.15)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--valid-sample-seed", type=int, default=None)
    p.add_argument(
        "--use-graph-update",
        action="store_true",
        help="Enable 2-node self-attn between pre/post embeddings (discussion placeholder).",
    )
    p.add_argument("--skip-post-eval", action="store_true")
    p.add_argument("--post-eval-fracs", type=str, default="0.15,0.10,0.05,0.02")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--no-input-batch-standardize",
        action="store_true",
        help="Disable per-batch (B,L,2) mean/std over batch before encoder (default: enabled, same as train_dan_traj).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    res, valid_pairs = run_tat(
        output_dir=args.output_dir,
        pairs_pkl=args.pairs_pkl,
        train_batch=int(args.train_batch),
        lr=float(args.lr),
        max_epochs=int(args.max_epochs),
        early_stop=int(args.early_stop),
        sample_ratio=float(args.sample_ratio),
        sample_seed=int(args.sample_seed),
        rnn_hidden=int(args.rnn_hidden),
        embed_dim=int(args.embed_dim),
        rnn_layers=int(args.rnn_layers),
        mlp_hidden=int(args.mlp_hidden),
        n_points=int(args.n_points),
        noise_std_deg=float(args.noise_std_deg),
        valid_sample_frac=float(args.valid_sample_frac),
        valid_n_passes=int(args.valid_n_passes),
        valid_sample_seed=args.valid_sample_seed,
        use_graph_update=bool(args.use_graph_update),
        input_batch_standardize=not bool(args.no_input_batch_standardize),
    )
    print(
        f"[tat_traj_baseline] done best_acc={res.best_acc:.4f} best_HR5={res.best_hr5:.4f} "
        f"best_HR10={res.best_hr10:.4f} best_HR20={res.best_hr20:.4f} "
        f"best_R5@10={res.best_r5_at_10:.4f} best_R10@20={res.best_r10_at_20:.4f} ckpt={res.best_ckpt}",
        flush=True,
    )

    if not args.skip_post_eval and res.best_ckpt and os.path.isfile(res.best_ckpt):
        dev = torch.device(cfg.device)
        cpu = torch.device("cpu")
        use_cuda = dev.type == "cuda"
        fracs = [float(x.strip()) for x in str(args.post_eval_fracs).split(",") if x.strip()]
        model = TATTrajTwoSegmentBaseline(
            rnn_hidden=int(args.rnn_hidden),
            embed_dim=int(args.embed_dim),
            rnn_layers=int(args.rnn_layers),
            mlp_hidden=int(args.mlp_hidden),
            use_graph_update=bool(args.use_graph_update),
        ).to(dev)
        model.load_state_dict(torch.load(res.best_ckpt, map_location=dev))
        model.eval()
        post_eval_sample_fracs(
            model,
            valid_pairs,
            dev=dev,
            cpu=cpu,
            use_cuda=use_cuda,
            fracs=fracs,
            valid_n_passes=int(args.valid_n_passes),
            valid_sample_seed=args.valid_sample_seed,
            sample_seed=int(args.sample_seed),
            train_last_epoch=int(res.last_epoch),
            n_points=int(args.n_points),
            noise_std_deg=float(args.noise_std_deg),
            input_batch_standardize=not bool(args.no_input_batch_standardize),
        )
    elif not args.skip_post_eval:
        print("[dan_mot_post_eval] skipped (no checkpoint saved)", flush=True)


if __name__ == "__main__":
    main()
