"""
Train trajectory completion: whole raw A -> whole B_hat (same T), supervised by raw B.

- Data: v2 maxdtw pickle (``pair_to_ab_lonlat``); no ``pair_pre_post_lonlat12`` extra noise.
- Train loss: mean per-step L2 in (lon, lat) degrees (same as valid dist).
- Valid: same subsample protocol as ``train_dan_traj.dan_valid_metrics`` (randperm_seed / noise_base
  formulas); ``(k,k)`` scores ``scores[i,j] = -mean_t ||pred_i - true_j||_2`` -> ``retrieval_metrics``
  (six scalars only; no CE in reporting).
- Training epochs: ``valid_sample_frac`` default 0.15, ``valid_n_passes`` default 3, early stop on ``acc``.
- After training: ``post_eval_sample_fracs`` equivalent (default 0.15,0.10,0.05,0.02), each frac with
  ``valid_n_passes`` averaged.

conda: traj_match

Example::

    conda run -n traj_match python cursor_baseline/train_completion_traj.py \\
        --pairs-pkl cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl \\
        --output-dir cursor_baseline/results/completion_attnmove_run1 \\
        --backbone attnmove --train-batch 64 --lr 1e-3 --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.modules["cfg"] = importlib.import_module("cursor_baseline.model_lib.cfg")
import cfg  # noqa: E402

from cursor_baseline.attnmove_model import AttnMoveCompletion  # noqa: E402
from cursor_baseline.attnmove_region import RegionVocab  # noqa: E402
from cursor_baseline.completion_models import build_completion_backbone  # noqa: E402
from cursor_baseline.completion_traj import (  # noqa: E402
    mean_step_l2,
    pair_to_ab_lonlat,
    scores_neg_mean_step_l2,
)
from cursor_baseline.metrics import retrieval_metrics  # noqa: E402
from cursor_baseline.train_dan_traj import (  # noqa: E402
    PairListDataset,
    TrainResult,
    _load_pairs_from_con_file,
    _load_pairs_from_pickle_list,
    batch_standardize_lonlat_in_batch,
)

ATTNMOVE_VOCAB_FN = "attnmove_region_vocab.json"


def _is_attnmove(backbone: str) -> bool:
    return str(backbone).lower().strip() == "attnmove"


def filter_completion_pairs(pairs: list, *, cpu: torch.device) -> list:
    """Keep only pairs where raw A/B lonlat tensors exist and len(A)==len(B)."""
    out: list = []
    for p in pairs:
        if pair_to_ab_lonlat(p, cpu) is not None:
            out.append(p)
    return out


def collate_raw_ab_batch(batch: list, device_cpu: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Stack (B,T,2) for A and B; skip bad items; require B>=2."""
    al: list[torch.Tensor] = []
    bl: list[torch.Tensor] = []
    for item in batch:
        pr = pair_to_ab_lonlat(item, device_cpu)
        if pr is None:
            continue
        al.append(pr[0])
        bl.append(pr[1])
    if len(al) < 2:
        return None
    ta, tb = al[0].shape[0], bl[0].shape[0]
    for a, b in zip(al, bl, strict=True):
        if a.shape[0] != ta or b.shape[0] != tb or a.shape[0] != b.shape[0]:
            return None
    return torch.stack(al, dim=0), torch.stack(bl, dim=0)


@torch.no_grad()
def completion_valid_one_subsample(
    model: torch.nn.Module,
    valid_pairs: list,
    *,
    dev: torch.device,
    cpu: torch.device,
    use_cuda: bool,
    n_valid: int,
    k: int,
    randperm_seed: int,
    input_batch_standardize: bool,
) -> dict[str, float]:
    """One randperm subsample of size k; build (k,k) scores from completion + raw B."""
    k = max(1, min(int(n_valid), int(k)))
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(int(randperm_seed))
    samp = torch.randperm(n_valid, generator=g_cpu)[:k].long()
    al: list[torch.Tensor] = []
    bl: list[torch.Tensor] = []
    for j in range(k):
        idx = int(samp[j].item())
        pr = pair_to_ab_lonlat(valid_pairs[idx], cpu)
        if pr is None:
            raise RuntimeError("valid_pairs must be pre-filtered; got None")
        al.append(pr[0])
        bl.append(pr[1])
    a_t = torch.stack(al, dim=0)
    b_t = torch.stack(bl, dim=0)
    a_t = batch_standardize_lonlat_in_batch(a_t, enabled=input_batch_standardize)
    b_t = batch_standardize_lonlat_in_batch(b_t, enabled=input_batch_standardize)
    a_t = a_t.to(dev, non_blocking=True)
    b_t = b_t.to(dev, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=use_cuda):
        pred = model(a_t)
    pred = pred.float()
    scores = scores_neg_mean_step_l2(pred, b_t)
    return retrieval_metrics(scores)


@torch.no_grad()
def completion_valid_metrics(
    model: torch.nn.Module,
    valid_pairs: list,
    *,
    dev: torch.device,
    cpu: torch.device,
    use_cuda: bool,
    n_valid: int,
    valid_sample_frac: float,
    valid_n_passes: int,
    base_seed: int,
    input_batch_standardize: bool,
) -> tuple[float, float, float, float, float, float, int]:
    """Mean over ``valid_n_passes`` independent subsamples (same formula as ``dan_valid_metrics``)."""
    k = max(1, min(n_valid, int(round(float(n_valid) * float(valid_sample_frac)))))
    n_passes = max(1, int(valid_n_passes))
    acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = 0.0
    for rep in range(n_passes):
        randperm_seed = int(base_seed) + rep * 7919
        m = completion_valid_one_subsample(
            model,
            valid_pairs,
            dev=dev,
            cpu=cpu,
            use_cuda=use_cuda,
            n_valid=n_valid,
            k=k,
            randperm_seed=randperm_seed,
            input_batch_standardize=input_batch_standardize,
        )
        acc_s += m["acc"]
        hr5_s += m["hr5"]
        hr10_s += m["hr10"]
        hr20_s += m["hr20"]
        r51_s += m["r5_at_10"]
        r102_s += m["r10_at_20"]

    return (
        acc_s / n_passes,
        hr5_s / n_passes,
        hr10_s / n_passes,
        hr20_s / n_passes,
        r51_s / n_passes,
        r102_s / n_passes,
        k,
    )


def completion_post_eval_sample_fracs(
    model: torch.nn.Module,
    valid_pairs: list,
    *,
    dev: torch.device,
    cpu: torch.device,
    use_cuda: bool,
    fracs: list[float],
    valid_n_passes: int,
    valid_sample_seed: int | None,
    sample_seed: int,
    train_last_epoch: int,
    input_batch_standardize: bool,
) -> None:
    """After training: each frac -> ``completion_valid_metrics`` (``valid_n_passes`` averaged)."""
    n_valid = len(valid_pairs)
    tag = "[completion_post_eval]"
    base_seed = int(valid_sample_seed) if valid_sample_seed is not None else int(sample_seed) + int(train_last_epoch)
    n_passes = max(1, int(valid_n_passes))
    print(
        f"{tag} protocol=completion_valid_metrics; n_valid={n_valid} n_passes={n_passes} base_seed={base_seed}",
        flush=True,
    )
    for frac in fracs:
        acc, hr5, hr10, hr20, r5_at_10, r10_at_20, k = completion_valid_metrics(
            model,
            valid_pairs,
            dev=dev,
            cpu=cpu,
            use_cuda=use_cuda,
            n_valid=n_valid,
            valid_sample_frac=float(frac),
            valid_n_passes=n_passes,
            base_seed=base_seed,
            input_batch_standardize=input_batch_standardize,
        )
        print(
            f"{tag} valid_sample_frac={frac:.4f} "
            f"(mean over {n_passes} passes @ k={k}/{n_valid}) "
            f"valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
            f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f}",
            flush=True,
        )


def run(
    *,
    output_dir: str,
    pairs_pkl: str | None,
    backbone: str,
    train_batch: int,
    lr: float,
    max_epochs: int,
    early_stop: int,
    sample_ratio: float,
    sample_seed: int,
    rnn_hidden: int,
    rnn_layers: int,
    valid_sample_frac: float,
    valid_n_passes: int,
    valid_sample_seed: int | None,
    use_amp: bool = False,
    input_batch_standardize: bool = True,
    attn_hidden: int = 64,
    attn_layers: int = 2,
    attn_heads: int = 1,
    attn_drop: float = 0.3,
    attn_fb_drop: float = 0.3,
    label_confidence: float = 0.9,
    reg_lambda: float = 0.001,
    reg_dist_lambda: float = 1.0,
    region_cell_deg: float = 0.0002,
) -> tuple[TrainResult, list]:
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(cfg.device)
    cpu = torch.device("cpu")

    if pairs_pkl:
        train_pairs, valid_pairs = _load_pairs_from_pickle_list(pairs_pkl, sample_ratio, sample_seed)
    else:
        train_pairs, valid_pairs = _load_pairs_from_con_file(sample_ratio, sample_seed)

    train_pairs = filter_completion_pairs(train_pairs, cpu=cpu)
    valid_pairs = filter_completion_pairs(valid_pairs, cpu=cpu)
    if not train_pairs:
        raise RuntimeError("no train pairs after completion filter")
    if not valid_pairs:
        raise RuntimeError("no valid pairs after completion filter")

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
    is_attn = _is_attnmove(backbone)
    if is_attn:
        input_batch_standardize = False

    region_vocab: RegionVocab | None = None
    if is_attn:
        region_vocab = RegionVocab.fit_from_pairs(train_pairs, cpu=cpu, cell_deg=float(region_cell_deg))
        vocab_path = os.path.join(output_dir, ATTNMOVE_VOCAB_FN)
        region_vocab.save(vocab_path)
        print(
            f"[completion_attnmove] fitted region vocab size={region_vocab.vocab_size} "
            f"cell_deg={region_cell_deg} saved={vocab_path}",
            flush=True,
        )

    model = build_completion_backbone(
        backbone,
        rnn_hidden=rnn_hidden,
        rnn_layers=rnn_layers,
        region_vocab=region_vocab,
        attn_hidden=attn_hidden,
        attn_layers=attn_layers,
        attn_heads=attn_heads,
        attn_drop=attn_drop,
        attn_fb_drop=attn_fb_drop,
        label_confidence=label_confidence,
        reg_lambda=reg_lambda,
        reg_dist_lambda=reg_dist_lambda,
    ).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    use_cuda = dev.type == "cuda"
    train_amp = bool(use_cuda and use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=train_amp)

    best_acc = -1.0
    best_hr5 = best_hr10 = best_hr20 = best_r5_at_10 = best_r10_at_20 = -1.0
    best_ckpt: str | None = None
    stall = 0
    tag = f"completion_{str(backbone).lower()}"
    last_epoch = 0

    print(
        f"[{tag}] input_batch_standardize={'ON' if input_batch_standardize else 'OFF'} "
        f"(per (L,2) over batch on A and B separately; --no-input-batch-standardize to disable)",
        flush=True,
    )

    for ep in range(1, max_epochs + 1):
        last_epoch = ep
        model.train()
        losses: list[float] = []
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[{tag}] train ep{ep}/{max_epochs}", mininterval=0.5)
        for batch in pbar:
            stacked = collate_raw_ab_batch(batch, cpu)
            if stacked is None:
                continue
            a_xy, b_xy = stacked
            a_xy = a_xy.to(dev, non_blocking=True)
            b_xy = b_xy.to(dev, non_blocking=True)
            a_xy = batch_standardize_lonlat_in_batch(a_xy, enabled=input_batch_standardize)
            b_xy = batch_standardize_lonlat_in_batch(b_xy, enabled=input_batch_standardize)
            optim.zero_grad(set_to_none=True)
            if is_attn:
                assert isinstance(model, AttnMoveCompletion)
                loss, _parts = model.compute_loss(a_xy.float(), b_xy.float())
            else:
                with torch.cuda.amp.autocast(enabled=train_amp):
                    pred = model(a_xy)
                with torch.cuda.amp.autocast(enabled=False):
                    loss = mean_step_l2(pred.float(), b_xy.float())
            if train_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{losses[-1]:.6f}")
        pbar.close()

        model.eval()
        base_seed = int(valid_sample_seed) if valid_sample_seed is not None else (int(sample_seed) + int(ep))
        acc, hr5, hr10, hr20, r5_at_10, r10_at_20, k = completion_valid_metrics(
            model,
            valid_pairs,
            dev=dev,
            cpu=cpu,
            use_cuda=use_cuda,
            n_valid=n_valid,
            valid_sample_frac=valid_sample_frac,
            valid_n_passes=valid_n_passes,
            base_seed=base_seed,
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
            f"valid (mean over {n_passes} passes @ k={k}/{n_valid}) "
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
    p = argparse.ArgumentParser(description="Train raw A->B trajectory completion (see module docstring).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--pairs-pkl",
        type=str,
        default=None,
        help="Pickle list of 7-tuples (v2 maxdtw). If omitted, use cfg.con_file pipeline.",
    )
    p.add_argument(
        "--backbone",
        type=str,
        default="gru",
        help="gru | attnmove | rntrajrec (rntrajrec TODO).",
    )
    p.add_argument("--attn-hidden", type=int, default=64)
    p.add_argument("--attn-layers", type=int, default=2)
    p.add_argument("--attn-heads", type=int, default=1)
    p.add_argument("--attn-drop", type=float, default=0.3)
    p.add_argument("--attn-fb-drop", type=float, default=0.3)
    p.add_argument("--label-confidence", type=float, default=0.9)
    p.add_argument("--reg-lambda", type=float, default=0.001, help="L2 weight reg (attnmove).")
    p.add_argument("--reg-dist-lambda", type=float, default=1.0, help="region_distance reg scale (attnmove).")
    p.add_argument("--region-cell-deg", type=float, default=0.0002)
    p.add_argument("--train-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--early-stop", type=int, default=10, help="Stall epochs on valid acc.")
    p.add_argument("--rnn-hidden", type=int, default=128)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-frac", type=float, default=0.15)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--valid-sample-seed", type=int, default=None)
    p.add_argument("--skip-post-eval", action="store_true")
    p.add_argument(
        "--post-eval-fracs",
        type=str,
        default="0.15,0.10,0.05,0.02",
        help="Comma-separated valid_sample_frac for post-train report.",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--amp", action="store_true", help="CUDA autocast + GradScaler during training.")
    p.add_argument(
        "--no-input-batch-standardize",
        action="store_true",
        help="Disable per-batch (B,L,2) mean/std on A and B.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    res, valid_pairs = run(
        output_dir=args.output_dir,
        pairs_pkl=args.pairs_pkl,
        backbone=str(args.backbone),
        train_batch=int(args.train_batch),
        lr=float(args.lr),
        max_epochs=int(args.max_epochs),
        early_stop=int(args.early_stop),
        sample_ratio=float(args.sample_ratio),
        sample_seed=int(args.sample_seed),
        rnn_hidden=int(args.rnn_hidden),
        rnn_layers=int(args.rnn_layers),
        valid_sample_frac=float(args.valid_sample_frac),
        valid_n_passes=int(args.valid_n_passes),
        valid_sample_seed=args.valid_sample_seed,
        use_amp=bool(args.amp),
        input_batch_standardize=not bool(args.no_input_batch_standardize),
        attn_hidden=int(args.attn_hidden),
        attn_layers=int(args.attn_layers),
        attn_heads=int(args.attn_heads),
        attn_drop=float(args.attn_drop),
        attn_fb_drop=float(args.attn_fb_drop),
        label_confidence=float(args.label_confidence),
        reg_lambda=float(args.reg_lambda),
        reg_dist_lambda=float(args.reg_dist_lambda),
        region_cell_deg=float(args.region_cell_deg),
    )
    print(
        f"[completion] done best_acc={res.best_acc:.4f} best_HR5={res.best_hr5:.4f} "
        f"best_HR10={res.best_hr10:.4f} best_HR20={res.best_hr20:.4f} "
        f"best_R5@10={res.best_r5_at_10:.4f} best_R10@20={res.best_r10_at_20:.4f} ckpt={res.best_ckpt}",
        flush=True,
    )

    ckpt_ok = res.best_ckpt is not None and os.path.isfile(res.best_ckpt)
    want_post = not bool(args.skip_post_eval)
    if ckpt_ok and want_post:
        dev = torch.device(cfg.device)
        cpu = torch.device("cpu")
        use_cuda = dev.type == "cuda"
        region_vocab = None
        if _is_attnmove(str(args.backbone)):
            vp = os.path.join(args.output_dir, ATTNMOVE_VOCAB_FN)
            region_vocab = RegionVocab.load(vp)
        model = build_completion_backbone(
            str(args.backbone),
            rnn_hidden=int(args.rnn_hidden),
            rnn_layers=int(args.rnn_layers),
            region_vocab=region_vocab,
            attn_hidden=int(args.attn_hidden),
            attn_layers=int(args.attn_layers),
            attn_heads=int(args.attn_heads),
            attn_drop=float(args.attn_drop),
            attn_fb_drop=float(args.attn_fb_drop),
            label_confidence=float(args.label_confidence),
            reg_lambda=float(args.reg_lambda),
            reg_dist_lambda=float(args.reg_dist_lambda),
        ).to(dev)
        model.load_state_dict(torch.load(res.best_ckpt, map_location=dev))
        model.eval()
        fracs = [float(x.strip()) for x in str(args.post_eval_fracs).split(",") if x.strip()]
        post_std = not bool(args.no_input_batch_standardize)
        if _is_attnmove(str(args.backbone)):
            post_std = False
        completion_post_eval_sample_fracs(
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
            input_batch_standardize=post_std,
        )
    elif want_post and not ckpt_ok:
        print("[completion_post_eval] skipped (no checkpoint saved)", flush=True)


if __name__ == "__main__":
    main()
