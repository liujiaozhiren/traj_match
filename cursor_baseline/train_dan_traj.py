"""
Train simplified DAN-mot traj matcher (see dan_mot_traj/model.py module docstring).

Valid 与 train_repr_baseline 一致：k×k scores → retrieval_metrics + CE valid_loss。

**Project defaults（与 repr baseline 对齐）**

- Train / valid **pair 切分**：``sample_ratio=0.1``（10% 作 valid，其余 train）。
- **Early stop**：stall **10** 个 epoch（``--early-stop 10``）。
- **训练过程中**监控 valid：``valid_sample_frac=0.15``（k≈15% valid），``valid_n_passes=3`` 取平均。
- **训练结束后**（可选）：对 ``--post-eval-fracs`` 每一档调用 **与训练 valid 完全相同** 的 ``dan_valid_metrics``（无噪声–label 混合）。另可 **仅** 在加载 best 权重后加 ``--valid-noise-blend-sweep-post`` 做六档 α 混合 valid（训练期 valid 不做混合）。未设 ``--valid-sample-seed`` 时 post-eval / sweep 的 ``base_seed`` 与 **最后一轮训练 epoch** 一致：``sample_seed + last_epoch``。
- **默认**：对 pre/post 各自在 batch 维上做 ``(L,2)`` 位置的减均值/除标准差（避免同城坐标 batch 内过近导致 embedding 塌缩、loss 贴在 ``log(B)``）；可用 ``--no-input-batch-standardize`` 关闭。

conda:  traj_match  （见 .cursor/rules/traj_match-conda.mdc）

Example::

    conda run -n traj_match python cursor_baseline/train_dan_traj.py \\
        --pairs-pkl cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl \\
        --output-dir cursor_baseline/results/dan_mot_simplified_run1 \\
        --train-batch 64 --lr 1e-4 --device cuda:0
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from dataclasses import dataclass
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

from cursor_baseline.dan_mot_traj.model import DANMotTrajSimplified  # noqa: E402
from cursor_baseline.lonlat_traj import pair_pre_post_lonlat12  # noqa: E402
from cursor_baseline.metrics import retrieval_metrics  # noqa: E402
from cursor_baseline.model_lib.proc_pairs import clean_pairs, proc_multi_single_traj  # noqa: E402
from cursor_baseline.model_lib.sample_db import sample_db  # noqa: E402


def _load_pairs_from_con_file(sample_ratio: float, sample_seed: int):
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    valid = sample_db(all_pairs, ratio=sample_ratio, seed=sample_seed)
    train = [p for p in all_pairs if p not in valid]
    return train, valid


def _load_pairs_from_pickle_list(path: str, sample_ratio: float, sample_seed: int):
    """
    Load a list of 7-tuples (same as clean_pairs output / v2 maxdtw), e.g.
    cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl
    """
    raw = pickle.load(open(path, "rb"))
    if not isinstance(raw, list):
        raise TypeError(f"expected list in pickle, got {type(raw)}")
    all_pairs = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        a, b = item[1], item[2]
        if len(a) < 3 or len(b) < 3:
            continue
        all_pairs.append(tuple(item) if isinstance(item, list) else item)
    if not all_pairs:
        raise RuntimeError(f"no usable pairs after filter: {path}")
    valid = sample_db(all_pairs, ratio=sample_ratio, seed=sample_seed)
    train = [p for p in all_pairs if p not in valid]
    return train, valid


class PairListDataset(Dataset):
    def __init__(self, pairs: list):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        return self.pairs[i]


def collate_lonlat_noisy(batch: list, n_points: int, noise_std_deg: float, noise_base: int, device_cpu: torch.device):
    pres: list[torch.Tensor] = []
    posts: list[torch.Tensor] = []
    for j, item in enumerate(batch):
        pre, post = pair_pre_post_lonlat12(
            item,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            noise_seed_pre=int(noise_base) + j * 3,
            noise_seed_post=int(noise_base) + j * 3 + 1,
            device=device_cpu,
        )
        pres.append(pre)
        posts.append(post)
    return torch.stack(pres, dim=0), torch.stack(posts, dim=0)


def batch_standardize_lonlat_in_batch(x: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    """
    ``x``: ``(B, L, 2)`` lon/lat degrees（仅一条分支：pre **或** post）。

    同一 batch 内绝对坐标往往落在极小地理范围内，展平后样本方向几乎一致，BiGRU+mean pool
    后 embedding 在 batch 维上方差趋近 0，``(B,B)`` 每行 logits 几乎常数 → CE 对 backbone 梯度极小。
    在 **batch 维** 上对每一个固定的 ``(L, 通道)`` 位置做减均值、除标准差（pre/post **分别**做），
    保留轨迹形状相对差异，且 train / valid / post-eval 对「当前子 batch」约定一致。

    ``B<2`` 或未启用时原样返回。关闭：``--no-input-batch-standardize``。
    """
    if not enabled or x.size(0) < 2:
        return x
    mu = x.mean(dim=0, keepdim=True)
    sig = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    return (x - mu) / sig


@dataclass
class TrainResult:
    best_acc: float
    best_hr5: float
    best_hr10: float
    best_hr20: float
    best_r5_at_10: float
    best_r10_at_20: float
    best_ckpt: str | None
    last_epoch: int


@torch.no_grad()
def dan_valid_one_subsample(
    model: torch.nn.Module,
    valid_pairs: list,
    *,
    dev: torch.device,
    cpu: torch.device,
    use_cuda: bool,
    n_valid: int,
    k: int,
    randperm_seed: int,
    noise_base: int,
    n_points: int,
    noise_std_deg: float,
    input_batch_standardize: bool,
    noise_blend_alpha: float | None = None,
    noise_blend_check_label_ts: bool = False,
) -> dict[str, float]:
    """
    **一次** valid 子采样（与 ``train_repr_baseline`` 里单次 ``rep`` 相同）：
    用 ``randperm_seed`` 从 valid 里抽 **k** 条轨迹对 → 构图 → 一次 ``(k,k)`` 匹配分数
    → ``retrieval_metrics``（HR5/HR10/HR20 即 top-5/10/20 命中率，同 ft_match）。
    """
    k = max(1, min(int(n_valid), int(k)))
    g_cpu = torch.Generator(device="cpu")
    g_cpu.manual_seed(int(randperm_seed))
    samp = torch.randperm(n_valid, generator=g_cpu)[:k].long()
    pres, posts = [], []
    for j in range(k):
        idx = int(samp[j].item())
        pii, poo = pair_pre_post_lonlat12(
            valid_pairs[idx],
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            noise_seed_pre=int(noise_base) + j * 3,
            noise_seed_post=int(noise_base) + j * 3 + 1,
            device=cpu,
            noise_blend_alpha=noise_blend_alpha,
            noise_blend_check_label_ts=noise_blend_check_label_ts,
        )
        pres.append(pii)
        posts.append(poo)
    pre_t = torch.stack(pres, dim=0).to(dev)
    post_t = torch.stack(posts, dim=0).to(dev)
    pre_t = batch_standardize_lonlat_in_batch(pre_t, enabled=input_batch_standardize)
    post_t = batch_standardize_lonlat_in_batch(post_t, enabled=input_batch_standardize)
    with torch.cuda.amp.autocast(enabled=use_cuda):
        scores = model(pre_t, post_t)
    scores = scores.float()
    m = retrieval_metrics(scores)
    m["valid_loss_ce"] = float(F.cross_entropy(scores, torch.arange(k, device=dev)).item())
    return m


@torch.no_grad()
def dan_valid_metrics(
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
    n_points: int,
    noise_std_deg: float,
    input_batch_standardize: bool,
    noise_blend_alpha: float | None = None,
    noise_blend_check_label_ts: bool = False,
) -> tuple[float, float, float, float, float, float, float, int]:
    """
    训练期 valid：与 ``train_repr_baseline`` 一致——固定 ``k``，``valid_n_passes`` 次 **独立**
    randperm 子采样，对每次的 ``(k,k)`` 指标做算术平均。
    """
    k = max(1, min(n_valid, int(round(float(n_valid) * float(valid_sample_frac)))))
    n_passes = max(1, int(valid_n_passes))
    acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = vloss_s = 0.0
    for rep in range(n_passes):
        randperm_seed = int(base_seed) + rep * 7919
        noise_base = int(base_seed) * 10007 + rep * 1_000_003 + k * 17
        m = dan_valid_one_subsample(
            model,
            valid_pairs,
            dev=dev,
            cpu=cpu,
            use_cuda=use_cuda,
            n_valid=n_valid,
            k=k,
            randperm_seed=randperm_seed,
            noise_base=noise_base,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            input_batch_standardize=input_batch_standardize,
            noise_blend_alpha=noise_blend_alpha,
            noise_blend_check_label_ts=noise_blend_check_label_ts,
        )
        acc_s += m["acc"]
        hr5_s += m["hr5"]
        hr10_s += m["hr10"]
        hr20_s += m["hr20"]
        r51_s += m["r5_at_10"]
        r102_s += m["r10_at_20"]
        vloss_s += m["valid_loss_ce"]

    acc = acc_s / n_passes
    hr5 = hr5_s / n_passes
    hr10 = hr10_s / n_passes
    hr20 = hr20_s / n_passes
    r5_at_10 = r51_s / n_passes
    r10_at_20 = r102_s / n_passes
    valid_loss = vloss_s / n_passes
    return acc, hr5, hr10, hr20, r5_at_10, r10_at_20, valid_loss, k


def post_eval_sample_fracs(
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
    n_points: int,
    noise_std_deg: float,
    input_batch_standardize: bool,
) -> None:
    """
    与 ``train_repr_baseline`` / 训练 valid **同一实现**：每档 ``frac`` 调用
    ``dan_valid_metrics``（``valid_n_passes`` 次 rep、同一套 ``randperm_seed`` / ``noise_base`` 公式；无噪声混合）。

    ``base_seed`` 与训练时一致：``valid_sample_seed`` 若未设则为 ``sample_seed + train_last_epoch``
    （同 ``train_repr_baseline`` 里 ``valid_sample_seed is None`` 时的 ``sample_seed + ep``）。
    """
    n_valid = len(valid_pairs)
    tag = "[dan_mot_post_eval]"
    base_seed = (
        int(valid_sample_seed)
        if valid_sample_seed is not None
        else int(sample_seed) + int(train_last_epoch)
    )
    n_passes = max(1, int(valid_n_passes))
    print(
        f"{tag} protocol=train_repr_baseline valid loop (dan_valid_metrics); "
        f"n_valid={n_valid} n_passes={n_passes} base_seed={base_seed}",
        flush=True,
    )
    for frac in fracs:
        acc, hr5, hr10, hr20, r5_at_10, r10_at_20, valid_loss, k = dan_valid_metrics(
            model,
            valid_pairs,
            dev=dev,
            cpu=cpu,
            use_cuda=use_cuda,
            n_valid=n_valid,
            valid_sample_frac=float(frac),
            valid_n_passes=n_passes,
            base_seed=base_seed,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            input_batch_standardize=input_batch_standardize,
        )
        print(
            f"{tag} valid_sample_frac={frac:.4f} "
            f"valid_loss_ce={valid_loss:.6f} (mean over {n_passes} passes @ k={k}/{n_valid}) "
            f"valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
            f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f}",
            flush=True,
        )


def run(
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
    use_amp: bool = False,
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

    # 【简化版】DAN-mot；完整版见 dan_mot_traj/model.py 顶部说明
    model = DANMotTrajSimplified(
        rnn_hidden=rnn_hidden,
        embed_dim=embed_dim,
        rnn_layers=rnn_layers,
        mlp_hidden=mlp_hidden,
    ).to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    use_cuda = dev.type == "cuda"
    train_amp = bool(use_cuda and use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=train_amp)

    best_acc = -1.0
    best_hr5 = best_hr10 = best_hr20 = best_r5_at_10 = best_r10_at_20 = -1.0
    best_ckpt = None
    stall = 0
    step_noise = 0
    tag = "dan_mot_simplified"
    last_epoch = 0

    print(
        f"[{tag}] input_batch_standardize={'ON' if input_batch_standardize else 'OFF'} "
        f"(per (L,2) over batch on pre/post; disable: --no-input-batch-standardize)",
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
            with torch.cuda.amp.autocast(enabled=train_amp):
                logits = model(pre, post)
            with torch.cuda.amp.autocast(enabled=False):
                loss = F.cross_entropy(logits.float(), torch.arange(B, device=dev))
            if train_amp:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
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
    p = argparse.ArgumentParser(description="Simplified DAN-mot trajectory training (see dan_mot_traj/model.py).")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--pairs-pkl",
        type=str,
        default=None,
        help="Optional. Pickle list of 7-tuples (clean_pairs / v2 maxdtw). If omitted, use cfg.con_file pipeline.",
    )
    p.add_argument("--train-batch", type=int, default=64, help="Smaller if OOM (pairing is O(n^2)).")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument(
        "--early-stop",
        type=int,
        default=10,
        help="Stall epochs on valid acc (project default: 10).",
    )
    p.add_argument("--rnn-hidden", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--rnn-layers", type=int, default=1)
    p.add_argument("--mlp-hidden", type=int, default=256)
    p.add_argument("--n-points", type=int, default=12)
    p.add_argument("--noise-std-deg", type=float, default=5e-5)
    p.add_argument(
        "--sample-ratio",
        type=float,
        default=0.1,
        help="Fraction of all pairs held out as valid (train/valid split; default 0.1).",
    )
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument(
        "--valid-sample-frac",
        type=float,
        default=0.15,
        help="During training: k ≈ this fraction of valid set (default 0.15) for stall curve.",
    )
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--valid-sample-seed", type=int, default=None)
    p.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Skip multi-fraction report after training (default: run it when a checkpoint exists).",
    )
    p.add_argument(
        "--post-eval-fracs",
        type=str,
        default="0.15,0.10,0.05,0.02",
        help="Comma-separated valid_sample_frac values for post-train report (default: 15%%,10%%,5%%,2%%).",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast + GradScaler during training (default: fp32 for stable grads on the (B,B) CE head).",
    )
    p.add_argument(
        "--no-input-batch-standardize",
        action="store_true",
        help="Disable per-batch (B,L,2) mean/std over batch before encoder (default: enabled to avoid within-batch embedding collapse).",
    )
    p.add_argument(
        "--valid-noise-blend-sweep-post",
        action="store_true",
        help=(
            "After training only: reload best checkpoint and run dan_valid_metrics once per α in "
            "{1,0.8,...,0} (noisy/label blend; see cursor_mod_4.traj_noise_blend). "
            "Does not affect per-epoch training valid or post-eval fracs."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    res, valid_pairs = run(
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
        use_amp=bool(args.amp),
        input_batch_standardize=not bool(args.no_input_batch_standardize),
    )
    print(
        f"[dan_mot_simplified] done best_acc={res.best_acc:.4f} best_HR5={res.best_hr5:.4f} "
        f"best_HR10={res.best_hr10:.4f} best_HR20={res.best_hr20:.4f} "
        f"best_R5@10={res.best_r5_at_10:.4f} best_R10@20={res.best_r10_at_20:.4f} ckpt={res.best_ckpt}",
        flush=True,
    )

    ckpt_ok = bool(res.best_ckpt) and os.path.isfile(str(res.best_ckpt))
    want_sweep = bool(args.valid_noise_blend_sweep_post)
    want_post_eval = not bool(args.skip_post_eval)

    if ckpt_ok and (want_sweep or want_post_eval):
        dev = torch.device(cfg.device)
        cpu = torch.device("cpu")
        use_cuda = dev.type == "cuda"
        fracs = [float(x.strip()) for x in str(args.post_eval_fracs).split(",") if x.strip()]
        model = DANMotTrajSimplified(
            rnn_hidden=int(args.rnn_hidden),
            embed_dim=int(args.embed_dim),
            rnn_layers=int(args.rnn_layers),
            mlp_hidden=int(args.mlp_hidden),
        ).to(dev)
        model.load_state_dict(torch.load(res.best_ckpt, map_location=dev))
        model.eval()
        if want_sweep:
            from cursor_mod_4.traj_noise_blend import NOISE_BLEND_LEVEL_ALPHAS  # noqa: PLC0415

            n_valid = len(valid_pairs)
            n_passes = max(1, int(args.valid_n_passes))
            base_seed = (
                int(args.valid_sample_seed)
                if args.valid_sample_seed is not None
                else int(args.sample_seed) + int(res.last_epoch)
            )
            tag = "[dan_mot_noise_blend_sweep]"
            print(f"{tag} protocol=dan_valid_metrics alphas={NOISE_BLEND_LEVEL_ALPHAS} base_seed={base_seed}", flush=True)
            for a in NOISE_BLEND_LEVEL_ALPHAS:
                acc, hr5, hr10, hr20, r5_at_10, r10_at_20, valid_loss, k = dan_valid_metrics(
                    model,
                    valid_pairs,
                    dev=dev,
                    cpu=cpu,
                    use_cuda=use_cuda,
                    n_valid=n_valid,
                    valid_sample_frac=float(args.valid_sample_frac),
                    valid_n_passes=int(args.valid_n_passes),
                    base_seed=base_seed,
                    n_points=int(args.n_points),
                    noise_std_deg=float(args.noise_std_deg),
                    input_batch_standardize=not bool(args.no_input_batch_standardize),
                    noise_blend_alpha=float(a),
                )
                print(
                    f"{tag} alpha={a:g} valid_loss_ce={valid_loss:.6f} (mean over {n_passes} passes @ k={k}/{n_valid}) "
                    f"valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
                    f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f}",
                    flush=True,
                )
        if want_post_eval:
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
    else:
        if want_sweep:
            print("[dan_mot_noise_blend_sweep] skipped (no checkpoint saved)", flush=True)
        if want_post_eval:
            print("[dan_mot_post_eval] skipped (no checkpoint saved)", flush=True)


if __name__ == "__main__":
    main()
