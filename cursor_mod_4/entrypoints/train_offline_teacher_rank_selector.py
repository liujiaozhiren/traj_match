"""
Offline teacher + ranking for joint tail selection (cursor_mod_4).

Student (`JointTailRankSelector`):
  - Rich head/tail encoder: per-point head tokens + Conv1d/Transformer on L=4 tails + cross-attn to head.
  - Outputs per-arm scores (B, N).

Training:
  1) ListNet distillation on frozen teacher **k=1 joint-arm** logits (per sample, over N arms).
  2) Optional **mini m×m retrieval** alignment on each batch:
     - Teacher matrix uses frozen match with **vanilla first-k_select** tails (same as eval baseline).
     - Student matrix is a **softmax mixture** over per-arm k=1 teacher logits (stored), weighted by
       student softmax over arms — differentiable w.r.t. student scores, closer to retrieval than k=1 alone.

Valid mirrors `run_rl_tail_select.py` k×k retrieval vs vanilla first-k_select baseline.
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
from cursor_mod_4.match import MatchModel  # noqa: E402
from cursor_mod_4.proc import clean_pairs, proc_multi_single_traj  # noqa: E402
from cursor_mod_4.sample_db import sample_db  # noqa: E402
from cursor_mod_4.rl.joint_tail_rank_selector import JointTailRankSelector, RankSelectorCheckpoint  # noqa: E402


class PairIdxDataset(Dataset):
    def __init__(self, pairs, max_len: int | None = None):
        self.pairs = list(pairs)
        if max_len is not None:
            self.pairs = self.pairs[: int(max_len)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int):
        return i, self.pairs[i]


def _pair_to_info_xy8(item) -> tuple[torch.Tensor, torch.Tensor]:
    _, A, B, _, _, _, _ = item
    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    Brev = list(reversed(B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    return torch.tensor(A_xy, dtype=torch.float32), torch.tensor(B_xy, dtype=torch.float32)


def _collate_idx_info(batch):
    idxs, pre, post = [], [], []
    for idx, item in batch:
        ip, io = _pair_to_info_xy8(item)
        idxs.append(int(idx))
        pre.append(ip)
        post.append(io)
    return torch.tensor(idxs, dtype=torch.long), torch.stack(pre, 0), torch.stack(post, 0)


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


@torch.no_grad()
def _teacher_logits_joint_k1(
    match: MatchModel,
    *,
    info_pre: torch.Tensor,  # (B,L,2)
    info_post: torch.Tensor,  # (B,L,2)
    hydra_pre: torch.Tensor,  # (B,N,L2,2)
    hydra_post: torch.Tensor,  # (B,N,L2,2)
) -> torch.Tensor:
    B, N, L2, Dc = hydra_pre.shape
    assert hydra_post.shape == (B, N, L2, Dc)
    i_pre = info_pre.unsqueeze(1).expand(B, N, -1, -1).reshape(B * N, -1, 2)
    i_post = info_post.unsqueeze(1).expand(B, N, -1, -1).reshape(B * N, -1, 2)
    h_pre = hydra_pre.reshape(B * N, 1, L2, Dc)
    h_post = hydra_post.reshape(B * N, 1, L2, Dc)
    logits = _match_logits(match, i_pre, i_post, h_pre, h_post).view(B, N)
    return logits


@torch.no_grad()
def _L_k1_pair_matrix(match: MatchModel, ip: torch.Tensor, iq: torch.Tensor, hp: torch.Tensor, hq: torch.Tensor, *, N: int) -> torch.Tensor:
    """
    For all ordered pairs (i,j) in batch slice m, and each arm n, return k=1 match logit.
    Returns: (m, m, N) on the same device as ip.
    """
    m = ip.size(0)
    device = ip.device
    ii = torch.arange(m, device=device).view(m, 1).expand(m, m).reshape(-1)
    jj = torch.arange(m, device=device).view(1, m).expand(m, m).reshape(-1)
    i_pre_pair = ip[ii]
    i_post_pair = iq[jj]
    hp_pair = hp[ii]
    hq_pair = hq[jj]
    out = torch.empty(ii.numel(), N, device=device, dtype=torch.float32)
    for n in range(int(N)):
        h1 = hp_pair[:, n : n + 1, :, :]
        h2 = hq_pair[:, n : n + 1, :, :]
        out[:, n] = _match_logits(match, i_pre_pair, i_post_pair, h1, h2)
    return out.view(m, m, N)


@torch.no_grad()
def _T_vanilla_k16_matrix(
    match: MatchModel,
    ip: torch.Tensor,
    iq: torch.Tensor,
    hp: torch.Tensor,
    hq: torch.Tensor,
    *,
    ksel: int,
) -> torch.Tensor:
    m = ip.size(0)
    device = ip.device
    ii = torch.arange(m, device=device).view(m, 1).expand(m, m).reshape(-1)
    jj = torch.arange(m, device=device).view(1, m).expand(m, m).reshape(-1)
    i_pre_pair = ip[ii]
    i_post_pair = iq[jj]
    hp_pair = hp[ii][:, :ksel, :, :]
    hq_pair = hq[jj][:, :ksel, :, :]
    return _match_logits(match, i_pre_pair, i_post_pair, hp_pair, hq_pair).view(m, m)


def _mini_retrieval_loss(
    match: MatchModel,
    rank: JointTailRankSelector,
    *,
    info_pre: torch.Tensor,
    info_post: torch.Tensor,
    hydra_pre: torch.Tensor,
    hydra_post: torch.Tensor,
    m_in: int,
    k_select: int,
    tau: float,
    kl_w: float,
) -> torch.Tensor:
    """
    Differentiable mini retrieval on the first m samples of the current batch.
    """
    m = max(2, min(int(m_in), int(info_pre.size(0))))
    ip = info_pre[:m]
    iq = info_post[:m]
    hp = hydra_pre[:m]
    hq = hydra_post[:m]
    N = int(hp.size(1))

    with torch.no_grad():
        L_k1 = _L_k1_pair_matrix(match, ip, iq, hp, hq, N=N)  # (m,m,N)
        T_mat = _T_vanilla_k16_matrix(match, ip, iq, hp, hq, ksel=int(k_select))  # (m,m)

    ii = torch.arange(m, device=ip.device).view(m, 1).expand(m, m).reshape(-1)
    jj = torch.arange(m, device=ip.device).view(1, m).expand(m, m).reshape(-1)
    i_pre_exp = ip[ii]
    i_post_exp = iq[jj]
    hp_exp = hp[ii]
    hq_exp = hq[jj]
    u = rank.forward(i_pre_exp, i_post_exp, hp_exp, hq_exp)  # (m*m, N)
    w = F.softmax(u / float(tau), dim=1)
    S_flat = (w * L_k1.view(m * m, N)).sum(dim=1)
    S_mat = S_flat.view(m, m)

    gt = torch.arange(m, device=ip.device)
    ce = F.cross_entropy(S_mat, gt)
    p_rows = F.softmax(T_mat.detach() / float(tau), dim=1).clamp_min(1e-8)
    logq_rows = F.log_softmax(S_mat / float(tau), dim=1)
    kl_rows = F.kl_div(logq_rows, p_rows, reduction="batchmean")
    return ce + float(kl_w) * kl_rows


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--cache", type=str, required=True)
    p.add_argument("--match-ckpt", type=str, required=True)
    p.add_argument("--match-scheme", type=int, default=4, choices=[0, 1, 2, 3, 4])
    p.add_argument("--k-select", type=int, default=16)
    p.add_argument("--hydra-tail", type=int, default=64)
    p.add_argument("--train-batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--early-stop", type=int, default=10)
    p.add_argument("--teacher-temp", type=float, default=1.0, help="Temperature for softmax(teacher logits)")
    p.add_argument("--student-temp", type=float, default=1.0, help="Temperature for log_softmax(student scores)")
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-sample-frac", type=float, default=0.15)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--fixed-valid-seed", type=int, default=None)
    p.add_argument("--rep-dim", type=int, default=64, help="Must be divisible by 4 (MHA heads).")
    p.add_argument("--tail-mode", type=str, default="prepost_sep_fuse", choices=["pre_only", "prepost_sep_fuse"])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--rank-ckpt", type=str, default=None, help="Resume rank selector checkpoint")
    p.add_argument("--retrieval-m", type=int, default=8, help="Mini retrieval uses m×m on first m samples of each batch.")
    p.add_argument("--retrieval-loss-weight", type=float, default=0.5)
    p.add_argument("--retrieval-kl-weight", type=float, default=0.5, help="Row-wise KL to vanilla-k16 teacher distribution.")
    p.add_argument("--retrieval-tau", type=float, default=1.0, help="Softmax temperature for student mixture / KL.")
    return p.parse_args()


def _listnet_kl(student_scores: torch.Tensor, teacher_logits: torch.Tensor, *, t_teacher: float, t_student: float) -> torch.Tensor:
    """
    KL( softmax(teacher/t_t) || softmax(student/t_s) ) averaged over batch.
    """
    p = F.softmax(teacher_logits / float(t_teacher), dim=1).clamp_min(1e-12)
    logq = F.log_softmax(student_scores / float(t_student), dim=1)
    return (-(p * logq).sum(dim=1)).mean()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    if int(args.rep_dim) % 4 != 0:
        raise ValueError("--rep-dim must be divisible by 4 (multi-head attention).")

    os.makedirs(args.output_dir, exist_ok=True)

    train_pairs, valid_pairs = _load_pairs(float(args.sample_ratio), int(args.sample_seed))
    train_ds = PairIdxDataset(train_pairs)
    valid_ds = PairIdxDataset(valid_pairs)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.train_batch),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=_collate_idx_info,
    )

    cache = HydraOnlyCache.load(args.cache)
    N = min(int(args.hydra_tail), int(cache.hydra_pre.shape[1]))
    hydra_pre_all = cache.hydra_pre[:, :N]
    hydra_post_all = cache.hydra_post[:, :N]

    match = MatchModel(dim=128, scheme=int(args.match_scheme)).to(cfg.device)
    match.load_state_dict(torch.load(args.match_ckpt, map_location=cfg.device), strict=True)
    match.eval()
    match.freeze(True)

    rank = JointTailRankSelector(rep_dim=int(args.rep_dim), lr=float(args.lr), device=cfg.device, tail_mode=str(args.tail_mode))
    if args.rank_ckpt is not None:
        rank.load_state(RankSelectorCheckpoint.load(args.rank_ckpt))

    best_acc = -1.0
    best_ckpt = None
    best_ep = None
    stall = 0
    best_hr5 = best_hr10 = best_hr20 = 0.0
    best_r5_at_10 = best_r10_at_20 = 0.0
    best_valid_loss_ce = float("inf")
    best_acc_vanilla16 = 0.0
    best_hr5_vanilla16 = best_hr10_vanilla16 = best_hr20_vanilla16 = 0.0
    best_r5_at_10_vanilla16 = best_r10_at_20_vanilla16 = 0.0
    best_valid_loss_ce_vanilla16 = float("inf")

    for ep in range(1, int(args.max_epochs) + 1):
        rank.train()
        losses = []
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"[offline_rank] train ep{ep}/{args.max_epochs}", mininterval=0.5)
        for idxs, info_pre, info_post in pbar:
            B = info_pre.size(0)
            if B <= 1:
                continue
            info_pre = info_pre.to(cfg.device)
            info_post = info_post.to(cfg.device)
            hydra_pre = hydra_pre_all[idxs].to(cfg.device)
            hydra_post = hydra_post_all[idxs].to(cfg.device)

            with torch.no_grad():
                t_logits = _teacher_logits_joint_k1(match, info_pre=info_pre, info_post=info_post, hydra_pre=hydra_pre, hydra_post=hydra_post)

            s = rank.forward(info_pre, info_post, hydra_pre, hydra_post)
            loss = _listnet_kl(s, t_logits, t_teacher=float(args.teacher_temp), t_student=float(args.student_temp))
            if float(args.retrieval_loss_weight) > 0.0 and B >= 2:
                loss = loss + float(args.retrieval_loss_weight) * _mini_retrieval_loss(
                    match,
                    rank,
                    info_pre=info_pre,
                    info_post=info_post,
                    hydra_pre=hydra_pre,
                    hydra_post=hydra_post,
                    m_in=int(args.retrieval_m),
                    k_select=int(args.k_select),
                    tau=float(args.retrieval_tau),
                    kl_w=float(args.retrieval_kl_weight),
                )

            rank.opt.zero_grad(set_to_none=True)
            loss.backward()
            rank.opt.step()
            losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        pbar.close()

        # valid (same as run_rl_tail_select)
        rank.eval()
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
                    q_info_pre = valid_info_pre[samp[i0:i1]].to(cfg.device)
                    q_hydra_pre = valid_hydra_pre[samp[i0:i1]].to(cfg.device)

                    B1, B2 = q_info_pre.size(0), post_info.size(0)
                    i_pre = q_info_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
                    i_post = post_info.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
                    h_pre = q_hydra_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)
                    h_post = post_hydra.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)

                    ksel = int(args.k_select)
                    h_pre_base = h_pre[:, :ksel]
                    h_post_base = h_post[:, :ksel]
                    logits_base = _match_logits(match, i_pre, i_post, h_pre_base, h_post_base).view(B1, B2)

                    h_pre_sel, h_post_sel, _ = rank.choose_topk(i_pre, i_post, h_pre, h_post, k_select=ksel)
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
            best_ckpt = os.path.join(args.output_dir, f"best_rank_k{int(args.k_select)}_acc{best_acc:.4f}_ep{ep}.pt")
            rank.state().save(best_ckpt)
        else:
            stall += 1
            if stall >= int(args.early_stop):
                break

        print(
            f"[offline_rank] ep{ep}/{args.max_epochs} train_loss_mean={sum(losses)/max(1,len(losses)):.6f} "
            f"valid_loss_ce={valid_loss_ce:.6f} valid_acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
            f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f} "
            f"vanilla16_loss_ce={valid_loss_ce_base:.6f} vanilla16_acc={acc_base:.4f} vanilla16_HR5={hr5_base:.4f} vanilla16_HR10={hr10_base:.4f} vanilla16_HR20={hr20_base:.4f} "
            f"vanilla16_R5@10={r5_at_10_base:.4f} vanilla16_R10@20={r10_at_20_base:.4f} "
            f"best_acc={best_acc:.4f} stall={stall}/{args.early_stop} best_rank_ckpt={best_ckpt}",
            flush=True,
        )

    print(
        f"[offline_rank] done best_ep={best_ep} best_valid_loss_ce={best_valid_loss_ce:.6f} best_acc={best_acc:.4f} "
        f"best_HR5={best_hr5:.4f} best_HR10={best_hr10:.4f} best_HR20={best_hr20:.4f} "
        f"best_R5@10={best_r5_at_10:.4f} best_R10@20={best_r10_at_20:.4f} "
        f"best_vanilla16_loss_ce={best_valid_loss_ce_vanilla16:.6f} best_vanilla16_acc={best_acc_vanilla16:.4f} "
        f"best_vanilla16_HR5={best_hr5_vanilla16:.4f} best_vanilla16_HR10={best_hr10_vanilla16:.4f} best_vanilla16_HR20={best_hr20_vanilla16:.4f} "
        f"best_vanilla16_R5@10={best_r5_at_10_vanilla16:.4f} best_vanilla16_R10@20={best_r10_at_20_vanilla16:.4f} "
        f"best_rank_ckpt={best_ckpt}",
        flush=True,
    )


if __name__ == "__main__":
    main()
