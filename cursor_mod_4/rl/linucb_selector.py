from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg


def _gather_by_index(hydra: torch.Tensor, idx_mat: torch.Tensor) -> torch.Tensor:
    """
    hydra: (B, N, L2, Dc)
    idx_mat: (B, K)
    returns: (B, K, L2, Dc)
    """
    B = hydra.shape[0]
    K = idx_mat.shape[1]
    idx = idx_mat.unsqueeze(-1)  # (B,K,1)
    expand_shape = [B, K] + [1] * (hydra.dim() - 2)
    idx = idx.view(*expand_shape).expand(-1, -1, *hydra.shape[2:])
    return torch.gather(hydra, dim=1, index=idx)


class HydraContextEncoder(nn.Module):
    """
    Pair-conditioned candidate encoder (FiLM).

    Inputs:
      info_pre/info_post: (B, L, 2)
      cand_pre/cand_post: (B, N, L2, 2)
    Outputs:
      pre_c/post_c: (B, N, H)
    """

    def __init__(self, d_info: int, d_cand: int, hidden: int):
        super().__init__()
        self.H = int(hidden)
        self.info_enc = nn.Sequential(
            nn.Linear(d_info * 2 * cfg.diff_pre_len, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        flat_dim = cfg.diff_infer_len * d_cand  # L2 * Dc
        self.cand_pre_proj = nn.Sequential(nn.Linear(flat_dim, hidden), nn.ReLU())
        self.cand_post_proj = nn.Sequential(nn.Linear(flat_dim, hidden), nn.ReLU())
        self.film = nn.Sequential(nn.Linear(hidden, hidden * 4))

    def forward(self, info_pre, info_post, cand_pre, cand_post):
        B, N, L2, Dc = cand_pre.shape
        H = self.H
        info_p = info_pre.reshape(B, -1)
        info_q = info_post.reshape(B, -1)
        info_f = self.info_enc(torch.cat([info_p, info_q], dim=-1))  # (B,H)

        pre_flat = cand_pre.reshape(B, N, L2 * Dc)
        post_flat = cand_post.reshape(B, N, L2 * Dc)
        pre_z = self.cand_pre_proj(pre_flat)  # (B,N,H)
        post_z = self.cand_post_proj(post_flat)

        gamma_beta = self.film(info_f)  # (B,4H)
        gamma_pre, gamma_post = gamma_beta[:, :H], gamma_beta[:, H : 2 * H]
        beta_pre, beta_post = gamma_beta[:, 2 * H : 3 * H], gamma_beta[:, 3 * H : 4 * H]
        gamma_pre = gamma_pre.unsqueeze(1)
        gamma_post = gamma_post.unsqueeze(1)
        beta_pre = beta_pre.unsqueeze(1)
        beta_post = beta_post.unsqueeze(1)
        pre_c = gamma_pre * pre_z + beta_pre
        post_c = gamma_post * post_z + beta_post
        return pre_c, post_c


class CandidateEncoder(nn.Module):
    """
    Output per-candidate representations Z_pre/Z_post for bandit selection.
    """

    def __init__(self, d_info: int, d_cand: int, rep_dim: int):
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.ctx = HydraContextEncoder(d_info=d_info, d_cand=d_cand, hidden=rep_dim)
        self.net_pre = nn.Sequential(nn.Linear(rep_dim, rep_dim), nn.ReLU(), nn.Linear(rep_dim, rep_dim), nn.ReLU())
        self.net_post = nn.Sequential(nn.Linear(rep_dim, rep_dim), nn.ReLU(), nn.Linear(rep_dim, rep_dim), nn.ReLU())

    def forward(self, info_pre, info_post, cand_pre, cand_post):
        pre_c, post_c = self.ctx(info_pre, info_post, cand_pre, cand_post)  # (B,N,H)
        return self.net_pre(pre_c), self.net_post(post_c)  # (B,N,H)


@dataclass
class SelectorCheckpoint:
    meta: dict[str, Any]
    encoder_state: dict[str, Any]
    A_pre: torch.Tensor
    b_pre: torch.Tensor
    A_post: torch.Tensor
    b_post: torch.Tensor

    def save(self, path: str):
        torch.save(
            {
                "meta": self.meta,
                "encoder_state": self.encoder_state,
                "A_pre": self.A_pre,
                "b_pre": self.b_pre,
                "A_post": self.A_post,
                "b_post": self.b_post,
            },
            path,
        )

    @staticmethod
    def load(path: str) -> "SelectorCheckpoint":
        d = torch.load(path, map_location="cpu")
        return SelectorCheckpoint(
            meta=d.get("meta", {}),
            encoder_state=d["encoder_state"],
            A_pre=d["A_pre"],
            b_pre=d["b_pre"],
            A_post=d["A_post"],
            b_post=d["b_post"],
        )


class DeepLinUCBSelector:
    """
    DeepLinUCB top-k selector for (hydra_pre, hydra_post).

    - Learns an encoder to embed each candidate tail, conditioned on (info_pre, info_post).
    - Maintains linear UCB statistics (A,b) separately for pre/post.
    - choose(): returns selected tails and caches ctx for update_batch().
    """

    def __init__(
        self,
        *,
        k_select: int,
        d_context: int = 2,
        rep_dim: int = 32,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        joint_select: bool = False,
        device: str | None = None,
        lr: float = 1e-4,
    ):
        self.k_select = int(k_select)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.rep_dim = int(rep_dim)
        self.joint_select = bool(joint_select)

        self.encoder = CandidateEncoder(d_info=d_context, d_cand=d_context, rep_dim=rep_dim).to(self.device)
        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=float(lr))

        self.A_pre = self.lambda_ * torch.eye(self.rep_dim, device=self.device)
        self.A_post = self.lambda_ * torch.eye(self.rep_dim, device=self.device)
        self.A_inv_pre = torch.inverse(self.A_pre)
        self.A_inv_post = torch.inverse(self.A_post)
        self.b_pre = torch.zeros(self.rep_dim, device=self.device)
        self.b_post = torch.zeros(self.rep_dim, device=self.device)

        self.ctx_cache = None

    def freeze(self, flag: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = not flag

    @torch.no_grad()
    def choose(self, info_pre, info_post, hydra_pre, hydra_post) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.encoder.eval()
        info_pre = info_pre.to(self.device)
        info_post = info_post.to(self.device)
        hydra_pre = hydra_pre.to(self.device)
        hydra_post = hydra_post.to(self.device)

        Z_pre, Z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)  # (B,N,H)
        B, N, H = Z_pre.shape
        k = min(self.k_select, N)
        chosen_idx_pre = []
        chosen_idx_post = []

        theta_hat_pre = self.A_inv_pre @ self.b_pre
        theta_hat_post = self.A_inv_post @ self.b_post

        for b in range(B):
            Zb_pre = Z_pre[b]  # (N,H)
            Zb_post = Z_post[b]
            # pre UCB
            mu_pre = Zb_pre @ theta_hat_pre
            Az_pre = Zb_pre @ self.A_inv_pre
            sigma_pre = torch.sqrt(torch.clamp((Az_pre * Zb_pre).sum(dim=1), min=1e-8))
            ucb_pre = mu_pre + self.alpha * sigma_pre
            # post UCB
            mu_post = Zb_post @ theta_hat_post
            Az_post = Zb_post @ self.A_inv_post
            sigma_post = torch.sqrt(torch.clamp((Az_post * Zb_post).sum(dim=1), min=1e-8))
            ucb_post = mu_post + self.alpha * sigma_post
            if self.joint_select:
                # Joint: select a SINGLE index set used for both pre/post.
                # This is critical when downstream reward treats arm j as a (pre_j, post_j) pair.
                ucb_joint = (mu_pre + mu_post) + self.alpha * (sigma_pre + sigma_post)
                idx = torch.topk(ucb_joint, k=k).indices
                idx_pre = idx
                idx_post = idx
            else:
                idx_pre = torch.topk(ucb_pre, k=k).indices
                idx_post = torch.topk(ucb_post, k=k).indices
            chosen_idx_pre.append(idx_pre)
            chosen_idx_post.append(idx_post)

        idx_pre = torch.stack(chosen_idx_pre, dim=0)
        idx_post = torch.stack(chosen_idx_post, dim=0)
        hydra_pre_sel = _gather_by_index(hydra_pre, idx_pre)
        hydra_post_sel = _gather_by_index(hydra_post, idx_post)

        self.ctx_cache = (
            info_pre.detach(),
            info_post.detach(),
            hydra_pre.detach(),
            hydra_post.detach(),
            idx_pre.detach(),
            idx_post.detach(),
        )
        return hydra_pre_sel, hydra_post_sel, idx_pre, idx_post

    def update_batch(self, r_sel: torch.Tensor):
        """
        r_sel: (B*k_select,) reward for each selected arm (repeat_interleave(k_select)).
        Call after choose().
        """
        assert self.ctx_cache is not None, "update_batch called before choose"
        info_pre, info_post, hydra_pre, hydra_post, idx_pre, idx_post = self.ctx_cache

        self.encoder.train()
        info_pre = info_pre.to(self.device)
        info_post = info_post.to(self.device)
        hydra_pre = hydra_pre.to(self.device)
        hydra_post = hydra_post.to(self.device)
        idx_pre = idx_pre.to(self.device)
        idx_post = idx_post.to(self.device)
        r_sel = r_sel.to(self.device)

        B = info_pre.size(0)
        k = idx_pre.size(1)
        assert r_sel.numel() == B * k, f"r_sel has {r_sel.numel()} elems, expected {B*k}"

        Z_pre, Z_post = self.encoder(info_pre, info_post, hydra_pre, hydra_post)  # (B,N,H)

        # flatten selected representations to (B*k, H)
        Z_sel_pre = torch.cat([Z_pre[b, idx_pre[b]] for b in range(B)], dim=0)
        Z_sel_post = torch.cat([Z_post[b, idx_post[b]] for b in range(B)], dim=0)

        with torch.no_grad():
            theta_hat_pre = self.A_inv_pre @ self.b_pre
            theta_hat_post = self.A_inv_post @ self.b_post

        pred_pre = Z_sel_pre @ theta_hat_pre
        pred_post = Z_sel_post @ theta_hat_post
        loss = F.mse_loss(pred_pre, r_sel) + F.mse_loss(pred_post, r_sel)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        with torch.no_grad():
            # recompute with updated encoder
            Z_pre2, Z_post2 = self.encoder(info_pre, info_post, hydra_pre, hydra_post)
            Z_sel_pre2 = torch.cat([Z_pre2[b, idx_pre[b]] for b in range(B)], dim=0)
            Z_sel_post2 = torch.cat([Z_post2[b, idx_post[b]] for b in range(B)], dim=0)

            for i in range(Z_sel_pre2.size(0)):
                z = Z_sel_pre2[i].view(-1, 1)
                self.A_pre += z @ z.t()
                self.b_pre += r_sel[i] * z.squeeze(1)
            for i in range(Z_sel_post2.size(0)):
                z = Z_sel_post2[i].view(-1, 1)
                self.A_post += z @ z.t()
                self.b_post += r_sel[i] * z.squeeze(1)
            self.A_inv_pre = torch.inverse(self.A_pre)
            self.A_inv_post = torch.inverse(self.A_post)

    def state(self) -> SelectorCheckpoint:
        meta = {
            "k_select": int(self.k_select),
            "rep_dim": int(self.rep_dim),
            "alpha": float(self.alpha),
            "lambda_": float(self.lambda_),
            "joint_select": bool(self.joint_select),
            "diff_pre_len": int(cfg.diff_pre_len),
            "diff_infer_len": int(cfg.diff_infer_len),
        }
        return SelectorCheckpoint(
            meta=meta,
            encoder_state=self.encoder.state_dict(),
            A_pre=self.A_pre.detach().cpu(),
            b_pre=self.b_pre.detach().cpu(),
            A_post=self.A_post.detach().cpu(),
            b_post=self.b_post.detach().cpu(),
        )

    def load_state(self, ckpt: SelectorCheckpoint):
        # Keep current behavior unless checkpoint explicitly carries a value.
        if isinstance(ckpt.meta, dict) and ("joint_select" in ckpt.meta):
            self.joint_select = bool(ckpt.meta["joint_select"])
        self.encoder.load_state_dict(ckpt.encoder_state, strict=True)
        self.A_pre = ckpt.A_pre.to(self.device)
        self.b_pre = ckpt.b_pre.to(self.device)
        self.A_post = ckpt.A_post.to(self.device)
        self.b_post = ckpt.b_post.to(self.device)
        self.A_inv_pre = torch.inverse(self.A_pre)
        self.A_inv_post = torch.inverse(self.A_post)


def bce_loss_each(logits: torch.Tensor, labels01: torch.Tensor) -> torch.Tensor:
    labels = labels01.float()
    return F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1), reduction="none")

