from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

import cfg
from cursor_mod_4.rl.linucb_selector import _gather_by_index


@dataclass
class RankSelectorCheckpoint:
    meta: dict[str, Any]
    core_state: dict[str, Any]

    def save(self, path: str):
        torch.save({"meta": self.meta, "core_state": self.core_state}, path)

    @staticmethod
    def load(path: str) -> "RankSelectorCheckpoint":
        d = torch.load(path, map_location="cpu")
        meta = d.get("meta", {})
        if "core_state" in d:
            return RankSelectorCheckpoint(meta=meta, core_state=d["core_state"])
        # Old checkpoints (encoder_state/head_state) are not compatible with RichArmScorer.
        return RankSelectorCheckpoint(meta=meta, core_state={})


class RichArmScorer(nn.Module):
    """
    Per-arm scorer with explicit head tokens + structured tail encoding + cross-attention.

    - Head: 8 pre points + 8 post points -> (B, 16, d) tokens
    Tail encoding modes:
      - pre_only: encode only hydra_pre tail (v1 behavior, but with the same head+attn stack)
      - prepost_sep_fuse: encode hydra_pre and hydra_post separately (shared weights),
        cross-attn each to head, then fuse (concat + MLP) -> score
    """

    def __init__(self, *, d_model: int = 64, n_heads: int = 4, tail_len: int = 8, tail_mode: str = "prepost_sep_fuse"):
        super().__init__()
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.tail_len = int(tail_len)
        self.tail_mode = str(tail_mode)

        self.point_in = nn.Linear(2, self.d_model)
        self.tail_pos = nn.Parameter(torch.zeros(1, self.tail_len, self.d_model))
        self.tail_conv = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.tail_self = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.cross = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads, batch_first=True, dropout=0.1)
        self.fuse = nn.Sequential(nn.LayerNorm(self.d_model * 2), nn.Linear(self.d_model * 2, self.d_model), nn.GELU())
        self.out = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 1))
        # SDPA treats (B*N) as batch dim; k×k valid can make B*N huge and CUDA returns "invalid configuration argument".
        self._max_bn = 4096

    def _encode_tail_seq(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B*N, L, 2) -> tail tokens (B*N, L, d)
        """
        t = self.point_in(x) + self.tail_pos[:, : x.size(1), :].expand(x.size(0), -1, -1)
        t = self.tail_conv(t.transpose(1, 2)).transpose(1, 2)
        return self.tail_self(t)

    def _cross_pool(self, q: torch.Tensor, hk: torch.Tensor) -> torch.Tensor:
        """
        q: (B*N, L, d), hk: (B*N, H, d) -> pooled (B*N, d)
        """
        ca, _ = self.cross(q, hk, hk, need_weights=False)
        return ca.mean(dim=1)

    def _forward_once(
        self,
        info_pre: torch.Tensor,
        info_post: torch.Tensor,
        hydra_pre: torch.Tensor,
        hydra_post: torch.Tensor,
    ) -> torch.Tensor:
        B, L0, _ = info_pre.shape
        B2, N, L2, Dc = hydra_pre.shape
        assert B == B2 and hydra_post.shape == (B, N, L2, Dc) and Dc == 2 and L2 <= self.tail_len

        pre_h = self.point_in(info_pre)  # (B,L0,d)
        post_h = self.point_in(info_post)
        head_seq = torch.cat([pre_h, post_h], dim=1)  # (B, 2*L0, d)

        hk = head_seq.unsqueeze(1).expand(B, N, head_seq.size(1), self.d_model).reshape(B * N, head_seq.size(1), self.d_model)
        if self.tail_mode == "pre_only":
            xpre = hydra_pre.reshape(B * N, L2, 2)
            qpre = self._encode_tail_seq(xpre)
            pooled = self._cross_pool(qpre, hk)
            scores = self.out(pooled).squeeze(-1).view(B, N)
            return scores
        if self.tail_mode != "prepost_sep_fuse":
            raise ValueError(f"Unknown tail_mode={self.tail_mode!r}")

        xpre = hydra_pre.reshape(B * N, L2, 2)
        xpost = hydra_post.reshape(B * N, L2, 2)
        qpre = self._encode_tail_seq(xpre)
        qpost = self._encode_tail_seq(xpost)
        pooled_pre = self._cross_pool(qpre, hk)
        pooled_post = self._cross_pool(qpost, hk)
        fused = self.fuse(torch.cat([pooled_pre, pooled_post], dim=1))
        scores = self.out(fused).squeeze(-1).view(B, N)
        return scores

    def forward(self, info_pre: torch.Tensor, info_post: torch.Tensor, hydra_pre: torch.Tensor, hydra_post: torch.Tensor) -> torch.Tensor:
        """
        info_*: (B, L0, 2), hydra_*: (B, N, L2, 2) -> scores (B, N)
        """
        B, _, _ = info_pre.shape
        N = int(hydra_pre.size(1))
        bn = B * N
        if bn <= self._max_bn:
            return self._forward_once(info_pre, info_post, hydra_pre, hydra_post)

        b_step = max(1, self._max_bn // max(N, 1))
        parts: list[torch.Tensor] = []
        for s in range(0, B, b_step):
            e = min(s + b_step, B)
            parts.append(
                self._forward_once(
                    info_pre[s:e],
                    info_post[s:e],
                    hydra_pre[s:e],
                    hydra_post[s:e],
                )
            )
        return torch.cat(parts, dim=0)


class JointTailRankSelector(nn.Module):
    """
    Offline / supervised tail ranker using RichArmScorer.

    Training losses live in `train_offline_teacher_rank_selector.py` (ListNet + optional mini-retrieval).
    """

    def __init__(self, *, rep_dim: int = 64, lr: float = 1e-4, device: str | None = None, tail_mode: str = "prepost_sep_fuse"):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rep_dim = int(rep_dim)
        self.tail_mode = str(tail_mode)
        self.core = RichArmScorer(
            d_model=self.rep_dim,
            n_heads=4,
            tail_len=int(cfg.diff_infer_len),
            tail_mode=self.tail_mode,
        ).to(self.device)
        self.opt = torch.optim.Adam(self.core.parameters(), lr=float(lr))

    def forward(self, info_pre: torch.Tensor, info_post: torch.Tensor, hydra_pre: torch.Tensor, hydra_post: torch.Tensor) -> torch.Tensor:
        return self.core(info_pre.to(self.device), info_post.to(self.device), hydra_pre.to(self.device), hydra_post.to(self.device))

    @torch.no_grad()
    def choose_topk(
        self,
        info_pre: torch.Tensor,
        info_post: torch.Tensor,
        hydra_pre: torch.Tensor,
        hydra_post: torch.Tensor,
        *,
        k_select: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.eval()
        s = self.forward(info_pre, info_post, hydra_pre, hydra_post)
        k = min(int(k_select), int(s.size(1)))
        _, idx = torch.topk(s, k=k, dim=1)
        hydra_pre_sel = _gather_by_index(hydra_pre.to(self.device), idx)
        hydra_post_sel = _gather_by_index(hydra_post.to(self.device), idx)
        return hydra_pre_sel, hydra_post_sel, idx

    def state(self) -> RankSelectorCheckpoint:
        meta = {"rep_dim": int(self.rep_dim), "arch": "rich_arm_v3_tail_mode", "tail_mode": str(self.tail_mode)}
        return RankSelectorCheckpoint(
            meta=meta,
            core_state={k: v.detach().cpu() for k, v in self.core.state_dict().items()},
        )

    def load_state(self, ckpt: RankSelectorCheckpoint):
        if not ckpt.core_state:
            print("[JointTailRankSelector] checkpoint has no core_state (old format?); skipping load.", flush=True)
            return
        dev = self.device
        cur = self.core.state_dict()

        # Compatibility: load what we can across tail_mode variants.
        adapted: dict[str, torch.Tensor] = {}
        for k, v in ckpt.core_state.items():
            if k not in cur:
                continue
            vv = v.to(dev, non_blocking=True)
            if cur[k].shape == vv.shape:
                adapted[k] = vv
                continue

        missing, unexpected = self.core.load_state_dict(adapted, strict=False)
        if missing or unexpected:
            print(
                f"[JointTailRankSelector] loaded with strict=False; missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
