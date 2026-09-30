from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg


SchemeId = Literal[3, 4]


class MatchInputs(NamedTuple):
    graph: object | None
    info_pre: torch.Tensor | None
    info_post: torch.Tensor | None
    hydra_pre: torch.Tensor | None
    hydra_post: torch.Tensor | None


def build_match_inputs(
    *,
    scheme: int,
    trajA_head: torch.Tensor,
    trajB_head: torch.Tensor,
    trajA_tail: torch.Tensor,
    trajB_tail: torch.Tensor,
    pair: bool = True,
    valid: bool = False,
):
    sid = int(scheme)
    if sid not in (3, 4):
        raise ValueError(f"hyper scripts only support scheme 3/4, got {scheme}")

    if trajA_head.shape[-1] != 2:
        trajA_head = torch.transpose(trajA_head, 1, 2)
        trajB_head = torch.transpose(trajB_head, 1, 2)
        trajA_tail = torch.transpose(trajA_tail, 2, 3)
        trajB_tail = torch.transpose(trajB_tail, 2, 3)

    if valid:
        B1 = trajA_head.size(0)
        B2 = trajB_head.size(0)
        info_pre = trajA_head.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
        info_post = trajB_head.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
        hydra_pre = trajA_tail.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1 * B2, trajA_tail.size(1), trajA_tail.size(2), 2)
        hydra_post = trajB_tail.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1 * B2, trajB_tail.size(1), trajB_tail.size(2), 2)
        return MatchInputs(graph=None, info_pre=info_pre, info_post=info_post, hydra_pre=hydra_pre, hydra_post=hydra_post)

    B = trajB_head.shape[0]
    if B == 1:
        raise ValueError("batch size=1 无法构造负样本（derangement）。")
    idx = torch.arange(B, device=cfg.device)
    shift = torch.randint(1, B, (1,), device=cfg.device).item()
    perm = (idx + shift) % B

    info_pre_pos, info_post_pos = trajA_head, trajB_head
    hydra_pre_pos, hydra_post_pos = trajA_tail, trajB_tail
    info_pre_neg, info_post_neg = trajA_head, trajB_head[perm]
    hydra_pre_neg, hydra_post_neg = trajA_tail, trajB_tail[perm]

    info_pre = torch.cat([info_pre_pos, info_pre_neg], dim=0)
    info_post = torch.cat([info_post_pos, info_post_neg], dim=0)
    hydra_pre = torch.cat([hydra_pre_pos, hydra_pre_neg], dim=0)
    hydra_post = torch.cat([hydra_post_pos, hydra_post_neg], dim=0)
    y = torch.cat([torch.ones(B, device=cfg.device), torch.zeros(B, device=cfg.device)], dim=0).long()
    return MatchInputs(graph=None, info_pre=info_pre, info_post=info_post, hydra_pre=hydra_pre, hydra_post=hydra_post), y


class Scheme3SetTransformer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.dim = dim
        self.point = nn.Linear(2, dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.sab = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out = nn.Sequential(nn.Linear(dim * 4, 256), nn.ReLU(), nn.Linear(256, 1))

    def _tail_tokens(self, hydra: torch.Tensor) -> torch.Tensor:
        x = self.point(hydra)
        return x.mean(dim=2)

    def _cross_attn(self, Q: torch.Tensor, K: torch.Tensor):
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(K)
        attn = torch.softmax((q @ k.transpose(1, 2)) / (self.dim**0.5), dim=-1)
        aligned = attn @ v
        return aligned, attn

    def forward(self, info_pre, info_post, hydra_pre, hydra_post):
        TA = self.sab(self._tail_tokens(hydra_pre))
        TB = self.sab(self._tail_tokens(hydra_post))
        A_aligned, _ = self._cross_attn(TA, TB)
        B_aligned, _ = self._cross_attn(TB, TA)
        u = A_aligned.mean(dim=1)
        v = B_aligned.mean(dim=1)
        feat = torch.cat([u, v, (u - v).abs(), u * v], dim=-1)
        return self.out(feat).squeeze(-1)


class Scheme4PairTransformer(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.dim = dim
        self.point = nn.Linear(2, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def _flatten_tokens(self, head: torch.Tensor, hydra: torch.Tensor) -> torch.Tensor:
        B = head.size(0)
        tail = hydra.reshape(B, -1, 2)
        return torch.cat([head, tail], dim=1)

    def forward(self, info_pre, info_post, hydra_pre, hydra_post):
        A = self._flatten_tokens(info_pre, hydra_pre)
        B = self._flatten_tokens(info_post, hydra_post)
        x = torch.cat([A, B], dim=1)
        x = self.point(x)
        cls = self.cls.repeat(x.size(0), 1, 1)
        x = torch.cat([cls, x], dim=1)
        h = self.enc(x)[:, 0, :]
        return self.out(h).squeeze(-1)


class MatchModel(nn.Module):
    def __init__(self, *, dim: int, scheme: int = 4):
        super().__init__()
        self.scheme = int(scheme)
        self.dim = int(dim)
        if self.scheme == 3:
            self.core = Scheme3SetTransformer(dim=self.dim)
        elif self.scheme == 4:
            self.core = Scheme4PairTransformer(dim=self.dim)
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    def match(self, x: Union[MatchInputs, object], pairs: bool = True) -> torch.Tensor:
        if not isinstance(x, MatchInputs) or x.info_pre is None:
            raise TypeError("scheme3/4 expects MatchInputs with tensors")
        return self.core(x.info_pre, x.info_post, x.hydra_pre, x.hydra_post).view(-1)

    def freeze(self, flag=True):
        for p in self.parameters():
            p.requires_grad = not flag


def pair_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        labels = labels.float()
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if loss.dim() == 1:
        return loss
    return loss.mean(dim=1)
