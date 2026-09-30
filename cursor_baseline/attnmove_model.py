"""AttnMove-style completion: A regions -> cross-attn queries -> B region logits -> lon/lat."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cursor_baseline.attnmove_region import PAD_ID, RegionVocab


def sinusoidal_position_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """(length, dim) PE."""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


def smoothing_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    vocab_size: int,
    confidence: float = 0.9,
    ignore_index: int = PAD_ID,
) -> torch.Tensor:
    """Label-smoothing CE; logits (B,T,V), labels (B,T)."""
    b, t, v = logits.shape
    flat_logits = logits.reshape(b * t, v)
    flat_labels = labels.reshape(b * t)
    valid = flat_labels != ignore_index
    if not valid.any():
        return flat_logits.sum() * 0.0
    low = (1.0 - confidence) / float(max(vocab_size - 2, 1))
    one_hot = torch.zeros_like(flat_logits).scatter_(1, flat_labels.unsqueeze(1), 1.0)
    soft = one_hot * confidence + (1.0 - one_hot) * low
    soft[flat_labels == PAD_ID] = 0.0
    log_probs = F.log_softmax(flat_logits, dim=-1)
    loss = -(soft * log_probs).sum(dim=-1)
    return loss[valid].mean()


def region_distance_loss(
    pred_ids: torch.Tensor,
    true_ids: torch.Tensor,
    dist_matrix: torch.Tensor,
) -> torch.Tensor:
    """Mean region_distance[pred, true] on CPU table."""
    dm = dist_matrix.to(device=pred_ids.device)
    p = pred_ids.clamp(0, dm.shape[0] - 1)
    t = true_ids.clamp(0, dm.shape[0] - 1)
    return dm[p, t].mean()


class _FFN(nn.Module):
    def __init__(self, dim: int, fb_drop: float):
        super().__init__()
        hid = dim * 4
        self.lin1 = nn.Linear(dim, hid)
        self.lin2 = nn.Linear(hid, dim)
        self.drop = nn.Dropout(fb_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        return x


class _AttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, drop: float, fb_drop: float):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.ffn = _FFN(dim, fb_drop)
        self.attn_drop = nn.Dropout(drop)
        self.res_drop = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def self_attn(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(self.norm1(x)))
        k = self._split(self.k(self.norm1(x)))
        v = self._split(self.v(self.norm1(x)))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        w = F.softmax(scores, dim=-1)
        w = self.attn_drop(w)
        out = torch.matmul(w, v)
        return self.o(self._merge(out))

    def cross_attn(self, q_in: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = self._split(self.q(self.norm2(q_in)))
        k = self._split(self.k(self.norm2(mem)))
        v = self._split(self.v(self.norm2(mem)))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        w = F.softmax(scores, dim=-1)
        w = self.attn_drop(w)
        out = torch.matmul(w, v)
        return self.o(self._merge(out))

    def forward_encoder(self, mem: torch.Tensor) -> torch.Tensor:
        mem = mem + self.res_drop(self.self_attn(mem))
        mem = mem + self.res_drop(self.ffn(self.norm3(mem)))
        return mem

    def forward_cross(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        q = q + self.res_drop(self.cross_attn(q, mem))
        q = q + self.res_drop(self.ffn(self.norm3(q)))
        return q


class AttnMoveCompletion(nn.Module):
    """
    A-encoder (self-attn) + T aligned queries (init from A embed+pos) cross-attn to A.
    Inference: argmax region -> centroid lon/lat.
    """

    def __init__(
        self,
        vocab: RegionVocab,
        *,
        hidden: int = 64,
        n_layers: int = 2,
        n_heads: int = 1,
        drop: float = 0.3,
        fb_drop: float = 0.3,
        label_confidence: float = 0.9,
        reg_lambda: float = 0.001,
        reg_dist_lambda: float = 1.0,
    ):
        super().__init__()
        self.vocab = vocab
        self.hidden = int(hidden)
        self.label_confidence = float(label_confidence)
        self.reg_lambda = float(reg_lambda)
        self.reg_dist_lambda = float(reg_dist_lambda)
        v = vocab.vocab_size
        self.embed = nn.Embedding(v, self.hidden)
        self._init_embed_from_centroids()
        self.blocks = nn.ModuleList(
            [_AttnBlock(self.hidden, n_heads, drop, fb_drop) for _ in range(int(n_layers))]
        )
        self.out = nn.Linear(self.hidden, v)
        self.embed_drop = nn.Dropout(drop)

    def _init_embed_from_centroids(self) -> None:
        """LocEmbedder-style: centroid lon/lat -> linear init."""
        cents = self.vocab.id2centroid.clone()
        c0 = cents[:, 0] - cents[:, 0].min()
        c1 = cents[:, 1] - cents[:, 1].min()
        coor = torch.stack([c0, c1], dim=-1)
        with torch.no_grad():
            proj = torch.randn(2, self.hidden) * 0.1
            bias = torch.randn(self.hidden) * 0.1
            self.embed.weight.copy_(coor @ proj + bias.unsqueeze(0))

    def _embed_seq(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids) * math.sqrt(self.hidden)
        t = ids.shape[1]
        pe = sinusoidal_position_encoding(t, self.hidden, ids.device)
        return self.embed_drop(x + pe.unsqueeze(0))

    def forward_logits(self, a_xy: torch.Tensor) -> torch.Tensor:
        """a_xy (B,T,2) -> logits (B,T,V)."""
        a_ids = self.vocab.lonlat_to_ids(a_xy)
        mem = self._embed_seq(a_ids)
        q = mem
        for blk in self.blocks:
            mem = blk.forward_encoder(mem)
            q = blk.forward_cross(q, mem)
        return self.out(q)

    def forward(self, a_xy: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(a_xy)
        pred_ids = logits.argmax(dim=-1)
        return self.vocab.ids_to_lonlat(pred_ids)

    def compute_loss(self, a_xy: torch.Tensor, b_xy: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        logits = self.forward_logits(a_xy)
        b_ids = self.vocab.lonlat_to_ids(b_xy)
        ce = smoothing_cross_entropy(
            logits,
            b_ids,
            vocab_size=self.vocab.vocab_size,
            confidence=self.label_confidence,
        )
        pred_ids = logits.argmax(dim=-1)
        rdist = region_distance_loss(pred_ids, b_ids, self.vocab.dist_matrix)
        l2 = torch.stack([p.pow(2).mean() for p in self.parameters()]).mean()
        loss = ce + self.reg_dist_lambda * rdist + self.reg_lambda * l2
        return loss, {
            "ce": float(ce.item()),
            "region_dist": float(rdist.item()),
            "l2": float(l2.item()),
        }
