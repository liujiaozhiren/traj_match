"""Completion heads: full A (B,T,2) -> full B_hat (B,T,2), same length."""

from __future__ import annotations

from torch import nn

from cursor_baseline.attnmove_model import AttnMoveCompletion
from cursor_baseline.attnmove_region import RegionVocab


class SeqGRUCompletion(nn.Module):
    """GRU over A, per-step linear -> B_hat."""

    def __init__(self, hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(2, hidden, num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(hidden, 2)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(a)
        return self.lin(h)


def build_completion_backbone(
    name: str,
    *,
    rnn_hidden: int,
    rnn_layers: int,
    region_vocab: RegionVocab | None = None,
    attn_hidden: int = 64,
    attn_layers: int = 2,
    attn_heads: int = 1,
    attn_drop: float = 0.3,
    attn_fb_drop: float = 0.3,
    label_confidence: float = 0.9,
    reg_lambda: float = 0.001,
    reg_dist_lambda: float = 1.0,
) -> nn.Module:
    n = str(name).lower().strip()
    if n in ("gru", "seqgru", "seq_gru"):
        return SeqGRUCompletion(hidden=int(rnn_hidden), num_layers=int(rnn_layers))
    if n == "attnmove":
        if region_vocab is None:
            raise ValueError("build_completion_backbone(attnmove): region_vocab required")
        return AttnMoveCompletion(
            region_vocab,
            hidden=int(attn_hidden),
            n_layers=int(attn_layers),
            n_heads=int(attn_heads),
            drop=float(attn_drop),
            fb_drop=float(attn_fb_drop),
            label_confidence=float(label_confidence),
            reg_lambda=float(reg_lambda),
            reg_dist_lambda=float(reg_dist_lambda),
        )
    if n in ("rntrajrec", "rn_trajrec", "rntraj"):
        raise NotImplementedError(
            "RNTrajRec backbone: needs road net + preprocess from baseline/RNTrajRec (TODO)."
        )
    raise ValueError(f"unknown --backbone {name!r}; try gru | attnmove")
