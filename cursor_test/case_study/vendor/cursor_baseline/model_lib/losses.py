"""Vendored from match_model.my.my_infer_demo (pair_loss only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pair_loss(logits, labels):
    if labels.dtype != torch.float32 and labels.dtype != torch.float16 and labels.dtype != torch.bfloat16:
        labels = labels.float()

    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

    if loss.dim() == 1:
        return loss
    return loss.mean(dim=1)
