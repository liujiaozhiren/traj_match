"""Parameter / structure element counts for all 14 methods."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .baseline_pipelines import FULL_BASELINE, build_artifacts, build_model, count_aux_params
from .constants import METHOD_NAMES
from .models import build_model as build_mod4_model
from .traj_data import make_synthetic_pairs


def _neutraj_sam_elements(model: nn.Module) -> int:
    n = 0
    for name, buf in model.named_buffers():
        if name.endswith("spatial_embedding.memory"):
            n += int(buf.numel())
    return n


def _model_buffer_elements(method: str, model: nn.Module) -> int:
    n = 0
    for name, buf in model.named_buffers():
        if method == "neutraj" and name.endswith("spatial_embedding.memory"):
            continue
        n += int(buf.numel())
    return n


def count_method_params(method: str) -> dict[str, Any]:
    method = method.lower()
    if method in ("dtw", "hausdorff", "frechet", "sspd"):
        return {
            "method": method,
            "p_learnable": 0,
            "p_aux": 0,
            "p_total": 0,
        }

    if method in FULL_BASELINE:
        pairs = make_synthetic_pairs(32, seed=0)
        device = torch.device("cpu")
        artifacts = build_artifacts(method, pairs, device)
        model = build_model(method, artifacts, device)
        p_param = sum(int(p.numel()) for p in model.parameters())
        p_buf = _model_buffer_elements(method, model)
        p_art = int(count_aux_params(method, artifacts, model))
        if method == "trajcl":
            cell = int(model.cell_emb.weight.numel())  # noqa: attr-defined
            p_learnable = p_param - cell
            p_aux = cell + p_buf
        elif method == "neutraj":
            p_learnable = p_param + _neutraj_sam_elements(model)
            p_aux = p_art + p_buf
        else:
            p_learnable = p_param
            p_aux = p_art + p_buf
        return {
            "method": method,
            "p_learnable": int(p_learnable),
            "p_aux": int(p_aux),
            "p_total": int(p_learnable + p_aux),
        }

    model = build_mod4_model(method)
    p_learnable = sum(int(p.numel()) for p in model.parameters())
    p_aux = _model_buffer_elements(method, model)
    return {
        "method": method,
        "p_learnable": int(p_learnable),
        "p_aux": int(p_aux),
        "p_total": int(p_learnable + p_aux),
    }


def count_all(only: list[str] | None = None) -> list[dict[str, Any]]:
    names = list(only) if only else list(METHOD_NAMES)
    return [count_method_params(m) for m in names]
