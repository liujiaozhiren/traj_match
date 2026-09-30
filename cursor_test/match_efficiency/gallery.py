"""Offline gallery index builders."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from .constants import EMBED_DIM, HYDRA_TAIL, N_POINTS
from .baseline_pipelines import FULL_BASELINE, build_artifacts, build_model as build_baseline_model, encode_posts_batched
from .probe import count_storage_elements
from .traj_data import pair_pre_post_tensors, pair_to_info_xy8


@dataclass
class GalleryIndex:
    method: str
    n: int
    build_sec: float
    storage_elements: int
    data: dict[str, Any]


def _device_of(m: nn.Module | None, fallback: torch.device) -> torch.device:
    if m is None:
        return fallback
    try:
        return next(m.parameters()).device
    except StopIteration:
        return fallback


@torch.no_grad()
def build_gallery_index(
    method: str,
    pairs: list[tuple],
    gallery_indices: list[int],
    *,
    device: torch.device,
    hydra_tail: int = HYDRA_TAIL,
    gallery_online: bool = False,
) -> GalleryIndex:
    t0 = time.perf_counter()
    method = method.lower()
    n = len(gallery_indices)
    data: dict[str, Any] = {"gallery_indices": list(gallery_indices)}

    if method in ("dtw", "hausdorff", "frechet", "sspd"):
        cpu = torch.device("cpu")
        trajs = []
        for gi in gallery_indices:
            _, post = pair_pre_post_tensors(pairs[gi], device=cpu)
            trajs.append(post)
        data["post_trajs"] = torch.stack(trajs, dim=0)

    elif method in FULL_BASELINE:
        artifacts = build_artifacts(method, pairs, device)
        model = build_baseline_model(method, artifacts, device).eval()
        data["artifacts"] = artifacts
        data["model"] = model
        if gallery_online:
            pass  # gallery materialized in T_gallery
        else:
            if method == "attnmove":
                from .baseline_pipelines import _stack_lonlat_posts

                data["post_trajs"] = _stack_lonlat_posts(pairs, gallery_indices, device)
            else:
                emb = encode_posts_batched(
                    method, model, artifacts, pairs, gallery_indices, device=device
                )
                if method in ("dan", "tat", "tat_graph"):
                    data["post_vecs"] = emb
                else:
                    data["embeddings"] = emb

    elif method in ("mod4", "mod4_rl"):
        from .models import build_model as build_mod4_model

        pipe = build_mod4_model(method).to(device).eval()
        data["model"] = pipe
        data["hydra_tail"] = hydra_tail
        if gallery_online:
            info_posts = _stack_mod4_info_posts(pairs, gallery_indices)
            data["info_post"] = info_posts
        else:
            info_posts, hydra_posts = _build_mod4_gallery_batched(
                pipe,
                pairs,
                gallery_indices,
                device=device,
                hydra_tail=hydra_tail,
            )
            data["info_post"] = info_posts
            data["hydra_post"] = hydra_posts

    else:
        raise ValueError(f"unknown method for gallery: {method}")

    storage = 0
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            storage += count_storage_elements(v)

    return GalleryIndex(
        method=method,
        n=n,
        build_sec=time.perf_counter() - t0,
        storage_elements=storage,
        data=data,
    )


@torch.no_grad()
def _stack_mod4_info_posts(pairs: list[tuple], gallery_indices: list[int]) -> torch.Tensor:
    n = len(gallery_indices)
    info_posts = torch.empty((n, 8, 2), dtype=torch.float32)
    for j, gi in enumerate(gallery_indices):
        _, info_post = pair_to_info_xy8(pairs[gi], device=torch.device("cpu"))
        info_posts[j] = info_post
    return info_posts


@torch.no_grad()
def mod4_online_gallery_scores(
    pipe,
    info_q: torch.Tensor,
    hydra_q: torch.Tensor,
    info_posts: torch.Tensor,
    *,
    device: torch.device,
    hydra_tail: int,
    gallery_batch: int,
) -> torch.Tensor:
    """Online: gallery post diffusion + batched match."""
    n = info_posts.shape[0]
    chunk = min(int(gallery_batch), n)
    parts: list[torch.Tensor] = []
    for j0 in range(0, n, chunk):
        j1 = min(j0 + chunk, n)
        heads = info_posts[j0:j1].permute(0, 2, 1).contiguous().to(device)
        hydra_g = pipe.infer_gallery_hydra_batch(heads, hydra_tail=hydra_tail)
        info_g = info_posts[j0:j1].to(device)
        parts.append(pipe.match_gallery_batch(info_q, hydra_q.to(device), info_g, hydra_g))
    return torch.cat(parts, dim=0)


@torch.no_grad()
def _build_mod4_gallery_batched(
    pipe,
    pairs: list[tuple],
    gallery_indices: list[int],
    *,
    device: torch.device,
    hydra_tail: int,
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Offline gallery hydra (post diffusion). Runs once before online serving.
    Batched parallel — not per-item Python loop.
    """
    n = len(gallery_indices)
    info_posts = torch.empty((n, 8, 2), dtype=torch.float32)
    hydra_posts = torch.empty((n, hydra_tail, 4, 2), dtype=torch.float32)
    for i0 in range(0, n, batch_size):
        i1 = min(i0 + batch_size, n)
        heads = []
        for j, gi in enumerate(gallery_indices[i0:i1]):
            _, info_post = pair_to_info_xy8(pairs[gi], device=torch.device("cpu"))
            info_posts[i0 + j] = info_post
            heads.append(info_post.T.contiguous())
        batch_heads = torch.stack(heads, dim=0).to(device)
        hyd = pipe.infer_gallery_hydra_batch(batch_heads, hydra_tail=hydra_tail).cpu()
        hydra_posts[i0:i1] = hyd
    return info_posts, hydra_posts


def slice_gallery_index(gal: GalleryIndex, n: int) -> GalleryIndex:
    """Use first n gallery rows (online path unchanged; no re-diffusion)."""
    n = min(int(n), gal.n)
    data = dict(gal.data)
    if "gallery_indices" in data:
        data["gallery_indices"] = list(data["gallery_indices"][:n])
    for key in ("post_trajs", "embeddings", "post_vecs", "info_post", "hydra_post"):
        if key in data and isinstance(data[key], torch.Tensor):
            data[key] = data[key][:n]
    storage = sum(int(data[k].numel()) for k in data if isinstance(data.get(k), torch.Tensor))
    return GalleryIndex(method=gal.method, n=n, build_sec=0.0, storage_elements=storage, data=data)
