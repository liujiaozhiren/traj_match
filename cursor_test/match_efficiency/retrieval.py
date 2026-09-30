"""E2E retrieval: 1 query vs N gallery (batched parallel)."""

from __future__ import annotations

import torch

from .baseline_pipelines import FULL_BASELINE, encode_query, gallery_encode_and_score, score_gallery
from .constants import HYDRA_TAIL
from .distances import query_gallery_scores
from .gallery import GalleryIndex, mod4_online_gallery_scores
from .models import Mod4Pipeline
from .probe import E2ETiming, MemoryProbe, timed_section
from .traj_data import pair_pre_post_tensors, pair_to_info_xy8


def _argmax_top1(scores: torch.Tensor) -> int:
    return int(scores.argmax().item())


@torch.no_grad()
def run_retrieval_e2e(
    method: str,
    query_pair: tuple,
    gallery: GalleryIndex,
    *,
    pairs: list[tuple],
    device: torch.device,
    gallery_batch: int,
    hydra_tail: int = HYDRA_TAIL,
    query_noise_base: int = 0,
    gallery_online: bool = False,
) -> tuple[int, E2ETiming]:
    method = method.lower()
    mem = MemoryProbe()
    timing = E2ETiming()
    top_idx = 0
    indices = gallery.data["gallery_indices"]

    # ---- T_query ----
    with timed_section() as tq:
        if method in ("dtw", "hausdorff", "frechet", "sspd"):
            pre, _ = pair_pre_post_tensors(query_pair, device=torch.device("cpu"))
        elif method in FULL_BASELINE:
            qstate = encode_query(
                method,
                gallery.data["model"],
                gallery.data["artifacts"],
                query_pair,
                device=device,
                noise_base=query_noise_base,
            )
            gallery.data["_query"] = qstate
        elif method in ("mod4", "mod4_rl"):
            pipe: Mod4Pipeline = gallery.data["model"]
            info_pre, _ = pair_to_info_xy8(query_pair, device=device)
            head_pre = info_pre.T.contiguous()
            gallery.data["_info_q"] = info_pre
            gallery.data["_hydra_q"] = pipe.infer_query_hydra(head_pre, hydra_tail=hydra_tail)
        else:
            raise ValueError(f"unknown method: {method}")
    timing.t_query = tq[0]

    # ---- T_gallery (batched parallel) ----
    with timed_section() as tg:
        if method in ("dtw", "hausdorff", "frechet", "sspd"):
            scores = query_gallery_scores(pre, gallery.data["post_trajs"], method, gallery_batch=gallery_batch)
        elif method in FULL_BASELINE:
            if gallery_online:
                scores = gallery_encode_and_score(
                    method,
                    gallery.data["model"],
                    gallery.data["artifacts"],
                    pairs,
                    indices,
                    gallery.data["_query"],
                    device=device,
                    gallery_batch=gallery_batch,
                )
            else:
                scores = score_gallery(
                    method,
                    gallery.data["model"],
                    gallery.data["_query"],
                    gallery.data,
                    device=device,
                    gallery_batch=gallery_batch,
                )
        elif method in ("mod4", "mod4_rl"):
            pipe = gallery.data["model"]
            info_q = gallery.data["_info_q"]
            hydra_q = gallery.data["_hydra_q"]
            if gallery_online:
                scores = mod4_online_gallery_scores(
                    pipe,
                    info_q,
                    hydra_q,
                    gallery.data["info_post"],
                    device=device,
                    hydra_tail=hydra_tail,
                    gallery_batch=gallery_batch,
                )
            else:
                hydra_q = hydra_q.to(device)
                info_g_all = gallery.data["info_post"].to(device)
                hydra_g_all = gallery.data["hydra_post"].to(device)
                n = info_g_all.shape[0]
                chunk = min(int(gallery_batch), n)
                parts = []
                for j0 in range(0, n, chunk):
                    j1 = min(j0 + chunk, n)
                    parts.append(
                        pipe.match_gallery_batch(
                            info_q,
                            hydra_q,
                            info_g_all[j0:j1],
                            hydra_g_all[j0:j1],
                        )
                    )
                scores = torch.cat(parts, dim=0)
        top_idx = _argmax_top1(scores)
    timing.t_gallery = tg[0]

    with timed_section() as ta:
        _ = top_idx
    timing.t_argmax = ta[0]
    timing.t_e2e = timing.t_query + timing.t_gallery + timing.t_argmax
    mem.sample()
    timing.peak_rss_mib, timing.peak_vram_mib = mem.finish()
    return top_idx, timing


@torch.no_grad()
def run_retrieval_median(
    method: str,
    query_pairs: list[tuple],
    gallery: GalleryIndex,
    *,
    pairs: list[tuple],
    device: torch.device,
    gallery_batch: int,
    hydra_tail: int = HYDRA_TAIL,
    gallery_online: bool = False,
) -> E2ETiming:
    runs: list[E2ETiming] = []
    for i, qp in enumerate(query_pairs):
        _, t = run_retrieval_e2e(
            method,
            qp,
            gallery,
            pairs=pairs,
            device=device,
            gallery_batch=gallery_batch,
            hydra_tail=hydra_tail,
            query_noise_base=i * 17,
            gallery_online=gallery_online,
        )
        runs.append(t)

    def med(attr: str) -> float:
        xs = sorted(getattr(r, attr) for r in runs)
        return xs[len(xs) // 2]

    return E2ETiming(
        t_query=med("t_query"),
        t_gallery=med("t_gallery"),
        t_argmax=med("t_argmax"),
        t_e2e=med("t_e2e"),
        peak_rss_mib=med("peak_rss_mib"),
        peak_vram_mib=med("peak_vram_mib"),
    )
