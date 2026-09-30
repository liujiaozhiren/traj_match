#!/usr/bin/env python3
"""Segmented timing breakdown for cursor_mod_4 (mod4) only."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from match_efficiency.constants import HYDRA_TAIL, K_SELECT_RL, K_SELECT_VANILLA, DIFFUSION_STEPS  # noqa: E402
from match_efficiency.gallery import build_gallery_index, slice_gallery_index  # noqa: E402
from match_efficiency.models import Mod4Pipeline, build_model  # noqa: E402
from match_efficiency.probe import timed_section  # noqa: E402
from match_efficiency.traj_data import make_synthetic_pairs, pair_to_info_xy8  # noqa: E402


@dataclass
class Mod4Profile:
    n_gallery: int
    gallery_batch: int
    hydra_tail: int
    k_select: int
    diffusion_steps: int
    # offline (index build, once)
    offline_info_extract_sec: float
    offline_gallery_diffusion_sec: float
    offline_index_total_sec: float
    # online query
    online_preprocess_sec: float
    online_query_diffusion_sec: float
    online_query_unet_calls: int
    # online gallery
    online_gallery_h2d_sec: float
    online_tail_select_sec: float
    online_match_forward_sec: float
    online_argmax_sec: float
    online_gallery_total_sec: float
    online_e2e_sec: float
    peak_vram_mib: float


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def profile_mod4_online(
    pipe: Mod4Pipeline,
    query_pair: tuple,
    info_post: torch.Tensor,
    hydra_post: torch.Tensor,
    *,
    device: torch.device,
    gallery_batch: int,
    hydra_tail: int,
) -> dict:
    n = info_post.shape[0]
    chunk = min(gallery_batch, n)

    with timed_section() as t_pre:
        info_q, _ = pair_to_info_xy8(query_pair, device=device)
    preprocess_sec = t_pre[0]

    t_diff, unet_calls, hydra_q = _profile_query_diffusion(pipe, info_q, hydra_tail, device)

    with timed_section() as t_h2d:
        ig = info_post.to(device, non_blocking=True)
        hg = hydra_post.to(device, non_blocking=True)
    h2d_sec = t_h2d[0]

    match_sec = 0.0
    tail_sec = 0.0
    parts = []
    for j0 in range(0, n, chunk):
        j1 = min(j0 + chunk, n)
        info_g = ig[j0:j1]
        hydra_g = hg[j0:j1]
        b = info_g.shape[0]
        iq = info_q.unsqueeze(0).expand(b, -1, -1)
        hq = hydra_q.unsqueeze(0).expand(b, *hydra_q.shape)

        with timed_section() as t_tail:
            ks = min(K_SELECT_VANILLA, hq.shape[1])
            hq_use, hg_use = hq[:, :ks], hydra_g[:, :ks]
        tail_sec += t_tail[0]

        with timed_section() as t_match:
            _sync()
            logits = pipe.match(iq, info_g, hq_use, hg_use)
            _sync()
        match_sec += t_match[0]
        parts.append(logits)

    with timed_section() as t_arg:
        scores = torch.cat(parts, dim=0)
        _ = int(scores.argmax().item())
    argmax_sec = t_arg[0]

    if torch.cuda.is_available():
        _sync()
        vram = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        vram = 0.0

    gallery_total = h2d_sec + tail_sec + match_sec
    e2e = preprocess_sec + t_diff + gallery_total + argmax_sec
    return {
        "online_preprocess_sec": preprocess_sec,
        "online_query_diffusion_sec": t_diff,
        "online_query_unet_calls": unet_calls,
        "online_gallery_h2d_sec": h2d_sec,
        "online_tail_select_sec": tail_sec,
        "online_match_forward_sec": match_sec,
        "online_argmax_sec": argmax_sec,
        "online_gallery_total_sec": gallery_total,
        "online_e2e_sec": e2e,
        "peak_vram_mib": vram,
    }


def _profile_query_diffusion(
    pipe: Mod4Pipeline, info_q: torch.Tensor, hydra_tail: int, device: torch.device
) -> tuple[float, int, torch.Tensor]:
    unet_calls = 0
    orig_forward = pipe.pre_diff.unet.forward

    def counted_forward(*args, **kwargs):
        nonlocal unet_calls
        unet_calls += 1
        return orig_forward(*args, **kwargs)

    pipe.pre_diff.unet.forward = counted_forward  # type: ignore[method-assign]
    try:
        _sync()
        with timed_section() as t:
            hydra_q = pipe.infer_query_hydra(info_q.T.contiguous(), hydra_tail=hydra_tail).to(device)
        _sync()
        return t[0], unet_calls, hydra_q
    finally:
        pipe.pre_diff.unet.forward = orig_forward  # type: ignore[method-assign]


@torch.no_grad()
def profile_mod4_offline(pipe: Mod4Pipeline, pairs: list, indices: list[int], *, device: torch.device, hydra_tail: int, batch: int = 32) -> dict:
    n = len(indices)
    with timed_section() as t_info:
        heads = []
        for gi in indices:
            _, info_post = pair_to_info_xy8(pairs[gi], device=torch.device("cpu"))
            heads.append(info_post.T.contiguous())
    info_sec = t_info[0]

    with timed_section() as t_diff:
        for i0 in range(0, n, batch):
            i1 = min(i0 + batch, n)
            batch_heads = torch.stack(heads[i0:i1], dim=0).to(device)
            _ = pipe.infer_gallery_hydra_batch(batch_heads, hydra_tail=hydra_tail)
        _sync()
    diff_sec = t_diff[0]

    return {
        "offline_info_extract_sec": info_sec,
        "offline_gallery_diffusion_sec": diff_sec,
        "offline_index_total_sec": info_sec + diff_sec,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=str, default="100,500,1000,4000")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--gallery-batch", type=int, default=None)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output-json", type=str, default="cursor_test/match_efficiency/out/mod4_profile.json")
    args = p.parse_args()

    device = torch.device(args.device)
    sizes = [int(x) for x in args.sizes.split(",")]
    max_n = max(sizes) + 10
    pairs = make_synthetic_pairs(max_n, seed=42)

    rows = []
    # Build full index once at max N
    print(f"[mod4_profile] offline index N={max(sizes)} ...", flush=True)
    gal_full = build_gallery_index("mod4", pairs, list(range(max(sizes))), device=device)

    for n in sizes:
        gb = args.gallery_batch if args.gallery_batch else (n if n <= 500 else min(n, 256))
        gal = slice_gallery_index(gal_full, n)
        pipe: Mod4Pipeline = gal.data["model"]
        pipe.eval()

        acc: dict[str, float] = {}
        unet_calls = 0
        for rep in range(int(args.repeats)):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            on = profile_mod4_online(
                pipe,
                pairs[n],
                gal.data["info_post"],
                gal.data["hydra_post"],
                device=device,
                gallery_batch=gb,
                hydra_tail=HYDRA_TAIL,
            )
            unet_calls = on["online_query_unet_calls"]
            for k, v in on.items():
                acc[k] = acc.get(k, 0.0) + float(v)

        row = {k: v / int(args.repeats) for k, v in acc.items()}
        row["n_gallery"] = n
        row["gallery_batch"] = gb
        row["hydra_tail"] = HYDRA_TAIL
        row["k_select_vanilla"] = K_SELECT_VANILLA
        row["k_select_rl"] = K_SELECT_RL
        row["diffusion_steps"] = DIFFUSION_STEPS
        row["online_query_unet_calls"] = unet_calls
        row["offline_index_total_sec"] = gal_full.build_sec
        rows.append(row)

        print(f"\n=== mod4 N={n} (median of {args.repeats} online runs) ===", flush=True)
        _print_row(row)

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}", flush=True)


def _print_row(r: dict) -> None:
    e2e = r["online_e2e_sec"]
    def pct(x: float) -> str:
        return f"{100.0 * x / e2e:.1f}%" if e2e > 0 else "—"

    print(f"  [OFFLINE once] index_total={r['offline_index_total_sec']:.2f}s (not in online e2e)", flush=True)
    print(f"  [ONLINE query]", flush=True)
    print(f"    preprocess:        {r['online_preprocess_sec']*1000:.2f} ms  ({pct(r['online_preprocess_sec'])})", flush=True)
    print(f"    query diffusion:   {r['online_query_diffusion_sec']:.3f} s  ({pct(r['online_query_diffusion_sec'])})  unet_calls={int(r['online_query_unet_calls'])}", flush=True)
    print(f"  [ONLINE gallery N={r['n_gallery']}]", flush=True)
    print(f"    H2D load index:    {r['online_gallery_h2d_sec']*1000:.2f} ms  ({pct(r['online_gallery_h2d_sec'])})", flush=True)
    print(f"    tail slice k={r.get('k_select_vanilla', r.get('k_select', '?'))}: {r['online_tail_select_sec']*1000:.2f} ms  ({pct(r['online_tail_select_sec'])})", flush=True)
    print(f"    match transformer: {r['online_match_forward_sec']:.3f} s  ({pct(r['online_match_forward_sec'])})", flush=True)
    print(f"    argmax:            {r['online_argmax_sec']*1000:.2f} ms", flush=True)
    print(f"  ONLINE E2E:          {e2e:.3f} s   peak_vram={r['peak_vram_mib']:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
