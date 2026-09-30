#!/usr/bin/env python3
"""Sweep gallery sizes; scratch models only (no ckpt load).

14 methods: 13 baselines + cursor_mod_4 (see constants.ALL_14_METHODS).
Gallery index built once at max N; online only query diffusion for mod4.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from match_efficiency.constants import ALL_14_METHODS, HYDRA_TAIL  # noqa: E402
from match_efficiency.count_params import count_method_params  # noqa: E402
from match_efficiency.gallery import build_gallery_index, slice_gallery_index  # noqa: E402
from match_efficiency.retrieval import run_retrieval_median  # noqa: E402
from match_efficiency.traj_data import make_synthetic_pairs  # noqa: E402

DEFAULT_SIZES = (100, 200, 500, 1000, 2000, 4000)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=str, default=",".join(str(s) for s in DEFAULT_SIZES))
    p.add_argument(
        "--methods",
        type=str,
        default=",".join(ALL_14_METHODS),
        help="Default: all 14 (13 baselines + mod4).",
    )
    p.add_argument("--n-queries", type=int, default=3)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hydra-tail", type=int, default=HYDRA_TAIL)
    p.add_argument("--gallery-online", action="store_true", help="Include gallery encode/diffusion in online T_gallery.")
    p.add_argument("--output-json", type=str, default="cursor_test/match_efficiency/out/sweep_sizes.json")
    p.add_argument("--output-md", type=str, default="cursor_test/match_efficiency/out/sweep_sizes.md")
    return p.parse_args()


def gallery_batch_for(n: int, device: torch.device) -> int:
    if device.type != "cuda":
        return n
    if n <= 500:
        return n
    if n <= 1000:
        return min(n, 512)
    return min(n, 256)


def main():
    args = parse_args()
    sizes = sorted({int(x) for x in args.sizes.split(",") if x.strip()})
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    device = torch.device(args.device)
    max_n = max(sizes)
    pairs = make_synthetic_pairs(max_n + int(args.n_queries) + 50, seed=int(args.seed))
    max_indices = list(range(max_n))

    # Pre-build every method's gallery once at max N (offline).
    index_cache: dict[str, object] = {}
    for method in methods:
        print(f"[sweep] offline index {method} N={max_n} ...", flush=True)
        index_cache[method] = build_gallery_index(
            method,
            pairs,
            max_indices,
            device=device,
            hydra_tail=int(args.hydra_tail),
            gallery_online=bool(args.gallery_online),
        )
        print(f"  t_index={index_cache[method].build_sec:.1f}s", flush=True)  # type: ignore[union-attr]

    rows: list[dict] = []
    for n in sizes:
        gb = gallery_batch_for(n, device)
        query_pairs = [pairs[n + i] for i in range(int(args.n_queries))]
        print(f"\n=== online N={n} gallery_batch={gb} ===", flush=True)
        for method in methods:
            if device.type == "cuda":
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            print(f"  {method} ...", flush=True, end=" ")
            params = count_method_params(method)
            gal_full = index_cache[method]
            gal = slice_gallery_index(gal_full, n)  # type: ignore[arg-type]
            timing = run_retrieval_median(
                method,
                query_pairs,
                gal,
                pairs=pairs,
                device=device,
                gallery_batch=gb,
                hydra_tail=int(args.hydra_tail),
                gallery_online=bool(args.gallery_online),
            )
            row = {
                "method": method,
                "n_gallery": n,
                "gallery_batch": gb,
                "p_total": params["p_total"],
                "gallery_online": bool(args.gallery_online),
                "t_index_sec": round(gal_full.build_sec, 4),  # type: ignore[union-attr]
                "t_index_amortized": round(gal_full.build_sec / max_n, 6),  # type: ignore[union-attr]
                "s_index_elements": gal.storage_elements,
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in timing.as_dict().items()},
            }
            rows.append(row)
            print(
                f"T_e2e={row['t_e2e']:.3f}s (q={row['t_query']:.3f} gal={row['t_gallery']:.3f})",
                flush=True,
            )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "scratch_models": True,
            "no_ckpt_load": True,
            "methods": methods,
            "n_methods": len(methods),
            "gallery_index": (
                "online gallery encode/diffusion in T_gallery"
                if args.gallery_online
                else "offline once at max N; mod4 online query-only diffusion"
            ),
            "gallery_online": bool(args.gallery_online),
            "sizes": sizes,
            "device": str(device),
            "n_queries": int(args.n_queries),
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = _to_markdown(rows, methods, sizes)
    out_md = Path(args.output_md)
    out_md.write_text(md, encoding="utf-8")
    print(f"\nWrote {out_json}\nWrote {out_md}", flush=True)
    print(md, flush=True)


def _to_markdown(rows: list[dict], methods: list[str], sizes: list[int]) -> str:
    lines = [
        "# Gallery size sweep — 14 methods (scratch, no ckpt)",
        "",
        "13 baselines + cursor_mod_4. Online `T_e2e` = `T_query` + `T_gallery`.",
    ]
    if rows and rows[0].get("gallery_online"):
        lines.append("Gallery encode/diffusion counted in **online** `T_gallery`.")
    lines.append("")
    for metric, key in [
        ("Params (M)", "p_total"),
        ("T_e2e (s)", "t_e2e"),
        ("T_query (s)", "t_query"),
        ("T_gallery (s)", "t_gallery"),
        ("VRAM (MiB)", "peak_vram_mib"),
        ("RSS delta (MiB)", "peak_rss_mib"),
    ]:
        lines.append(f"## {metric}")
        lines.append("")
        hdr = "| method | " + " | ".join(str(s) for s in sizes) + " |"
        sep = "|---|" + "|".join("---:" for _ in sizes) + "|"
        lines.append(hdr)
        lines.append(sep)
        lookup = {(r["method"], r["n_gallery"]): r[key] for r in rows}
        for m in methods:
            cells = [m]
            for n in sizes:
                v = lookup.get((m, n))
                if v is None:
                    cells.append("—")
                elif key == "p_total":
                    cells.append(f"{v / 1e6:.3f}")
                else:
                    cells.append(f"{v:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
