#!/usr/bin/env python3
"""Run isolated per-method memory/timing sweep; write JSON + markdown tables."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PKG = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from match_efficiency.constants import ALL_14_METHODS  # noqa: E402

DEFAULT_SIZES = (100, 200, 500, 1000, 2000, 5000)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=str, default=",".join(str(s) for s in DEFAULT_SIZES))
    p.add_argument("--methods", type=str, default=",".join(ALL_14_METHODS))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--n-queries", type=int, default=3)
    p.add_argument("--output-json", type=str, default="cursor_test/match_efficiency/out/isolated_full14.json")
    p.add_argument("--output-md", type=str, default="cursor_test/SUMMARY_FULL14.md")
    return p.parse_args()


def _run_one(method: str, n: int, device: str, n_queries: int) -> dict:
    cmd = [
        sys.executable,
        str(_PKG / "measure_isolated.py"),
        "--method",
        method,
        "--n-gallery",
        str(n),
        "--device",
        device,
        "--n-queries",
        str(n_queries),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"{method} N={n} failed:\n{proc.stderr}")
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def _cell(v, key: str) -> str:
    if v is None:
        return "—"
    if key in ("p_learnable", "p_aux", "p_total"):
        return str(int(v))
    return f"{float(v):.6f}"


def _table(title: str, key: str, methods: list[str], sizes: list[int], lookup: dict) -> list[str]:
    lines = [f"## {title}", ""]
    hdr = "| method | " + " | ".join(str(s) for s in sizes) + " |"
    sep = "|---|" + "|".join("---:" for _ in sizes) + "|"
    lines.extend([hdr, sep])
    for m in methods:
        cells = [m] + [_cell(lookup.get((m, n), {}).get(key), key) for n in sizes]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def main():
    args = parse_args()
    sizes = sorted({int(x) for x in args.sizes.split(",") if x.strip()})
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rows: list[dict] = []
    for n in sizes:
        for method in methods:
            print(f"[isolated] {method} N={n} ...", flush=True)
            row = _run_one(method, n, args.device, int(args.n_queries))
            rows.append(row)
            print(
                f"  rss={row['peak_rss_mib']:.3f} vram={row['peak_vram_mib']:.3f} "
                f"t_e2e={row['t_e2e']:.6f}",
                flush=True,
            )

    out_json = Path(args.output_json)
    if not out_json.is_absolute():
        out_json = _ROOT.parent / out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "isolated_process": True,
            "gallery_online": True,
            "methods": methods,
            "sizes": sizes,
            "device": args.device,
            "n_queries": int(args.n_queries),
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lookup = {(r["method"], r["n_gallery"]): r for r in rows}
    p_row = rows[0] if rows else {}
    static = {m: count_static(methods, lookup, m) for m in methods}

    lines = ["# SUMMARY_FULL14", ""]
    lines.extend(_static_table("可学习参数量 (elements)", "p_learnable", methods, static))
    lines.extend(_static_table("辅助结构大小 (elements)", "p_aux", methods, static))
    lines.extend(_static_table("总用量 (elements)", "p_total", methods, static))
    lines.extend(_table("进程峰值 RSS (MiB)", "peak_rss_mib", methods, sizes, lookup))
    lines.extend(_table("峰值显存 (MiB)", "peak_vram_mib", methods, sizes, lookup))
    lines.extend(_table("T_e2e (s)", "t_e2e", methods, sizes, lookup))
    lines.extend(_table("T_query (s)", "t_query", methods, sizes, lookup))
    lines.extend(_table("T_gallery (s)", "t_gallery", methods, sizes, lookup))

    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = _ROOT.parent / out_md
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}\nWrote {out_md}", flush=True)


def count_static(methods: list[str], lookup: dict, method: str) -> dict:
    for n in sorted({k[1] for k in lookup if k[0] == method}):
        r = lookup.get((method, n))
        if r:
            return r
    return {"p_learnable": 0, "p_aux": 0, "p_total": 0}


def _static_table(title: str, key: str, methods: list[str], static: dict) -> list[str]:
    lines = [f"## {title}", ""]
    lines.append("| method | elements |")
    lines.append("|---|---:|")
    for m in methods:
        lines.append(f"| {m} | {int(static[m][key])} |")
    lines.append("")
    return lines


if __name__ == "__main__":
    main()
