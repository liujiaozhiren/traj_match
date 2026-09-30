#!/usr/bin/env python3
"""Compact table: per-gap retrained RL vs vanilla @ frac=0.15, α=1."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALID_ROOT = Path(
    os.environ.get(
        "GAP_VALID_ROOT",
        str(ROOT / "cursor_data_proc/out/gap_mod4_v2cache_rl_pergap"),
    )
)
RL_ROOT = Path(os.environ.get("GAP_RL_ROOT", str(ROOT / "cursor_data_proc/out")))


def _pick(gap: int, mode: str, frac: float = 0.15) -> str:
    p = VALID_ROOT / f"gap{gap}" / "grid.json"
    if not p.is_file():
        return "—"
    for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
        if r.get("method") == mode and abs(float(r["valid_sample_frac"]) - frac) < 1e-9:
            return f"acc={r['acc']:.4f} HR10={r['hr10']:.4f}"
    return "—"


def main() -> None:
    lines = [
        "# Per-gap RL retrain + valid (gap pkl + v2 cache)\n",
        f"Valid: `{VALID_ROOT.relative_to(ROOT)}`\n",
        "\n| gap | match_vanilla @15% | rl_selector @15% (retrained) |\n",
        "|---|---|---|\n",
    ]
    for g in range(1, 7):
        sel = list((RL_ROOT / f"gap{g}_rl_retrain_v2cache").glob("best_selector_*.pt"))
        rl = _pick(g, "rl_selector")
        if sel and rl != "—":
            rl += f" (`{sel[0].name}`)"
        lines.append(f"| {g} | {_pick(g, 'match_vanilla')} | {rl} |\n")
    out = VALID_ROOT / "summary_rl_retrain.md"
    out.write_text("".join(lines), encoding="utf-8")
    print(out)
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
