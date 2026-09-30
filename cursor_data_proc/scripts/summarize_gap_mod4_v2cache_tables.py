#!/usr/bin/env python3
"""Summarize gap_mod4_v2cache_valid grid.json → markdown tables (α=1)."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(
    os.environ.get(
        "GAP_MOD4_OUT",
        str(ROOT / "cursor_data_proc/out/gap_mod4_v2cache_valid"),
    )
)


def load_rows(gap: int) -> list[dict]:
    p = OUT / f"gap{gap}" / "grid.json"
    if not p.is_file():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("rows", [])


def main() -> None:
    fracs = [0.15, 0.10, 0.05, 0.02]
    modes = ["match_vanilla", "rl_selector"]
    lines: list[str] = [
        "# gap1~6 mod_4 valid (gap A/B info + v2 hydra cache)\n",
        f"Source: `{OUT.relative_to(ROOT)}/gap*/grid.json`\n",
    ]
    for mode in modes:
        lines.append(f"\n## {mode} (α=1.0)\n")
        header = "| gap | " + " | ".join(f"frac={f:g}" for f in fracs) + " |"
        sep = "|---|" + "|".join("---:" for _ in fracs) + "|"
        lines.extend([header, sep])
        for gap in range(1, 7):
            rows = {r["valid_sample_frac"]: r for r in load_rows(gap) if r.get("method") == mode}
            cells = []
            for f in fracs:
                r = rows.get(f)
                if r is None:
                    cells.append("—")
                else:
                    cells.append(
                        f"acc={r['acc']:.4f} HR5={r['hr5']:.4f} HR10={r['hr10']:.4f} HR20={r['hr20']:.4f}"
                    )
            lines.append("| " + str(gap) + " | " + " | ".join(cells) + " |")
    out_md = OUT / "summary_tables.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_md)
    print(out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
