"""Re-draw the case-study figures using ONLY the coordinates parsed from
``out/case_study_raw_trajectories.md`` — a verification that the md carries the
exact data behind the original figures. Rendering style mirrors
``plot_case_study.py`` 1:1. Outputs ``*_from_md.png``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

ANCHOR_C = "#111111"
GT_C = "#1b9e3a"
TOP_CMAP = ["#d7301f", "#ef6548", "#fc8d59", "#fdbb84", "#fdd49e"]
# pad=1.1 was 2x zoom vs original 2.2; another 1.5x zoom → 1.1/1.5.
FOCUS_PAD = 1.1 / 1.5
MAP_PATH = Path(__file__).resolve().parent / "map.jpg"
MAP_CROP = 0.10          # trim 10% from each edge (labels / bus icons)
MAP_OPACITY = 0.42       # fade toward white so trajectories stay readable
PANEL_WSPACE = 0.04
PANEL_HSPACE = 0.22
NCOL, NROW = 4, 2

_KNOWN_PRETTY = [
    "DTW", "Hausdorff", "Fréchet", "SSPD", "T3S", "Traj2SimVec", "NeuTraj",
    "TrajGAT", "ST2Vec", "TrajCL", "DAN", "AttnMove", "Ours (mod4)",
]
# Display order for the 2x4 figure. md pretty name → panel title.
KEEP_PANELS = [
    ("DTW", "DTW"),
    ("Hausdorff", "Hausdorff"),
    ("Traj2SimVec", "Traj2SimVec"),
    ("NeuTraj", "NeuTraj"),
    ("ST2Vec", "ST2Vec"),
    ("DAN", "DAN"),
    ("AttnMove", "AttnMove"),
    ("Ours (mod4)", "MMHM"),
]


def parse_md(path: str) -> dict:
    """Parse the dumped md back into {direction: {meta..., panels:[...]}}."""
    with open(path) as f:
        text = f.read()
    lines = text.split("\n")

    out: dict = {}
    cur_dir = None
    cur_panel = None
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln in ("# pre_anchor", "# post_anchor"):
            cur_dir = ln[2:].strip()
            out[cur_dir] = {"q": None, "gidx": None, "pool_size": None,
                            "hero_rank": None, "panels": []}
            cur_panel = None
        elif cur_dir and ln.startswith("- anchor pool#"):
            m = re.search(r"anchor pool#(\d+) \(gidx=(\d+)\); pool_size=(\d+)", ln)
            if m:
                out[cur_dir]["q"] = int(m.group(1))
                out[cur_dir]["gidx"] = int(m.group(2))
                out[cur_dir]["pool_size"] = int(m.group(3))
        elif cur_dir and ln.startswith("- mod4 (hero) rank ="):
            m = re.search(r"rank = (\d+)", ln)
            if m:
                out[cur_dir]["hero_rank"] = int(m.group(1))
        elif cur_dir and ln.startswith("## "):
            # "## pre_anchor · DTW  (GT miss (rank 18))" — note pretty may itself
            # contain parens ("Ours (mod4)"), so match against known names.
            body = ln[3:].split("·", 1)[1].strip()
            pretty = body
            tag = ""
            for cand in sorted(_KNOWN_PRETTY, key=len, reverse=True):
                if body.startswith(cand):
                    pretty = cand
                    rest = body[len(cand):].strip()
                    if rest.startswith("(") and rest.endswith(")"):
                        rest = rest[1:-1]
                    tag = rest
                    break
            cur_panel = {"pretty": pretty, "tag": tag,
                         "query": None, "gt": None, "ranks": [], "ranks_is_gt": []}
            out[cur_dir]["panels"].append(cur_panel)
        elif cur_panel is not None and ln.startswith("- **"):
            label = ln[4:].split("**", 1)[0]
            is_gt_flag = "this IS the ground-truth" in ln
            # find the json payload: next ```json then one line then ```
            j = i + 1
            while j < n and lines[j].strip() != "```json":
                j += 1
            coords = json.loads(lines[j + 1]) if j + 1 < n else []
            xy = np.asarray(coords, dtype=np.float64)
            if label.startswith("query"):
                cur_panel["query"] = xy
            elif label.startswith("ground-truth"):
                cur_panel["gt"] = xy
            elif label.startswith("rank"):
                cur_panel["ranks"].append(xy)
                cur_panel["ranks_is_gt"].append(is_gt_flag)
            i = j + 2
            continue
        i += 1
    return out


def _plot_traj(ax, xy, marker_start=False, **kw):
    kw.setdefault("clip_on", True)
    ax.plot(xy[:, 0], xy[:, 1], **kw)
    if marker_start:
        ax.scatter(
            [xy[0, 0]], [xy[0, 1]], s=18, color=kw.get("color"),
            zorder=kw.get("zorder", 5) + 0.5, edgecolors="white", linewidths=0.5,
            clip_on=True,
        )


def _in_window(x, y, xlim, ylim) -> bool:
    return xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]


def _focus_window(anchor_xy, gt_xy, pad=FOCUS_PAD):
    pts = np.concatenate([anchor_xy, gt_xy], axis=0)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    span = max(pts[:, 0].ptp(), pts[:, 1].ptp(), 1e-4) * pad
    return (cx - span, cx + span), (cy - span, cy + span)


def _load_map_bg() -> np.ndarray:
    im = Image.open(MAP_PATH).convert("RGB")
    w, h = im.size
    m = int(min(w, h) * MAP_CROP)
    side = min(w, h) - 2 * m
    left = (w - side) // 2
    top = (h - side) // 2
    im = im.crop((left, top, left + side, top + side))
    arr = np.asarray(im, dtype=np.float32)
    faded = arr * MAP_OPACITY + 255.0 * (1.0 - MAP_OPACITY)
    return np.clip(faded, 0, 255).astype(np.uint8)


def _draw_map(ax, xlim, ylim, img: np.ndarray) -> None:
    ax.imshow(
        img,
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        origin="upper",
        aspect="auto",
        zorder=0,
        interpolation="bilinear",
        clip_on=True,
    )


def render(direction: str, info: dict, out_path: str):
    by_pretty = {p["pretty"]: p for p in info["panels"]}
    panels = []
    titles = []
    for src, title in KEEP_PANELS:
        if src not in by_pretty:
            raise KeyError(f"missing panel {src!r} in {direction}")
        panels.append(by_pretty[src])
        titles.append(title)

    # window comes from anchor+GT (identical across panels) — use first panel
    anchor_xy = panels[0]["query"]
    gt_xy = panels[0]["gt"]
    xlim, ylim = _focus_window(anchor_xy, gt_xy)

    fig, axes = plt.subplots(NROW, NCOL, figsize=(3.2 * NCOL, 3.2 * NROW))
    axes = axes.ravel()
    map_bg = _load_map_bg()
    for ax in axes[len(panels):]:
        ax.axis("off")

    for ax, p, title in zip(axes, panels, titles):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_clip_on(True)
        _draw_map(ax, xlim, ylim, map_bg)
        ranks = p["ranks"]
        for rnk in range(len(ranks) - 1, -1, -1):
            xy = ranks[rnk]
            _plot_traj(
                ax, xy,
                color=TOP_CMAP[rnk] if rnk < len(TOP_CMAP) else TOP_CMAP[-1],
                lw=2.4 if rnk == 0 else 1.4,
                alpha=0.95 if rnk == 0 else 0.6,
                zorder=6 - rnk,
                marker_start=(rnk == 0),
            )
            ex, ey = float(xy[-1, 0]), float(xy[-1, 1])
            if _in_window(ex, ey, xlim, ylim):
                ax.annotate(
                    str(rnk + 1), xy=(ex, ey), fontsize=10,
                    color=TOP_CMAP[rnk] if rnk < len(TOP_CMAP) else TOP_CMAP[-1],
                    weight="bold", zorder=10, clip_on=True,
                    annotation_clip=True,
                )
            if p["ranks_is_gt"][rnk]:
                _plot_traj(ax, xy, color=GT_C, lw=4.0, alpha=0.35, zorder=2)

        _plot_traj(ax, p["gt"], color=GT_C, lw=2.4, dashes=(0.8, 2.6),
                   dash_capstyle="round", alpha=0.9, zorder=7)
        _plot_traj(ax, p["query"], color=ANCHOR_C, lw=2.6, alpha=0.9, zorder=8, marker_start=True)
        ax.set_title(title, fontsize=20, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

    legend = [
        Line2D([0], [0], color=ANCHOR_C, lw=2.6, label="query trajectory"),
        Line2D([0], [0], color=GT_C, lw=2.4, dashes=(0.8, 2.6),
               dash_capstyle="round", label="ground-truth answer"),
        Line2D([0], [0], color=TOP_CMAP[0], lw=2.4, label="method answer"),
        Line2D([0], [0], color=TOP_CMAP[3], lw=1.4, label="method answer (top2-5)"),
    ]
    fig.legend(
        handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.078),
        ncol=4, fontsize=16, frameon=False, handlelength=2.6,
        columnspacing=1.4, handletextpad=0.5,
    )
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.88, bottom=0.08,
        wspace=PANEL_WSPACE, hspace=PANEL_HSPACE,
    )
    fig.savefig(out_path, dpi=160)
    print(f"[plot_from_md] wrote {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="out/case_study_raw_trajectories.md")
    ap.add_argument("--out-prefix", default="out/case_study_from_md")
    args = ap.parse_args()
    parsed = parse_md(args.md)
    for direction, info in parsed.items():
        render(direction, info, f"{args.out_prefix}_{direction}.png")


if __name__ == "__main__":
    main()
