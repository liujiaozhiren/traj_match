"""Render the case-study figure for one direction.

Each method gets a panel showing:
  - the anchor (query) trajectory      (bold black)
  - the ground-truth counterpart       (green, dashed, thick)
  - the method's top-5 retrieved counterparts (rank 1 strong -> rank 5 faint),
    numbered 1..5; if the GT is among them it is outlined in green.

Direction A (pre_anchor):  anchor = PRE,  retrieved/GT = POST.
Direction B (post_anchor): anchor = POST, retrieved/GT = PRE.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

ANCHOR_C = "#111111"
GT_C = "#1b9e3a"
TOP_CMAP = ["#d7301f", "#ef6548", "#fc8d59", "#fdbb84", "#fdd49e"]  # rank1..5
# pad=1.1 was 2x zoom vs original 2.2; another 1.5x zoom → 1.1/1.5.
FOCUS_PAD = 1.1 / 1.5
MAP_PATH = Path(__file__).resolve().parent / "map.jpg"
MAP_CROP = 0.10
MAP_OPACITY = 0.42
PANEL_WSPACE = 0.04
PANEL_HSPACE = 0.22
NCOL, NROW = 4, 2

KEEP_METHODS = [
    "dtw", "hausdorff", "traj2simvec", "neutraj",
    "st2vec", "dan", "attnmove", "mod4",
]
PRETTY = {
    "dtw": "DTW", "hausdorff": "Hausdorff",
    "traj2simvec": "Traj2SimVec", "neutraj": "NeuTraj",
    "st2vec": "ST2Vec", "dan": "DAN", "attnmove": "AttnMove",
    "mod4": "MMHM",
}


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
    """Consistent square window centred on anchor+GT (same for every panel)."""
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


def render(data, direction: str, out_path: str):
    dinfo = data["directions"][direction]
    q = dinfo["anchor_pool_idx"]
    per_method = dinfo["per_method"]

    if direction == "pre_anchor":
        anchor_xy = data["pool_pre_lonlat"][q]
        counterpart = data["pool_post_lonlat"]   # retrieved/GT are posts
    else:
        anchor_xy = data["pool_post_lonlat"][q]
        counterpart = data["pool_pre_lonlat"]     # retrieved/GT are pres
    gt_xy = counterpart[q]
    xlim, ylim = _focus_window(anchor_xy, gt_xy)

    methods = [m for m in KEEP_METHODS if m in per_method]
    fig, axes = plt.subplots(NROW, NCOL, figsize=(3. * NCOL, 3. * NROW))
    axes = axes.ravel()
    map_bg = _load_map_bg()

    for ax in axes[len(methods):]:
        ax.axis("off")

    for ax, m in zip(axes, methods):
        rec = per_method[m]
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_clip_on(True)
        _draw_map(ax, xlim, ylim, map_bg)
        # top5 retrieved, faint -> strong (draw rank5 first so rank1 is on top)
        for rnk in range(len(rec["top5_idx"]) - 1, -1, -1):
            idx = rec["top5_idx"][rnk]
            xy = counterpart[idx]
            is_gt = idx == q
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
            if is_gt:
                _plot_traj(ax, xy, color=GT_C, lw=4.0, alpha=0.35, zorder=2)

        # ground truth (dashed)
        _plot_traj(ax, gt_xy, color=GT_C, lw=2.4, dashes=(0.8, 2.6),
                   dash_capstyle="round", alpha=0.9, zorder=7)
        # anchor query
        _plot_traj(ax, anchor_xy, color=ANCHOR_C, lw=2.6, alpha=0.9, zorder=8, marker_start=True)

        ax.set_title(PRETTY.get(m, m), fontsize=20, pad=10)
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
    print(f"[plot] wrote {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/case_study.pkl")
    ap.add_argument("--direction", default="both", choices=["pre_anchor", "post_anchor", "both"])
    ap.add_argument("--out-prefix", default="out/case_study")
    args = ap.parse_args()
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    dirs = ["pre_anchor", "post_anchor"] if args.direction == "both" else [args.direction]
    for d in dirs:
        render(data, d, f"{args.out_prefix}_{d}.png")


if __name__ == "__main__":
    main()
