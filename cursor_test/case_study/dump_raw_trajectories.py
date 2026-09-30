"""Dump the raw (clean lon/lat) trajectories behind the case-study figures to a
single markdown file.

For each direction (pre_anchor / post_anchor) and each of the 13 plotted methods,
we emit the 7 trajectories shown in that panel:
  query, ground-truth, rank1(top1), rank2, rank3, rank4, rank5.
query + ground-truth are identical across a direction's panels but are repeated
in every method section (so each section is self-contained), per request.

Coordinates are the exact clean polylines drawn in the figure (lon/lat degrees),
i.e. data["pool_pre_lonlat"] / data["pool_post_lonlat"]. NOTE: model *scoring*
uses a different 12-point noise-blended resample, not dumped here.
"""

from __future__ import annotations

import argparse
import json
import pickle

PRETTY = {
    "dtw": "DTW", "hausdorff": "Hausdorff", "frechet": "Fréchet", "sspd": "SSPD",
    "t3s": "T3S", "traj2simvec": "Traj2SimVec", "neutraj": "NeuTraj",
    "trajgat": "TrajGAT", "st2vec": "ST2Vec", "trajcl": "TrajCL",
    "dan": "DAN", "attnmove": "AttnMove", "mod4": "Ours (mod4)",
}


def _coords(xy) -> str:
    """JSON array of [lon, lat] pairs (full float precision)."""
    return json.dumps([[float(p[0]), float(p[1])] for p in xy])


def _emit_traj(lines, label, xy, *, pool_idx, gidx, extra=""):
    lines.append(f"- **{label}** — pool#{pool_idx}, gidx={gidx}, npts={len(xy)}{extra}")
    lines.append("")
    lines.append("```json")
    lines.append(_coords(xy))
    lines.append("```")
    lines.append("")


def render_direction(data, direction: str, lines: list[str]):
    meta = data["meta"]
    dinfo = data["directions"][direction]
    q = dinfo["anchor_pool_idx"]
    gidx_of = data["pool_gidx"]
    per_method = dinfo["per_method"]

    if direction == "pre_anchor":
        query_pool = data["pool_pre_lonlat"]      # query = PRE
        target_pool = data["pool_post_lonlat"]    # GT / retrieved = POST
        q_name, t_name = "query (PRE)", "POST"
    else:
        query_pool = data["pool_post_lonlat"]     # query = POST
        target_pool = data["pool_pre_lonlat"]     # GT / retrieved = PRE
        q_name, t_name = "query (POST)", "PRE"

    methods = [m for m in meta["method_order"] if m in per_method]

    lines.append(f"# {direction}")
    lines.append("")
    lines.append(
        f"- anchor pool#{q} (gidx={gidx_of[q]}); pool_size={meta['pool_size']}; "
        f"query is the {q_name.split()[1]}, retrieved/GT are {t_name}.")
    lines.append(f"- mod4 (hero) rank = {dinfo['hero_rank']} "
                 f"({'top1 hit' if dinfo['hero_rank'] == 0 else 'rank ' + str(dinfo['hero_rank'] + 1)})")
    lines.append(f"- methods ({len(methods)}): {', '.join(PRETTY.get(m, m) for m in methods)}")
    lines.append("")

    for m in methods:
        rec = per_method[m]
        gr = rec["gt_rank"]
        tag = "top1 hit" if gr == 0 else (
            f"GT in top5 (rank {gr + 1})" if rec["gt_in_top5"] else f"GT miss (rank {gr + 1})")
        lines.append(f"## {direction} · {PRETTY.get(m, m)}  ({tag})")
        lines.append("")

        # 1) query (shared), 2) ground-truth (shared)
        _emit_traj(lines, q_name, query_pool[q], pool_idx=q, gidx=gidx_of[q])
        _emit_traj(lines, f"ground-truth {t_name}", target_pool[q], pool_idx=q, gidx=gidx_of[q])

        # 3..7) rank1..rank5 retrieved
        for rnk, idx in enumerate(rec["top5_idx"]):
            is_gt = idx == q
            label = f"rank{rnk + 1}" + (" (top1)" if rnk == 0 else "")
            extra = "  ← this IS the ground-truth" if is_gt else ""
            _emit_traj(lines, label, target_pool[idx], pool_idx=idx, gidx=gidx_of[idx], extra=extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="out/case_study.pkl")
    ap.add_argument("--out", default="out/case_study_raw_trajectories.md")
    args = ap.parse_args()

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    meta = data["meta"]
    lines: list[str] = []
    lines.append("# Case study — raw trajectories (clean lon/lat, as plotted)")
    lines.append("")
    lines.append(f"- pairs_pkl: `{meta['pairs_pkl']}`")
    lines.append(f"- pool_size={meta['pool_size']}, frac={meta['frac']}, "
                 f"pool_seed={meta['pool_seed']}, sample_ratio={meta['sample_ratio']}, "
                 f"sample_seed={meta['sample_seed']}")
    lines.append("- Coordinates are `[lon, lat]` in degrees — the exact clean polylines drawn in "
                 "the figures. Each panel lists 7 trajectories: query, ground-truth, rank1(top1), "
                 "rank2..rank5. query/GT are identical within a direction but repeated per method.")
    lines.append("- NOTE: model scoring internally uses a 12-point noise-blended resample; that "
                 "version is NOT dumped here (these are the plotted coordinates).")
    lines.append("")

    for direction in ("pre_anchor", "post_anchor"):
        if direction in data["directions"]:
            render_direction(data, direction, lines)

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"[dump] wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
