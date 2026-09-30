"""mod4 score worker (self-contained).

Computes a k x k retrieval score matrix S over a fixed pool of pairs, using the
full mod4 pipeline: precomputed hydra cache + frozen MatchModel + RL selector.
S[i, j] = match logit of pre(pool[i]) vs post(pool[j]); diagonal = ground truth.

Run as a SEPARATE process from the baseline worker (the two stacks register
different top-level ``cfg`` modules).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys

import numpy as np
import torch

import _bootstrap  # noqa: F401  (puts vendor/ on sys.path)

sys.modules["cfg"] = importlib.import_module("cursor_mod_4.cfg")
import cfg  # noqa: E402

import registry  # noqa: E402
from common import load_pairs  # noqa: E402

from cursor_mod_4.geo_coords import match_traj_pts_to_xy_m  # noqa: E402
from cursor_mod_4.match import MatchModel  # noqa: E402
from cursor_mod_4.match.matcher import MatchInputs  # noqa: E402
from cursor_mod_4.rl.linucb_selector import DeepLinUCBSelector, SelectorCheckpoint  # noqa: E402


def _pair_to_info_xy8(item):
    _, A, B, _, _, _, _ = item
    A_xy = match_traj_pts_to_xy_m(A)[: cfg.diff_pre_len]
    Brev = list(reversed(B))
    B_xy = match_traj_pts_to_xy_m(Brev)[: cfg.diff_pre_len]
    return (
        torch.tensor(np.asarray(A_xy, dtype=np.float32)),
        torch.tensor(np.asarray(B_xy, dtype=np.float32)),
    )


def _build_selector(ckpt_path: str, device: str) -> DeepLinUCBSelector:
    ck = SelectorCheckpoint.load(ckpt_path)
    meta = ck.meta if isinstance(ck.meta, dict) else {}
    sel = DeepLinUCBSelector(
        k_select=int(meta.get("k_select", 16)),
        rep_dim=int(meta.get("rep_dim", 32)),
        alpha=float(meta.get("alpha", 1.0)),
        lambda_=float(meta.get("lambda_", 1.0)),
        joint_select=bool(meta.get("joint_select", False)),
        device=device,
        lr=1e-4,
    )
    sel.load_state(ck)
    sel.freeze(True)
    return sel


@torch.no_grad()
def _match_logits(match, i_pre, i_post, h_pre, h_post):
    x = MatchInputs(graph=None, info_pre=i_pre, info_post=i_post, hydra_pre=h_pre, hydra_post=h_post)
    return match.match(x, pairs=True).view(-1)


@torch.no_grad()
def score_pool(pool_gidx, *, mode, valid_batch=10):
    dev = cfg.device
    all_pairs = load_pairs(registry.PAIRS_PKL)
    pool_pairs = [all_pairs[g] for g in pool_gidx]
    k = len(pool_pairs)

    # info tensors (meters, 8 pts) for every pool pair
    info_pre = torch.zeros((k, cfg.diff_pre_len, 2), dtype=torch.float32)
    info_post = torch.zeros((k, cfg.diff_pre_len, 2), dtype=torch.float32)
    for i, item in enumerate(pool_pairs):
        ip, io = _pair_to_info_xy8(item)
        info_pre[i] = ip
        info_post[i] = io

    # hydra tails from precomputed cache, indexed by global pair id
    cache = torch.load(str(registry.MOD4["cache"]), map_location="cpu")
    N = min(int(registry.MOD4["hydra_tail"]), int(cache["hydra_pre"].shape[1]))
    gidx_t = torch.tensor(pool_gidx, dtype=torch.long)
    hydra_pre = cache["hydra_pre"][gidx_t, :N]
    hydra_post = cache["hydra_post"][gidx_t, :N]

    match = MatchModel(dim=128, scheme=int(registry.MOD4["match_scheme"])).to(dev)
    match.load_state_dict(torch.load(str(registry.MOD4["match"]), map_location=dev), strict=True)
    match.eval()
    match.freeze(True)

    selector = None
    if mode == "rl_selector":
        selector = _build_selector(str(registry.MOD4["selector"]), dev)
        k_select = int(selector.k_select)
    else:
        k_select = int(registry.MOD4.get("k_select", 16))

    post_info = info_post.to(dev)
    post_hydra = hydra_post.to(dev)
    scores = torch.zeros((k, k), device=dev)
    n_batches = (k + valid_batch - 1) // valid_batch
    for bi in range(n_batches):
        i0 = bi * valid_batch
        i1 = min(i0 + valid_batch, k)
        b = i1 - i0
        q_info_pre = info_pre[i0:i1].to(dev)
        q_hydra_pre = hydra_pre[i0:i1].to(dev)
        B1, B2 = b, k
        i_pre = q_info_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
        i_post = post_info.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
        h_pre = q_hydra_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)
        h_post = post_hydra.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1 * B2, N, cfg.diff_infer_len, 2)
        if mode == "match_vanilla":
            h_pre_use = h_pre[:, :k_select]
            h_post_use = h_post[:, :k_select]
        else:
            h_pre_use, h_post_use, _, _ = selector.choose(i_pre, i_post, h_pre, h_post)
        logits = _match_logits(match, i_pre, i_post, h_pre_use, h_post_use).view(B1, B2)
        scores[i0:i1] = logits
    return scores.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-json", required=True, help="JSON file with {'pool_gidx': [...]}")
    ap.add_argument("--out", required=True, help="output npz path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--modes", default="rl_selector,match_vanilla")
    args = ap.parse_args()
    if args.device:
        cfg.device = args.device

    with open(args.pool_json) as f:
        pool_gidx = json.load(f)["pool_gidx"]

    out = {}
    name_map = {"rl_selector": "mod4", "match_vanilla": "mod4_vanilla"}
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        print(f"[worker_mod4] scoring mode={mode} pool={len(pool_gidx)}", flush=True)
        S = score_pool(pool_gidx, mode=mode)
        out[name_map.get(mode, mode)] = S
        diag_top1 = float((S.argmax(axis=1) == np.arange(len(pool_gidx))).mean())
        print(f"[worker_mod4] {name_map.get(mode, mode)} top1-acc on pool = {diag_top1:.4f}", flush=True)

    np.savez(args.out, **out)
    print(f"[worker_mod4] wrote {args.out} methods={list(out)}", flush=True)


if __name__ == "__main__":
    main()
