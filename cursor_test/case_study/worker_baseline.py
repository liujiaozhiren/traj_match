"""Baseline score worker (self-contained).

Computes k x k retrieval score matrices over the fixed pool for:
  - distance metrics: dtw, hausdorff, frechet, sspd  (no checkpoint)
  - repr encoders: t3s, traj2simvec, neutraj, trajgat, st2vec  (trained ckpts)
  - dan                                                          (trained ckpt)

S[i, j] = score(pre(pool[i]), post(pool[j])); diagonal = ground truth (higher = better).
Each method is wrapped in try/except so one failure does not kill the rest.

Run as a SEPARATE process from worker_mod4 (different top-level ``cfg``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback

import numpy as np
import torch

import _bootstrap  # noqa: F401

sys.modules["cfg"] = importlib.import_module("cursor_baseline.model_lib.cfg")
import cfg  # noqa: E402

import registry  # noqa: E402
from common import load_pairs, valid_global_indices  # noqa: E402

from cursor_baseline.lonlat_traj import pair_pre_post_lonlat12  # noqa: E402
from cursor_baseline.traj_distances import pairwise_neg_distance_matrix  # noqa: E402
from cursor_baseline.trajgat_build import graphs_for_valid_indices  # noqa: E402
from cursor_baseline.st2vec_build import st2vec_valid_encode_batch  # noqa: E402
from cursor_baseline.dan_mot_traj.model import DANMotTrajSimplified  # noqa: E402
from cursor_baseline.train_dan_traj import batch_standardize_lonlat_in_batch  # noqa: E402
from cursor_baseline.run_noise_blend_eval_suite import _build_repr_model  # noqa: E402

NOISE_BASE = 12345
ALPHA = 1.0  # pure noisy observation (no clean-label leak)


def _tolerant_load(model, sd) -> str:
    """Load a checkpoint saved by an older code revision.

    - rename legacy key ``.subPart.`` -> ``.subParts.0.`` (traj2simvec)
    - fall back to strict=False for benign missing/extra keys (neutraj
      ``lonlat_bounds`` buffer, dan unused ``unmatched_embed``).
    Real shape mismatches (e.g. trajgat data-dependent vocab) still raise.
    """
    remap = {k.replace(".subPart.", ".subParts.0."): v for k, v in sd.items()}
    try:
        model.load_state_dict(remap, strict=True)
        return "strict"
    except RuntimeError:
        res = model.load_state_dict(remap, strict=False)
        return f"nonstrict(missing={len(res.missing_keys)},unexpected={len(res.unexpected_keys)})"


def _lonlat_stack(pool_pairs, n_points, noise_std_deg, cpu):
    pres, posts = [], []
    for j, item in enumerate(pool_pairs):
        pii, poo = pair_pre_post_lonlat12(
            item,
            n_points=n_points,
            noise_std_deg=noise_std_deg,
            noise_seed_pre=NOISE_BASE + j * 3,
            noise_seed_post=NOISE_BASE + j * 3 + 1,
            device=cpu,
            noise_blend_alpha=ALPHA,
        )
        pres.append(pii)
        posts.append(poo)
    return torch.stack(pres, dim=0), torch.stack(posts, dim=0)


@torch.no_grad()
def distance_scores(metric, pre_t, post_t, dev):
    return pairwise_neg_distance_matrix(pre_t.to(dev), post_t.to(dev), metric).detach().cpu().numpy()


@torch.no_grad()
def repr_scores(encoder, pool_pairs, train_pairs, dev, cpu):
    d = registry.REPR_DEFAULTS
    model, enc, trajgat_helper, st2vec_helper = _build_repr_model(
        train_pairs,
        encoder=encoder,
        dev=dev,
        sample_seed=registry.SAMPLE_SEED,
        rnn_dim=d["rnn_dim"],
        embed_dim=d["embed_dim"],
        t3s_lstm_hidden=d["t3s_lstm_hidden"],
        t3s_tf_layers=d["t3s_tf_layers"],
        t3s_nhead=d["t3s_nhead"],
        trajgat_d_model=d["trajgat_d_model"],
        trajgat_max_items=d["trajgat_max_items"],
        trajgat_max_depth=d["trajgat_max_depth"],
        trajgat_num_layers=d["trajgat_num_layers"],
        trajgat_nhead=d["trajgat_nhead"],
        trajgat_d_lap=d["trajgat_d_lap"],
        trajgat_d_input=d["trajgat_d_input"],
        neutraj_grid_h=d["neutraj_grid_h"],
        neutraj_grid_w=d["neutraj_grid_w"],
        neutraj_recurrent=d["neutraj_recurrent"],
        trajcl_cell_size_m=d["trajcl_cell_size_m"],
        trajcl_cellspace_buffer_m=d["trajcl_cellspace_buffer_m"],
        trajcl_moco_nqueue=d["trajcl_moco_nqueue"],
        trajcl_moco_proj_dim=d["trajcl_moco_proj_dim"],
        trajcl_moco_temperature=d["trajcl_moco_temperature"],
        trajcl_aug1=d["trajcl_aug1"],
        trajcl_aug2=d["trajcl_aug2"],
        trajcl_bounds_max_pairs=d["trajcl_bounds_max_pairs"],
        trajcl_bounds_margin_deg=d["trajcl_bounds_margin_deg"],
        trajcl_max_grid_cells=d["trajcl_max_grid_cells"],
        trajcl_seed=d["trajcl_seed"],
        st2vec_mapper_k=d["st2vec_mapper_k"],
        n_points=d["n_points"],
    )
    ckpt = registry.REPR_CKPT[encoder]
    how = _tolerant_load(model, torch.load(str(ckpt), map_location=dev))
    print(f"[worker_baseline] {encoder} load: {how}", flush=True)
    model.eval()

    k = len(pool_pairs)
    samp = torch.arange(k).long()
    if enc == "trajgat":
        g_pre, g_post = graphs_for_valid_indices(
            trajgat_helper, pool_pairs, samp,
            n_points=d["n_points"], noise_std_deg=d["noise_std_deg"],
            noise_base=NOISE_BASE, device=dev, noise_blend_alpha=ALPHA,
        )
        z_pre = model.encode_pre(g_pre)
        z_post = model.encode_post(g_post)
    elif enc == "st2vec":
        mapper = st2vec_helper[0]
        max_node_idx = int(model.road.x.size(0)) - 1
        pre_ids, pre_te, post_ids, post_te = st2vec_valid_encode_batch(
            mapper, pool_pairs, samp,
            n_points=d["n_points"], noise_std_deg=d["noise_std_deg"],
            noise_base=NOISE_BASE, embed_dim=d["embed_dim"], device=dev,
            max_node_idx=max_node_idx, noise_blend_alpha=ALPHA,
        )
        z_pre = model.encode_pre(pre_ids, pre_te)
        z_post = model.encode_post(post_ids, post_te)
    else:
        pres, posts = [], []
        for j in range(k):
            pii, poo = pair_pre_post_lonlat12(
                pool_pairs[j], n_points=d["n_points"], noise_std_deg=d["noise_std_deg"],
                noise_seed_pre=NOISE_BASE + j * 3, noise_seed_post=NOISE_BASE + j * 3 + 1,
                device=cpu, noise_blend_alpha=ALPHA,
            )
            pres.append(pii)
            posts.append(poo)
        pre_t = torch.stack(pres, dim=0).to(dev)
        post_t = torch.stack(posts, dim=0).to(dev)
        z_pre = model.encode_pre(pre_t)
        z_post = model.encode_post(post_t)
    dist = torch.cdist(z_pre, z_post, p=2)
    return (-dist).detach().cpu().numpy()


@torch.no_grad()
def attnmove_scores(pool_pairs, dev, cpu):
    from cursor_baseline.attnmove_region import RegionVocab
    from cursor_baseline.completion_models import build_completion_backbone
    from cursor_baseline.completion_traj import pair_to_ab_lonlat, scores_neg_mean_step_l2

    vocab = RegionVocab.load(str(registry.ATTNMOVE_VOCAB))
    model = build_completion_backbone(
        "attnmove", rnn_hidden=128, rnn_layers=1, region_vocab=vocab,
        attn_hidden=64, attn_layers=2, attn_heads=1, attn_drop=0.3, attn_fb_drop=0.3,
        label_confidence=0.9, reg_lambda=0.001, reg_dist_lambda=1.0,
    ).to(dev)
    how = _tolerant_load(model, torch.load(str(registry.ATTNMOVE_CKPT), map_location=dev))
    print(f"[worker_baseline] attnmove load: {how}", flush=True)
    model.eval()

    al, bl = [], []
    for item in pool_pairs:
        pr = pair_to_ab_lonlat(item, cpu)
        if pr is None:
            raise RuntimeError("attnmove: pair_to_ab_lonlat returned None")
        al.append(pr[0])
        bl.append(pr[1])
    a = batch_standardize_lonlat_in_batch(torch.stack(al, 0).to(dev), enabled=True)
    b = batch_standardize_lonlat_in_batch(torch.stack(bl, 0).to(dev), enabled=True)
    with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
        pred = model(a)
    S = scores_neg_mean_step_l2(pred.float(), b)
    return S.detach().cpu().numpy()


@torch.no_grad()
def dan_scores(pool_pairs, pre_t, post_t, dev):
    dd = registry.DAN_DEFAULTS
    model = DANMotTrajSimplified(
        rnn_hidden=dd["rnn_hidden"], embed_dim=dd["embed_dim"],
        rnn_layers=dd["rnn_layers"], mlp_hidden=dd["mlp_hidden"],
    ).to(dev)
    how = _tolerant_load(model, torch.load(str(registry.DAN_CKPT), map_location=dev))
    print(f"[worker_baseline] dan load: {how}", flush=True)
    model.eval()
    pre = batch_standardize_lonlat_in_batch(pre_t.to(dev), enabled=dd["input_batch_standardize"])
    post = batch_standardize_lonlat_in_batch(post_t.to(dev), enabled=dd["input_batch_standardize"])
    with torch.cuda.amp.autocast(enabled=(dev.type == "cuda")):
        S = model(pre, post)
    return S.float().detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--methods", default="dtw,hausdorff,frechet,sspd,t3s,traj2simvec,neutraj,trajgat,st2vec,trajcl,dan,attnmove")
    args = ap.parse_args()
    if args.device:
        cfg.device = args.device
    dev = torch.device(cfg.device)
    cpu = torch.device("cpu")

    with open(args.pool_json) as f:
        pool_gidx = json.load(f)["pool_gidx"]

    all_pairs = load_pairs(registry.PAIRS_PKL)
    pool_pairs = [all_pairs[g] for g in pool_gidx]
    vg = set(valid_global_indices(len(all_pairs), registry.SAMPLE_RATIO, registry.SAMPLE_SEED))
    train_pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in vg]
    k = len(pool_pairs)
    n_points = registry.REPR_DEFAULTS["n_points"]
    noise_std_deg = registry.REPR_DEFAULTS["noise_std_deg"]

    pre_t, post_t = _lonlat_stack(pool_pairs, n_points, noise_std_deg, cpu)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    out = {}
    for m in methods:
        try:
            if m in registry.DISTANCE_METRICS:
                S = distance_scores(m, pre_t, post_t, dev)
            elif m in registry.REPR_CKPT:
                S = repr_scores(m, pool_pairs, train_pairs, dev, cpu)
            elif m == "dan":
                S = dan_scores(pool_pairs, pre_t, post_t, dev)
            elif m == "attnmove":
                S = attnmove_scores(pool_pairs, dev, cpu)
            else:
                print(f"[worker_baseline] SKIP unknown method {m}", flush=True)
                continue
            out[m] = S
            acc = float((S.argmax(axis=1) == np.arange(k)).mean())
            print(f"[worker_baseline] {m:12s} OK top1-acc={acc:.4f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[worker_baseline] {m:12s} FAILED: {e}", flush=True)
            traceback.print_exc()
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    np.savez(args.out, **out)
    print(f"[worker_baseline] wrote {args.out} methods={list(out)}", flush=True)


if __name__ == "__main__":
    main()
