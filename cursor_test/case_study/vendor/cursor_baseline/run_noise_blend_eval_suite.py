"""
6×4 noise–label blend × valid-pool grid; each cell = mean over ``--valid-n-passes`` (default 3).

- Alphas: ``cursor_mod_4.traj_noise_blend.NOISE_BLEND_LEVEL_ALPHAS`` (1.0 … 0.0).
- Fracs: 15%%, 10%%, 5%%, 2%% of the valid set for k×k retrieval.
- Rows: acc, hr5, hr10, hr20, r5_at_10, r10_at_20, valid_loss_ce (0 for distance baselines).

Learning encoders: load checkpoint only (no training). Distances: DTW / Fréchet / Hausdorff / SSPD.

Example::

    python cursor_baseline/run_noise_blend_eval_suite.py \\
      --pairs-pkl /path/to/shenzhen_all_pairs_v2_maxdtw.pkl \\
      --output-json cursor_baseline/results/noise_blend_grid.json \\
      --repr-ckpt-dir cursor_baseline/results \\
      --dan-ckpt cursor_baseline/results/best_dan_mot_simplified_ep10_acc0.2500.pth \\
      --eval-base-seed 42
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

sys.modules["cfg"] = importlib.import_module("cursor_baseline.model_lib.cfg")
import cfg  # noqa: E402

from cursor_baseline.dan_mot_traj.model import DANMotTrajSimplified  # noqa: E402
from cursor_baseline.lonlat_traj import stack_subsample_lonlat  # noqa: E402
from cursor_baseline.metrics import retrieval_metrics  # noqa: E402
from cursor_baseline.model_lib.sample_db import sample_db  # noqa: E402
from cursor_baseline.neutraj_twin import NeuTrajTwin, lonlat_mean_std_from_pairs  # noqa: E402
from cursor_baseline.st2vec_build import build_st2vec_road_network  # noqa: E402
from cursor_baseline.st2vec_twin import ST2VecTwin  # noqa: E402
from cursor_baseline.t3s_twin import T3STwin  # noqa: E402
from cursor_baseline.traj2simvec_twin import Traj2SimVecTwin  # noqa: E402
from cursor_baseline.traj_distances import pairwise_neg_distance_matrix  # noqa: E402
from cursor_baseline.trajgat_build import build_trajgat_index  # noqa: E402
from cursor_baseline.trajgat_twin import TrajGATTwin  # noqa: E402
from cursor_baseline.trajcl_official import (  # noqa: E402
    TrajCLOfficialBaseline,
    apply_trajcl_config,
    build_cellspace_from_lonlat_bounds,
    lonlat_bounds_from_train_pairs,
)
from cursor_baseline.train_dan_traj import dan_valid_metrics  # noqa: E402
from cursor_baseline.train_repr_baseline import repr_baseline_valid_metrics_averaged  # noqa: E402

from cursor_mod_4.traj_noise_blend import NOISE_BLEND_LEVEL_ALPHAS  # noqa: E402

DEFAULT_FRACS: tuple[float, ...] = (0.15, 0.10, 0.05, 0.02)
DIST_METRIC_SEED_OFF = {"dtw": 101, "frechet": 202, "hausdorff": 303, "sspd": 404}


def load_train_valid_from_confile(sample_ratio: float, sample_seed: int) -> tuple[list, list]:
    """Same train/valid split as ``train_repr_baseline`` / original repr ckpt training."""
    from cursor_baseline.model_lib.proc_pairs import clean_pairs, proc_multi_single_traj  # noqa: PLC0415

    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, "rb"))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    all_pairs = clean_pairs(mul_p + sing_p)
    valid = sample_db(all_pairs, ratio=float(sample_ratio), seed=int(sample_seed))
    train = [p for p in all_pairs if p not in valid]
    if not train or not valid:
        raise RuntimeError("empty train or valid after con_file split")
    return train, valid


def load_train_valid_from_pkl(pairs_pkl: str, sample_ratio: float, sample_seed: int) -> tuple[list, list]:
    raw = pickle.load(open(pairs_pkl, "rb"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"pairs_pkl must be non-empty list: {pairs_pkl}")
    valid = sample_db(raw, ratio=float(sample_ratio), seed=int(sample_seed))
    train = [p for p in raw if p not in valid]
    if not train or not valid:
        raise RuntimeError("empty train or valid after split")
    return train, valid


def _build_repr_model(
    train_pairs: list,
    *,
    encoder: str,
    dev: torch.device,
    sample_seed: int,
    rnn_dim: int,
    embed_dim: int,
    t3s_lstm_hidden: int,
    t3s_tf_layers: int,
    t3s_nhead: int,
    trajgat_d_model: int,
    trajgat_max_items: int,
    trajgat_max_depth: int,
    trajgat_num_layers: int,
    trajgat_nhead: int,
    trajgat_d_lap: int,
    trajgat_d_input: int,
    neutraj_grid_h: int,
    neutraj_grid_w: int,
    neutraj_recurrent: str,
    trajcl_cell_size_m: float,
    trajcl_cellspace_buffer_m: float,
    trajcl_moco_nqueue: int,
    trajcl_moco_proj_dim: int,
    trajcl_moco_temperature: float,
    trajcl_aug1: str,
    trajcl_aug2: str,
    trajcl_bounds_max_pairs: int,
    trajcl_bounds_margin_deg: float,
    trajcl_max_grid_cells: int,
    trajcl_seed: int | None,
    st2vec_mapper_k: int,
    n_points: int,
) -> tuple[torch.nn.Module, str, Any, tuple | None]:
    enc = encoder.lower().strip()
    trajgat_helper = None
    st2vec_helper: tuple | None = None
    if enc == "traj2simvec":
        model = Traj2SimVecTwin(rnn_dim=rnn_dim).to(dev)
        tag = "traj2simvec"
    elif enc == "t3s":
        model = T3STwin(
            embed_dim=embed_dim,
            lstm_hidden=t3s_lstm_hidden,
            tf_layers=t3s_tf_layers,
            nhead=t3s_nhead,
            max_len=max(n_points * 2, 64),
        ).to(dev)
        tag = "t3s"
    elif enc == "trajgat":
        _st, _qt, _nid, pre_emb, trajgat_helper = build_trajgat_index(
            train_pairs,
            d_model=trajgat_d_model,
            max_items=trajgat_max_items,
            max_depth=trajgat_max_depth,
            d_lap_pos=trajgat_d_lap,
        )
        model = TrajGATTwin(
            pre_emb,
            d_model=trajgat_d_model,
            d_input=trajgat_d_input,
            num_head=trajgat_nhead,
            num_encoder_layers=trajgat_num_layers,
            d_lap_pos=trajgat_d_lap,
        ).to(dev)
        tag = "trajgat"
    elif enc == "neutraj":
        m_lon, s_lon = lonlat_mean_std_from_pairs(train_pairs)
        model = NeuTrajTwin(
            target_size=rnn_dim,
            lonlat_mean=m_lon,
            lonlat_std=s_lon,
            grid_size=(neutraj_grid_h, neutraj_grid_w),
            recurrent_unit=neutraj_recurrent,
            device=dev,
        ).to(dev)
        tag = "neutraj"
    elif enc == "trajcl":
        min_lon, max_lon, min_lat, max_lat = lonlat_bounds_from_train_pairs(
            train_pairs,
            max_pairs=int(trajcl_bounds_max_pairs),
            margin_deg=float(trajcl_bounds_margin_deg),
        )
        apply_trajcl_config(
            device=dev,
            cell_embedding_dim=int(embed_dim),
            seq_embedding_dim=int(embed_dim),
            moco_proj_dim=int(trajcl_moco_proj_dim),
            moco_nqueue=int(trajcl_moco_nqueue),
            moco_temperature=float(trajcl_moco_temperature),
            cell_size_m=float(trajcl_cell_size_m),
            cellspace_buffer_m=float(trajcl_cellspace_buffer_m),
            trajcl_aug1=str(trajcl_aug1),
            trajcl_aug2=str(trajcl_aug2),
            seed=trajcl_seed if trajcl_seed is not None else int(sample_seed),
        )
        cs, _ = build_cellspace_from_lonlat_bounds(
            min_lon,
            max_lon,
            min_lat,
            max_lat,
            cell_size_m=float(trajcl_cell_size_m),
            buffer_m=float(trajcl_cellspace_buffer_m),
            max_grid_cells=int(trajcl_max_grid_cells),
        )
        model = TrajCLOfficialBaseline(
            cs,
            cell_embedding_dim=int(embed_dim),
            aug1_name=str(trajcl_aug1),
            aug2_name=str(trajcl_aug2),
        ).to(dev)
        tag = "trajcl"
    elif enc == "st2vec":
        if int(embed_dim) % 4 != 0:
            raise ValueError("ST2Vec requires embed_dim divisible by 4")
        road, mapper = build_st2vec_road_network(
            train_pairs,
            embed_dim=int(embed_dim),
            mapper_k=int(st2vec_mapper_k),
            device=dev,
        )
        model = ST2VecTwin(road, int(embed_dim), str(dev)).to(dev)
        st2vec_helper = (mapper,)
        tag = "st2vec"
    else:
        raise ValueError(f"unknown encoder {encoder!r}")
    return model, enc, trajgat_helper, st2vec_helper


def _find_repr_ckpt(ckpt_dir: Path, encoder: str) -> str | None:
    cands = sorted(ckpt_dir.glob(f"*{encoder}*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0]) if cands else None


def eval_repr_encoder_grid(
    *,
    encoder: str,
    ckpt_path: str,
    train_pairs: list,
    valid_pairs: list,
    dev: torch.device,
    cpu: torch.device,
    alphas: Sequence[float],
    fracs: Sequence[float],
    valid_n_passes: int,
    eval_base_seed: int,
    n_points: int,
    noise_std_deg: float,
    embed_dim: int,
    rnn_dim: int,
    t3s_lstm_hidden: int,
    t3s_tf_layers: int,
    t3s_nhead: int,
    trajgat_d_model: int,
    trajgat_max_items: int,
    trajgat_max_depth: int,
    trajgat_num_layers: int,
    trajgat_nhead: int,
    trajgat_d_lap: int,
    trajgat_d_input: int,
    neutraj_grid_h: int,
    neutraj_grid_w: int,
    neutraj_recurrent: str,
    trajcl_cell_size_m: float,
    trajcl_cellspace_buffer_m: float,
    trajcl_moco_nqueue: int,
    trajcl_moco_proj_dim: int,
    trajcl_moco_temperature: float,
    trajcl_aug1: str,
    trajcl_aug2: str,
    trajcl_bounds_max_pairs: int,
    trajcl_bounds_margin_deg: float,
    trajcl_max_grid_cells: int,
    trajcl_seed: int | None,
    st2vec_mapper_k: int,
    sample_seed: int,
    noise_blend_check_label_ts: bool = False,
) -> list[dict[str, Any]]:
    model, enc, th, sh = _build_repr_model(
        train_pairs,
        encoder=encoder,
        dev=dev,
        sample_seed=sample_seed,
        rnn_dim=rnn_dim,
        embed_dim=embed_dim,
        t3s_lstm_hidden=t3s_lstm_hidden,
        t3s_tf_layers=t3s_tf_layers,
        t3s_nhead=t3s_nhead,
        trajgat_d_model=trajgat_d_model,
        trajgat_max_items=trajgat_max_items,
        trajgat_max_depth=trajgat_max_depth,
        trajgat_num_layers=trajgat_num_layers,
        trajgat_nhead=trajgat_nhead,
        trajgat_d_lap=trajgat_d_lap,
        trajgat_d_input=trajgat_d_input,
        neutraj_grid_h=neutraj_grid_h,
        neutraj_grid_w=neutraj_grid_w,
        neutraj_recurrent=neutraj_recurrent,
        trajcl_cell_size_m=trajcl_cell_size_m,
        trajcl_cellspace_buffer_m=trajcl_cellspace_buffer_m,
        trajcl_moco_nqueue=trajcl_moco_nqueue,
        trajcl_moco_proj_dim=trajcl_moco_proj_dim,
        trajcl_moco_temperature=trajcl_moco_temperature,
        trajcl_aug1=trajcl_aug1,
        trajcl_aug2=trajcl_aug2,
        trajcl_bounds_max_pairs=trajcl_bounds_max_pairs,
        trajcl_bounds_margin_deg=trajcl_bounds_margin_deg,
        trajcl_max_grid_cells=trajcl_max_grid_cells,
        trajcl_seed=trajcl_seed,
        st2vec_mapper_k=st2vec_mapper_k,
        n_points=n_points,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    n_valid = len(valid_pairs)
    out: list[dict[str, Any]] = []
    cell = 0
    total = len(alphas) * len(fracs)
    for ai, alpha in enumerate(alphas):
        for fi, frac in enumerate(fracs):
            cell += 1
            base_seed = int(eval_base_seed) + ai * 10_007 + fi * 97
            vm = repr_baseline_valid_metrics_averaged(
                model,
                enc=enc,
                trajgat_helper=th,
                st2vec_helper=sh,
                valid_pairs=valid_pairs,
                n_valid=n_valid,
                valid_sample_frac=float(frac),
                valid_n_passes=int(valid_n_passes),
                base_seed=base_seed,
                n_points=n_points,
                noise_std_deg=noise_std_deg,
                embed_dim=embed_dim,
                dev=dev,
                cpu=cpu,
                noise_blend_alpha=float(alpha),
                noise_blend_check_label_ts=noise_blend_check_label_ts,
            )
            row = {
                "method": f"repr_{encoder}",
                "encoder": encoder,
                "ckpt": ckpt_path,
                "noise_blend_alpha": float(alpha),
                "valid_sample_frac": float(frac),
                "valid_n_passes": int(valid_n_passes),
                "eval_base_seed": int(eval_base_seed),
                "cell_base_seed": int(base_seed),
                **{k: float(vm[k]) for k in ("acc", "hr5", "hr10", "hr20", "r5_at_10", "r10_at_20", "valid_loss_ce", "k", "n_valid")},
            }
            out.append(row)
            print(
                f"[noise_blend_grid] {encoder} α={alpha:g} frac={frac:g} "
                f"acc={row['acc']:.4f} HR5={row['hr5']:.4f} HR10={row['hr10']:.4f} HR20={row['hr20']:.4f} "
                f"R5@10={row['r5_at_10']:.4f} R10@20={row['r10_at_20']:.4f} CE={row['valid_loss_ce']:.4f} "
                f"k={int(row['k'])}/{int(row['n_valid'])} ({cell}/{total})",
                flush=True,
            )
    return out


def eval_dan_grid(
    *,
    ckpt_path: str,
    valid_pairs: list,
    dev: torch.device,
    cpu: torch.device,
    alphas: Sequence[float],
    fracs: Sequence[float],
    valid_n_passes: int,
    eval_base_seed: int,
    n_points: int,
    noise_std_deg: float,
    rnn_hidden: int,
    embed_dim: int,
    rnn_layers: int,
    mlp_hidden: int,
    input_batch_standardize: bool,
    noise_blend_check_label_ts: bool = False,
) -> list[dict[str, Any]]:
    model = DANMotTrajSimplified(
        rnn_hidden=rnn_hidden,
        embed_dim=embed_dim,
        rnn_layers=rnn_layers,
        mlp_hidden=mlp_hidden,
    ).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    use_cuda = dev.type == "cuda"
    n_valid = len(valid_pairs)
    out: list[dict[str, Any]] = []
    cell = 0
    total = len(alphas) * len(fracs)
    for ai, alpha in enumerate(alphas):
        for fi, frac in enumerate(fracs):
            cell += 1
            base_seed = int(eval_base_seed) + ai * 10_007 + fi * 97
            acc, hr5, hr10, hr20, r5_at_10, r10_at_20, valid_loss, k = dan_valid_metrics(
                model,
                valid_pairs,
                dev=dev,
                cpu=cpu,
                use_cuda=use_cuda,
                n_valid=n_valid,
                valid_sample_frac=float(frac),
                valid_n_passes=int(valid_n_passes),
                base_seed=base_seed,
                n_points=n_points,
                noise_std_deg=noise_std_deg,
                input_batch_standardize=input_batch_standardize,
                noise_blend_alpha=float(alpha),
                noise_blend_check_label_ts=noise_blend_check_label_ts,
            )
            row = {
                "method": "dan_mot_simplified",
                "ckpt": ckpt_path,
                "noise_blend_alpha": float(alpha),
                "valid_sample_frac": float(frac),
                "valid_n_passes": int(valid_n_passes),
                "eval_base_seed": int(eval_base_seed),
                "cell_base_seed": int(base_seed),
                "acc": float(acc),
                "hr5": float(hr5),
                "hr10": float(hr10),
                "hr20": float(hr20),
                "r5_at_10": float(r5_at_10),
                "r10_at_20": float(r10_at_20),
                "valid_loss_ce": float(valid_loss),
                "k": float(k),
                "n_valid": float(n_valid),
            }
            out.append(row)
            print(
                f"[noise_blend_grid] DAN α={alpha:g} frac={frac:g} "
                f"acc={acc:.4f} HR5={hr5:.4f} HR10={hr10:.4f} HR20={hr20:.4f} "
                f"R5@10={r5_at_10:.4f} R10@20={r10_at_20:.4f} CE={valid_loss:.4f} k={k}/{n_valid} ({cell}/{total})",
                flush=True,
            )
    return out


def eval_distance_metric_grid(
    *,
    metric: str,
    valid_pairs: list,
    device: torch.device,
    alphas: Sequence[float],
    fracs: Sequence[float],
    valid_n_passes: int,
    eval_base_seed: int,
    n_points: int,
    noise_std_deg: float,
    chunk_queries: int,
    noise_blend_check_label_ts: bool = False,
) -> list[dict[str, Any]]:
    n_valid = len(valid_pairs)
    out: list[dict[str, Any]] = []
    cell = 0
    total = len(alphas) * len(fracs)
    off = int(DIST_METRIC_SEED_OFF.get(str(metric), 0))
    for ai, alpha in enumerate(alphas):
        for fi, frac in enumerate(fracs):
            cell += 1
            base_sub = int(eval_base_seed) + off + ai * 10_007 + fi * 97
            k = max(1, min(n_valid, int(round(float(n_valid) * float(frac)))))
            acc_s = hr5_s = hr10_s = hr20_s = r51_s = r102_s = 0.0
            np = max(1, int(valid_n_passes))
            for rep in range(np):
                g_cpu = torch.Generator(device="cpu")
                g_cpu.manual_seed(int(base_sub) + rep * 7919)
                samp = torch.randperm(n_valid, generator=g_cpu)[:k].long()
                noise_base = int(base_sub) * 10007 + rep * 1_000_003 + k * 17
                pre_t, post_t = stack_subsample_lonlat(
                    valid_pairs,
                    samp,
                    n_points=n_points,
                    noise_std_deg=noise_std_deg,
                    noise_seed_base=noise_base,
                    device=device,
                    noise_blend_alpha=float(alpha),
                    noise_blend_check_label_ts=noise_blend_check_label_ts,
                )
                scores = pairwise_neg_distance_matrix(
                    pre_t,
                    post_t,
                    str(metric),
                    chunk_queries=int(chunk_queries),
                )
                m = retrieval_metrics(scores)
                acc_s += m["acc"]
                hr5_s += m["hr5"]
                hr10_s += m["hr10"]
                hr20_s += m["hr20"]
                r51_s += m["r5_at_10"]
                r102_s += m["r10_at_20"]
            row = {
                "method": f"metric_{metric}",
                "metric": metric,
                "noise_blend_alpha": float(alpha),
                "valid_sample_frac": float(frac),
                "valid_n_passes": int(valid_n_passes),
                "eval_base_seed": int(eval_base_seed),
                "cell_base_sub": int(base_sub),
                "acc": acc_s / np,
                "hr5": hr5_s / np,
                "hr10": hr10_s / np,
                "hr20": hr20_s / np,
                "r5_at_10": r51_s / np,
                "r10_at_20": r102_s / np,
                "valid_loss_ce": 0.0,
                "k": float(k),
                "n_valid": float(n_valid),
            }
            out.append(row)
            print(
                f"[noise_blend_grid] {metric} α={alpha:g} frac={frac:g} "
                f"acc={row['acc']:.4f} HR5={row['hr5']:.4f} HR10={row['hr10']:.4f} HR20={row['hr20']:.4f} "
                f"R5@10={row['r5_at_10']:.4f} R10@20={row['r10_at_20']:.4f} k={k}/{n_valid} ({cell}/{total})",
                flush=True,
            )
    return out


def parse_args():
    p = argparse.ArgumentParser(description="6×4 noise-blend × valid-frac grid (mean over N passes).")
    p.add_argument("--pairs-pkl", type=str, default=None, help="Used when --pairs-source pkl (7-tuple list pickle).")
    p.add_argument(
        "--pairs-source",
        type=str,
        default="pkl",
        choices=("pkl", "confile"),
        help="pkl: --pairs-pkl (e.g. v2 maxdtw). confile: same as train_repr_baseline (cfg.con_file); required for TrajGAT ckpts trained on that pipeline.",
    )
    p.add_argument("--output-json", type=str, required=True)
    p.add_argument("--sample-ratio", type=float, default=0.1)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--valid-n-passes", type=int, default=3)
    p.add_argument("--eval-base-seed", type=int, default=42)
    p.add_argument("--n-points", type=int, default=12)
    p.add_argument("--noise-std-deg", type=float, default=5e-5)
    p.add_argument(
        "--alphas",
        type=str,
        default=None,
        help="Comma-separated noise_blend_alpha values (default: all NOISE_BLEND_LEVEL_ALPHAS).",
    )
    p.add_argument(
        "--distance-metrics",
        type=str,
        default="dtw,frechet,hausdorff,sspd",
        help="Comma-separated distance baselines when not --skip-distances.",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--repr-ckpt-dir", type=str, default=None)
    p.add_argument("--repr-ckpt-traj2simvec", type=str, default=None)
    p.add_argument("--repr-ckpt-neutraj", type=str, default=None)
    p.add_argument("--repr-ckpt-trajgat", type=str, default=None)
    p.add_argument("--repr-ckpt-trajcl", type=str, default=None)
    p.add_argument("--repr-ckpt-t3s", type=str, default=None)
    p.add_argument("--repr-ckpt-st2vec", type=str, default=None)
    p.add_argument(
        "--repr-encoders",
        type=str,
        default="traj2simvec,neutraj,trajgat,trajcl",
        help="Comma-separated: traj2simvec,neutraj,trajgat,trajcl,t3s,st2vec (must match checkpoints provided).",
    )
    p.add_argument("--dan-ckpt", type=str, default=None)
    p.add_argument("--dan-rnn-hidden", type=int, default=128)
    p.add_argument("--dan-embed-dim", type=int, default=128)
    p.add_argument("--dan-rnn-layers", type=int, default=1)
    p.add_argument("--dan-mlp-hidden", type=int, default=256)
    p.add_argument("--no-input-batch-standardize", action="store_true")
    p.add_argument("--skip-repr", action="store_true")
    p.add_argument("--skip-dan", action="store_true")
    p.add_argument("--skip-distances", action="store_true")
    p.add_argument("--chunk-queries", type=int, default=32)
    p.add_argument(
        "--blend-check-label-ts",
        action="store_true",
        help="Also assert L_A/L_B timestamps match A/B (v1-style; v2 max-DTW usually fails).",
    )
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--rnn-dim", type=int, default=128)
    p.add_argument("--t3s-lstm-hidden", type=int, default=64)
    p.add_argument("--t3s-tf-layers", type=int, default=2)
    p.add_argument("--t3s-nhead", type=int, default=8)
    p.add_argument("--trajgat-d-model", type=int, default=128)
    p.add_argument("--trajgat-max-items", type=int, default=50)
    p.add_argument("--trajgat-max-depth", type=int, default=50)
    p.add_argument("--trajgat-num-layers", type=int, default=3)
    p.add_argument("--trajgat-nhead", type=int, default=8)
    p.add_argument("--trajgat-d-lap", type=int, default=8)
    p.add_argument("--trajgat-d-input", type=int, default=4)
    p.add_argument("--neutraj-grid-h", type=int, default=500)
    p.add_argument("--neutraj-grid-w", type=int, default=500)
    p.add_argument("--neutraj-recurrent", type=str, default="LSTM", choices=("LSTM", "GRU"))
    p.add_argument("--trajcl-cell-size", type=float, default=100.0)
    p.add_argument("--trajcl-cellspace-buffer", type=float, default=500.0)
    p.add_argument("--trajcl-moco-nqueue", type=int, default=2048)
    p.add_argument("--trajcl-moco-proj-dim", type=int, default=128)
    p.add_argument("--trajcl-moco-temperature", type=float, default=0.05)
    p.add_argument("--trajcl-aug1", type=str, default="mask")
    p.add_argument("--trajcl-aug2", type=str, default="subset")
    p.add_argument("--trajcl-bounds-max-pairs", type=int, default=4000)
    p.add_argument("--trajcl-bounds-margin-deg", type=float, default=0.02)
    p.add_argument("--trajcl-max-grid-cells", type=int, default=500_000)
    p.add_argument("--trajcl-seed", type=int, default=None)
    p.add_argument("--st2vec-mapper-k", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is not None:
        cfg.device = args.device
    dev = torch.device(cfg.device)
    cpu = torch.device("cpu")

    if str(args.pairs_source).lower() == "confile":
        train_pairs, valid_pairs = load_train_valid_from_confile(float(args.sample_ratio), int(args.sample_seed))
        pairs_meta = {"pairs_source": "confile", "con_file": os.path.abspath(str(cfg.con_file))}
    else:
        if not args.pairs_pkl:
            raise SystemExit("--pairs-pkl is required when --pairs-source pkl")
        train_pairs, valid_pairs = load_train_valid_from_pkl(
            str(args.pairs_pkl),
            float(args.sample_ratio),
            int(args.sample_seed),
        )
        pairs_meta = {"pairs_source": "pkl", "pairs_pkl": os.path.abspath(str(args.pairs_pkl))}
    if args.alphas:
        alphas = [float(x.strip()) for x in str(args.alphas).split(",") if x.strip()]
    else:
        alphas = list(NOISE_BLEND_LEVEL_ALPHAS)
    fracs = list(DEFAULT_FRACS)
    dist_metrics = [m.strip().lower() for m in str(args.distance_metrics).split(",") if m.strip()]
    ibs = not bool(args.no_input_batch_standardize)

    rows: list[dict[str, Any]] = []
    meta = {
        **pairs_meta,
        "sample_ratio": float(args.sample_ratio),
        "sample_seed": int(args.sample_seed),
        "n_train": len(train_pairs),
        "n_valid": len(valid_pairs),
        "alphas": alphas,
        "fracs": fracs,
        "valid_n_passes": int(args.valid_n_passes),
        "eval_base_seed": int(args.eval_base_seed),
        "n_points": int(args.n_points),
        "noise_std_deg": float(args.noise_std_deg),
        "input_batch_standardize_dan": ibs,
        "noise_blend_check_label_ts": bool(args.blend_check_label_ts),
    }

    ckpt_dir = Path(args.repr_ckpt_dir) if args.repr_ckpt_dir else None

    CKPT_ATTR = {
        "traj2simvec": "repr_ckpt_traj2simvec",
        "neutraj": "repr_ckpt_neutraj",
        "trajgat": "repr_ckpt_trajgat",
        "trajcl": "repr_ckpt_trajcl",
        "t3s": "repr_ckpt_t3s",
        "st2vec": "repr_ckpt_st2vec",
    }

    def _ckpt(enc: str) -> str | None:
        attr = CKPT_ATTR.get(enc)
        if not attr:
            return None
        explicit = getattr(args, attr, None)
        if explicit:
            return str(explicit)
        if ckpt_dir and ckpt_dir.is_dir():
            p = _find_repr_ckpt(ckpt_dir, enc)
            if p:
                return p
        return None

    repr_kw = dict(
        train_pairs=train_pairs,
        valid_pairs=valid_pairs,
        dev=dev,
        cpu=cpu,
        alphas=alphas,
        fracs=fracs,
        valid_n_passes=int(args.valid_n_passes),
        eval_base_seed=int(args.eval_base_seed),
        n_points=int(args.n_points),
        noise_std_deg=float(args.noise_std_deg),
        embed_dim=int(args.embed_dim),
        rnn_dim=int(args.rnn_dim),
        t3s_lstm_hidden=int(args.t3s_lstm_hidden),
        t3s_tf_layers=int(args.t3s_tf_layers),
        t3s_nhead=int(args.t3s_nhead),
        trajgat_d_model=int(args.trajgat_d_model),
        trajgat_max_items=int(args.trajgat_max_items),
        trajgat_max_depth=int(args.trajgat_max_depth),
        trajgat_num_layers=int(args.trajgat_num_layers),
        trajgat_nhead=int(args.trajgat_nhead),
        trajgat_d_lap=int(args.trajgat_d_lap),
        trajgat_d_input=int(args.trajgat_d_input),
        neutraj_grid_h=int(args.neutraj_grid_h),
        neutraj_grid_w=int(args.neutraj_grid_w),
        neutraj_recurrent=str(args.neutraj_recurrent),
        trajcl_cell_size_m=float(args.trajcl_cell_size),
        trajcl_cellspace_buffer_m=float(args.trajcl_cellspace_buffer),
        trajcl_moco_nqueue=int(args.trajcl_moco_nqueue),
        trajcl_moco_proj_dim=int(args.trajcl_moco_proj_dim),
        trajcl_moco_temperature=float(args.trajcl_moco_temperature),
        trajcl_aug1=str(args.trajcl_aug1),
        trajcl_aug2=str(args.trajcl_aug2),
        trajcl_bounds_max_pairs=int(args.trajcl_bounds_max_pairs),
        trajcl_bounds_margin_deg=float(args.trajcl_bounds_margin_deg),
        trajcl_max_grid_cells=int(args.trajcl_max_grid_cells),
        trajcl_seed=args.trajcl_seed,
        st2vec_mapper_k=int(args.st2vec_mapper_k),
        sample_seed=int(args.sample_seed),
        noise_blend_check_label_ts=bool(args.blend_check_label_ts),
    )

    if not args.skip_repr:
        enc_list = [x.strip().lower() for x in str(args.repr_encoders).split(",") if x.strip()]
        for enc in enc_list:
            if enc not in CKPT_ATTR:
                print(f"[noise_blend_grid] skip unknown encoder name={enc!r}", flush=True)
                continue
            ck = _ckpt(enc)
            if not ck or not os.path.isfile(ck):
                print(f"[noise_blend_grid] skip repr encoder={enc} (no ckpt)", flush=True)
                continue
            rows.extend(eval_repr_encoder_grid(encoder=enc, ckpt_path=ck, **repr_kw))

    if not args.skip_dan and args.dan_ckpt and os.path.isfile(str(args.dan_ckpt)):
        rows.extend(
            eval_dan_grid(
                ckpt_path=str(args.dan_ckpt),
                valid_pairs=valid_pairs,
                dev=dev,
                cpu=cpu,
                alphas=alphas,
                fracs=fracs,
                valid_n_passes=int(args.valid_n_passes),
                eval_base_seed=int(args.eval_base_seed) + 500_000,
                n_points=int(args.n_points),
                noise_std_deg=float(args.noise_std_deg),
                rnn_hidden=int(args.dan_rnn_hidden),
                embed_dim=int(args.dan_embed_dim),
                rnn_layers=int(args.dan_rnn_layers),
                mlp_hidden=int(args.dan_mlp_hidden),
                input_batch_standardize=ibs,
                noise_blend_check_label_ts=bool(args.blend_check_label_ts),
            )
        )
    elif not args.skip_dan:
        print("[noise_blend_grid] skip DAN (--dan-ckpt missing)", flush=True)

    if not args.skip_distances:
        for metric in dist_metrics:
            rows.extend(
                eval_distance_metric_grid(
                    metric=metric,
                    valid_pairs=valid_pairs,
                    device=dev,
                    alphas=alphas,
                    fracs=fracs,
                    valid_n_passes=int(args.valid_n_passes),
                    eval_base_seed=int(args.eval_base_seed) + 700_000,
                    n_points=int(args.n_points),
                    noise_std_deg=float(args.noise_std_deg),
                    chunk_queries=int(args.chunk_queries),
                    noise_blend_check_label_ts=bool(args.blend_check_label_ts),
                )
            )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"[noise_blend_grid] wrote {out_path} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
