"""Train the two missing encoders (trajGAT, trajCL) on the SAME pairs/split the
case study uses, so their checkpoints load cleanly here.

Why this exists: the repo's ``train_repr_baseline`` only reads ``cfg.con_file``
(``traj_cmb.pkl``, which is missing), and the old trajGAT ckpt was built on that
pipeline -> its id-embedding vocab (28642) does not match the v2-pkl build
(44858). We reuse the real training loop but monkeypatch ``_load_pairs`` to use
``all_pairs_v2_maxdtw.pkl`` with the project split (ratio 0.1, seed 42), and the
SAME ascending train order as ``worker_baseline`` (so the trajGAT cell->id map is
identical at inference).

Outputs go to ``cursor_baseline/results/<enc>_v2pkl_run/`` (weights = data).
"""

from __future__ import annotations

import argparse
import importlib
import sys

import _bootstrap  # noqa: F401

sys.modules["cfg"] = importlib.import_module("cursor_baseline.model_lib.cfg")
import cfg  # noqa: E402

import registry  # noqa: E402
from common import load_pairs, valid_global_indices  # noqa: E402

import cursor_baseline.train_repr_baseline as T  # noqa: E402


def _patched_load_pairs(sample_ratio, sample_seed):
    ap = load_pairs(registry.PAIRS_PKL)
    vg = set(valid_global_indices(len(ap), sample_ratio, sample_seed))
    # ascending order, identical to worker_baseline's train_pairs
    train = [ap[i] for i in range(len(ap)) if i not in vg]
    valid = [ap[i] for i in range(len(ap)) if i in vg]
    if not train or not valid:
        raise RuntimeError("empty split")
    return train, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=["trajgat", "trajcl"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--early-stop", type=int, default=6)
    ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    cfg.device = args.device
    T._load_pairs = _patched_load_pairs  # noqa: SLF001

    d = registry.REPR_DEFAULTS
    res = T.run_repr_baseline(
        output_dir=args.output_dir,
        train_batch=args.train_batch,
        valid_batch=64,
        lr=args.lr,
        max_epochs=args.max_epochs,
        early_stop=args.early_stop,
        sample_ratio=registry.SAMPLE_RATIO,
        sample_seed=registry.SAMPLE_SEED,
        encoder=args.encoder,
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
        noise_std_deg=d["noise_std_deg"],
        valid_sample_frac=0.15,
        valid_n_passes=3,
    )
    print(f"[train_missing] {args.encoder} best_ckpt={res.best_ckpt} acc={res.best_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
