"""Paths to data + trained checkpoints used by the case study.

Code is fully self-contained under ``vendor/``; checkpoints and the pairs pkl are
DATA artifacts referenced here by absolute path (not copied).
"""

from __future__ import annotations

import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]

# ---- data ----
PAIRS_PKL = REPO / "cursor_mod_4" / "pipeline" / "all_pairs_v2_maxdtw.pkl"

# ---- mod4 (full pipeline: precomputed hydra cache + match + RL selector) ----
MOD4 = {
    "cache": REPO / "cursor_mod_runs" / "match_ft_84" / "hydra_only_cache_84.pt",
    "match": REPO / "cursor_mod_runs" / "match_ft_84" / "best_match_s4_ep41_acc0.1590.pth",
    "selector": REPO / "cursor_mod_runs" / "rl_tail_select_s4_84" / "best_selector_k16_acc0.1498_ep20.pt",
    "match_scheme": 4,
    "hydra_tail": 64,
}

# ---- learned baselines: best (highest-acc) checkpoint per method ----
_RES = REPO / "cursor_baseline" / "results"


def _best_ckpt(run_dir: pathlib.Path, tag: str):
    """Highest-acc ``best_<tag>_ep*_acc*.pth`` in a run dir (acc encoded in name)."""
    run_dir = pathlib.Path(run_dir)
    cands = sorted(run_dir.glob(f"best_{tag}_ep*_acc*.pth"))
    if not cands:
        return None

    def _acc(p):
        try:
            return float(p.stem.split("acc")[-1])
        except ValueError:
            return -1.0

    return max(cands, key=_acc)


# trajgat / trajcl are (re)trained on the SAME v2-pkl split the case study uses
# (see train_missing_baselines.py) so their checkpoints load cleanly here.
REPR_CKPT = {
    "t3s": _RES / "t3s_run1" / "best_t3s_ep1_acc0.0214.pth",
    "traj2simvec": _RES / "traj2simvec_run1" / "best_traj2simvec_ep18_acc0.0183.pth",
    "neutraj": _RES / "neutraj_run1" / "best_neutraj_ep22_acc0.0245.pth",
    "st2vec": _RES / "st2vec_run1" / "best_st2vec_ep40_acc0.0183.pth",
    "trajgat": _best_ckpt(_RES / "trajgat_v2pkl_run", "trajgat"),
    "trajcl": _best_ckpt(_RES / "trajcl_v2pkl_run", "trajcl"),
}
DAN_CKPT = _RES / "shahe_out_shenzhen_dan" / "best_dan_mot_simplified_ep1_acc0.0765.pth"
ATTNMOVE_CKPT = _RES / "completion_attnmove_shenzhen_run1" / "best_completion_attnmove_ep1_acc0.0092.pth"
ATTNMOVE_VOCAB = _RES / "completion_attnmove_shenzhen_run1" / "attnmove_region_vocab.json"

# parameter-free distance baselines (no checkpoint)
DISTANCE_METRICS = ("dtw", "hausdorff", "frechet", "sspd")

# repr encoder architecture defaults (must match how the checkpoints were trained;
# copied from cursor_baseline/run_noise_blend_eval_suite.py argparse defaults)
REPR_DEFAULTS = dict(
    n_points=12,
    noise_std_deg=5e-5,
    embed_dim=128,
    rnn_dim=128,
    t3s_lstm_hidden=64,
    t3s_tf_layers=2,
    t3s_nhead=8,
    trajgat_d_model=128,
    trajgat_max_items=50,
    trajgat_max_depth=50,
    trajgat_num_layers=3,
    trajgat_nhead=8,
    trajgat_d_lap=8,
    trajgat_d_input=4,
    neutraj_grid_h=500,
    neutraj_grid_w=500,
    neutraj_recurrent="LSTM",
    trajcl_cell_size_m=100.0,
    trajcl_cellspace_buffer_m=500.0,
    trajcl_moco_nqueue=2048,
    trajcl_moco_proj_dim=128,
    trajcl_moco_temperature=0.05,
    trajcl_aug1="mask",
    trajcl_aug2="subset",
    trajcl_bounds_max_pairs=4000,
    trajcl_bounds_margin_deg=0.02,
    trajcl_max_grid_cells=500_000,
    trajcl_seed=None,
    st2vec_mapper_k=6,
)
DAN_DEFAULTS = dict(rnn_hidden=128, embed_dim=128, rnn_layers=1, mlp_hidden=256, input_batch_standardize=True)

# train/valid split (shared by both stacks; same seed => identical valid set)
SAMPLE_RATIO = 0.1
SAMPLE_SEED = 42


def check_exists() -> list[str]:
    missing = []
    core = [PAIRS_PKL, MOD4["cache"], MOD4["match"], MOD4["selector"], DAN_CKPT]
    for k, v in REPR_CKPT.items():
        if v is None:
            missing.append(f"{k}: <not trained yet>")
        else:
            core.append(v)
    for p in core:
        if not pathlib.Path(p).exists():
            missing.append(str(p))
    return missing


if __name__ == "__main__":
    miss = check_exists()
    if miss:
        print("MISSING:")
        for m in miss:
            print("  ", m)
    else:
        print("all core checkpoints present")
    print("attnmove ckpt exists:", ATTNMOVE_CKPT.exists(), ATTNMOVE_CKPT)
