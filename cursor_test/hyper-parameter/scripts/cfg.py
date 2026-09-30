"""
Config for hyper-parameter experiments (mod4-compatible, self-contained under scripts/).
"""

from __future__ import annotations

import os

import torch

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPTS)))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

diff_pre_len = 8
diff_infer_len = 4
hydra_tail = 64

diffusion_timestamp = 200 if device.startswith("cuda") else 100
beta_start = 0.0001
beta_end = 0.05
diff_lr = 1e-6
diff_ema = False
lambda_x0 = 0.0

pair_match = True
match_dim = 128
rl_use = 16
graph_between_attention = False
assignment_method = "hungarian"
conf_degree = 0.99

dim = 512
num_res_blocks = 2

pred_sample = 8
use_delta = True
input_height = False
start_dim = 4 if input_height else 3
pad_front = True
tqdm = True

pairs_pkl = os.path.join(_REPO, "cursor_mod_4", "pipeline", "all_pairs_v2_maxdtw.pkl")

_FN_DIR = os.path.join(_REPO, "cursor_mod_4", "pretrain", "file")
fn_pre = os.path.join(_FN_DIR, "model_para_diff_pre131.17892456.pth")
fn_post = os.path.join(_FN_DIR, "model_para_diff_post131.17892456.pth")

hyper_out_root = os.path.join(os.path.dirname(_SCRIPTS), "out")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


# Subprocess hyper-parameter propagation: run_tier launches separate Python
# processes (diff FT, cache/match, rl, eval). The in-process apply_hyper() in the
# parent does NOT reach them, so the per-tier values are passed via HYPER_* env
# vars and applied here at import time (children inherit the parent's environment).
diffusion_timestamp = _env_int("HYPER_T", diffusion_timestamp)
dim = _env_int("HYPER_UNET_DIM", dim)
num_res_blocks = _env_int("HYPER_RES_BLOCKS", num_res_blocks)
hydra_tail = _env_int("HYPER_HYDRA_TAIL", hydra_tail)
rl_use = _env_int("HYPER_RL_USE", rl_use)
match_dim = _env_int("HYPER_MATCH_DIM", match_dim)
lambda_x0 = _env_float("HYPER_LAMBDA_X0", lambda_x0)


def apply_hyper(
    *,
    T: int | None = None,
    unet_dim: int | None = None,
    res_blocks: int | None = None,
    lam_x0: float | None = None,
    tail: int | None = None,
    k: int | None = None,
    m_dim: int | None = None,
) -> None:
    global diffusion_timestamp, dim, num_res_blocks, lambda_x0, hydra_tail, rl_use, match_dim
    if T is not None:
        diffusion_timestamp = int(T)
    if unet_dim is not None:
        dim = int(unet_dim)
    if res_blocks is not None:
        num_res_blocks = int(res_blocks)
    if lam_x0 is not None:
        lambda_x0 = float(lam_x0)
    if tail is not None:
        hydra_tail = int(tail)
    if k is not None:
        rl_use = int(k)
    if m_dim is not None:
        match_dim = int(m_dim)
