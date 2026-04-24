"""
Vendored config for cursor_baseline (same fields as former cursor_mod_4.cfg).
Repo root = parents[2] of this file (…/cursor_baseline/model_lib/cfg.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

_REPO = str(Path(__file__).resolve().parents[2])

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

pred_sample = 8

use_delta = True
input_height = False
start_dim = 4 if input_height else 3
pad_front = True
tqdm = True

con_file = os.path.join(_REPO, "pipeline", "traj_cmb.pkl")

_FN_DIR = os.path.join(_REPO, "pretrain", "file")
fn_pre = os.path.join(_FN_DIR, "model_para_diff_pre131.17892456.pth")
fn_post = os.path.join(_FN_DIR, "model_para_diff_post131.17892456.pth")
