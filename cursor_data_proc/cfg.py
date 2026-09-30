"""
Config for cursor_mod_4 (diff_pre_len=8, diff_infer_len=4) meters diffusion FT.
"""

from __future__ import annotations

import os

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

diff_pre_len = 8
diff_infer_len = 4

# diffusion sampling candidates (used by some legacy modules during import)
hydra_tail = 64

diffusion_timestamp = 200 if device.startswith("cuda") else 100
beta_start = 0.0001
beta_end = 0.05
diff_lr = 1e-6
diff_ema = False
lambda_x0 = 0.0

# match defaults (needed by legacy modules that import cfg)
pair_match = True
match_dim = 128
rl_use = 16
graph_between_attention = False
assignment_method = "hungarian"
conf_degree = 0.99

# base channels used by our UNet84
dim = 512

# LSTM trajectory predictor (diffusion/predict/lstm.TrajPredHandler default sample count)
pred_sample = 8

# flags needed by proc imports
use_delta = True
input_height = False
start_dim = 4 if input_height else 3
pad_front = True
tqdm = True

# Data source for dumping pairs (v1) in cursor_data_proc.
# Keep logic identical to cursor_mod_4; use the same relative path under repo root,
# but we will *copy* the pickle into cursor_data_proc/pipeline/traj_cmb.pkl.
con_file = os.path.join(_REPO, "cursor_data_proc", "pipeline", "traj_cmb.pkl")

_FN_DIR = os.path.join(_REPO, "pretrain", "file")
fn_pre = os.path.join(_FN_DIR, "model_para_diff_pre131.17892456.pth")
fn_post = os.path.join(_FN_DIR, "model_para_diff_post131.17892456.pth")

