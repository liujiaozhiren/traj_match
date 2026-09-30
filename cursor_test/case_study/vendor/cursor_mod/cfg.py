"""
Cursor-mod config for Shenzhen pair diffusion FT (meters pipeline).

This module is designed to be injected as `cfg`:
  sys.modules["cfg"] = importlib.import_module("cursor_mod.cfg")
"""

from __future__ import annotations

import os

import torch

# device used by training scripts
device = "cuda:0" if torch.cuda.is_available() else "cpu"

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- diffusion shapes ---
diff_pre_len = 10
diff_infer_len = 2

# --- diffusion schedule ---
diffusion_timestamp = 200 if device.startswith("cuda") else 100
beta_start = 0.0001
beta_end = 0.05
diff_lr = 1e-6
diff_ema = False
lambda_x0 = 0.0

# AssistTrajUnetModel_10_2 uses cfg.dim as base channels
dim = 512

# --- match/pair pipeline flags (needed for imports) ---
use_delta = True
input_height = False
start_dim = 4 if input_height else 3
pad_front = True
tqdm = True

# --- file paths ---
con_file = os.path.join(_REPO, "pipeline", "traj_cmb.pkl")

_FN_DIR = os.path.join(_REPO, "pretrain", "file")
fn_pre = os.path.join(_FN_DIR, "model_para_diff_pre131.17892456.pth")
fn_post = os.path.join(_FN_DIR, "model_para_diff_post131.17892456.pth")

