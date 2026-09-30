"""Shared hyper-parameters for fair cross-method comparison (all learning methods dim=128)."""

from __future__ import annotations

EMBED_DIM = 128
RNN_HIDDEN = 128
MLP_HIDDEN = 256
N_HEADS = 8
N_POINTS = 12
INFO_LEN = 8
HYDRA_TAIL = 6            # diffusion 采样 tail 条数
HYDRA_STEP_LEN = 4
K_SELECT_VANILLA = 6      # vanilla match 使用全部 tail
K_SELECT_RL = 3           # RL selector 从 HYDRA_TAIL 里选 k 条
DIFFUSION_STEPS = 4       # DDPM 去噪步数
DIFFUSION_CH = 128
DEFAULT_GALLERY_SIZE = 4000
DEFAULT_GALLERY_BATCH = 4000

# Full baseline stacks (cursor_baseline); depth aligned across comparable methods.
DEEP_LAYERS = 6
TRAJ2SIMVEC_LSTM_LAYERS = 24       # 12×2
TRAJ2SIMVEC_SUBPART_BLOCKS = 12    # 6×2
NEUTRAJ_GRID = (5000, 5000)
NEUTRAJ_RECURRENT = "LSTM"
NEUTRAJ_SAM_LAYERS = 2
NEUTRAJ_STARD_LSTM = False         # SAM_LSTMCell + grid indices
DAN_RNN_LAYERS = 24
TAT_RNN_LAYERS = 24
DAN_USE_UNMATCHED_DIM = True       # (B+1)×(B+1) affinity at train; (2)×(N+1) at 1-vs-N retrieve
TRAJGAT_NUM_LAYERS = DEEP_LAYERS
TRAJCL_TRANS_LAYERS = DEEP_LAYERS
ST2VEC_NUM_LAYERS = DEEP_LAYERS
T3S_TF_LAYERS = DEEP_LAYERS
ATTNMOVE_LAYERS = 12
ATTNMOVE_HIDDEN = EMBED_DIM
ENCODE_BATCH = 32
NOISE_STD_DEG = 0.0

# 13 baselines + cursor_mod_4 (paper table order)
ALL_14_METHODS = (
    "dtw",
    "hausdorff",
    "frechet",
    "sspd",
    "t3s",
    "traj2simvec",
    "neutraj",
    "trajgat",
    "trajcl",
    "st2vec",
    "dan",       # Deep Affinity Network
    "tat",       # Tracklet Association Tracker
    "attnmove",  # AttnMove
    "mod4",      # cursor_mod_4
)

METHOD_NAMES = ALL_14_METHODS + ("tat_graph", "mod4_rl")  # optional extras
