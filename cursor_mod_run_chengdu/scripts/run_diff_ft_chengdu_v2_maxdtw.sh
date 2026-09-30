#!/usr/bin/env bash
# Retrain 8/4 diffusion FT on Chengdu all_pairs_v2_maxdtw.pkl (same recipe as diff_ft_v2_from_pretrained/run_config.json),
# but with a longer early-stop patience on valid RMSE (default 40 vs previous 20).

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CONDA_ENV="${CONDA_ENV:-traj_match}"
if [[ -n "${PYTHON:-}" ]]; then
  PY=("$PYTHON")
elif command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
  echo "[run_diff_ft_chengdu] WARNING: conda not found; falling back to python3" >&2
  PY=(python3)
fi

OUT="${DIFF_FT_OUT_DIR:-$REPO/cursor_mod_run_chengdu/ckpt/diff_ft_v2_es40}"
PAIRS="${PAIRS_PKL:-$REPO/cursor_mod_run_chengdu/data/all_pairs_v2_maxdtw.pkl}"
PRE="${CKPT_PRE:-$REPO/cursor_mod_runs/diff_ft_pairs131_meters_84/best_pre_ep289_rmse5.908m.pth}"
POST="${CKPT_POST:-$REPO/cursor_mod_runs/diff_ft_pairs131_meters_84/best_post_ep242_rmse4.678m.pth}"

for f in "$PAIRS" "$PRE" "$POST"; do
  if [[ ! -f "$f" ]]; then
    echo "[run_diff_ft_chengdu] missing file: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT"

# Aligned with cursor_mod_run_chengdu/ckpt/diff_ft_v2_from_pretrained/run_config.json
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-512}"
EVAL_BATCH="${EVAL_BATCH:-512}"
LR="${LR:-1e-6}"
MAX_PAIRS="${MAX_PAIRS:-20000}"
# Match `run_ft_match_84`: same 9:1 split on the [:max_pairs] prefix (diff valid == match valid).
SAMPLE_RATIO="${SAMPLE_RATIO:-0.1}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
# traj_cmb-only split (ignored when --pairs-pkl is set)
VALID_RATIO="${VALID_RATIO:-0.05}"
SEED="${SEED:-20250803}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-40}"
DEVICE="${DEVICE:-cuda:0}"

exec "${PY[@]}" "$REPO/cursor_mod_4/entrypoints/ft_diffusion_on_match_pairs.py" \
  --out-dir "$OUT" \
  --pairs-pkl "$PAIRS" \
  --ckpt-pre "$PRE" \
  --ckpt-post "$POST" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --eval-batch "$EVAL_BATCH" \
  --lr "$LR" \
  --max-pairs "$MAX_PAIRS" \
  --sample-ratio "$SAMPLE_RATIO" \
  --sample-seed "$SAMPLE_SEED" \
  --valid-ratio "$VALID_RATIO" \
  --seed "$SEED" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --device "$DEVICE" \
  "$@"
