#!/usr/bin/env bash
# Chengdu match FT (hydra cache + scheme 4) using v2 maxdtw pairs and diff_ft_v2_from_pretrained checkpoints.
# Post-hoc eval (after training) includes valid_sample_frac in {0.15, 0.10, 0.05, 0.02}; training valid uses --valid-sample-frac (default 0.15).

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Default: conda env `traj_match` (not system python3). Override with PYTHON=... or CONDA_ENV=...
CONDA_ENV="${CONDA_ENV:-traj_match}"
if [[ -n "${PYTHON:-}" ]]; then
  PY=("$PYTHON")
elif command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
  echo "[run_match_ft_chengdu] WARNING: conda not found; falling back to python3" >&2
  PY=(python3)
fi

OUT="${MATCH_OUT_DIR:-$REPO/cursor_mod_run_chengdu/match_ft_v2maxdtw_scheme4}"
PAIRS="${PAIRS_PKL:-$REPO/cursor_mod_run_chengdu/data/all_pairs_v2_maxdtw.pkl}"
CKPT_DIR="${DIFF_CKPT_DIR:-$REPO/cursor_mod_run_chengdu/ckpt/diff_ft_v2_from_pretrained}"
PRE="${DIFF_PRE:-$CKPT_DIR/best_pre.pth}"
POST="${DIFF_POST:-$CKPT_DIR/best_post.pth}"
CACHE="${HYDRA_CACHE:-$OUT/hydra_only_cache_84.pt}"

export PYTHONUNBUFFERED=1

for f in "$PAIRS" "$PRE" "$POST"; do
  if [[ ! -f "$f" ]]; then
    echo "[run_match_ft_chengdu] missing file: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT"

# Mirror diffusion FT cap from run_config.json (truncate list before cache + train if pkl is longer).
MAX_PAIRS="${MAX_PAIRS:-20000}"
# Smaller = more frequent tqdm steps during cache gen (default was 256 and first step can sit silent a long time).
CACHE_GEN_BATCH="${CACHE_GEN_BATCH:-16}"

REGEN=()
if [[ "${REGENERATE_CACHE:-0}" == "1" ]]; then
  REGEN=(--regenerate-cache)
fi

exec "${PY[@]}" "$REPO/cursor_mod_4/entrypoints/run_ft_match.py" \
  --output-dir "$OUT" \
  --cache "$CACHE" \
  "${REGEN[@]}" \
  --pairs-pkl "$PAIRS" \
  --max-pairs "$MAX_PAIRS" \
  --diff-pre-ckpt "$PRE" \
  --diff-post-ckpt "$POST" \
  --cache-gen-batch "$CACHE_GEN_BATCH" \
  --match-scheme 4 \
  --hydra-tail 64 \
  --rl-use 16 \
  --train-batch 32 \
  --valid-batch 10 \
  --lr 1e-5 \
  --max-epochs 500 \
  --early-stop 10 \
  --sample-ratio 0.1 \
  --sample-seed 42 \
  --valid-sample-frac 0.15 \
  --valid-n-passes 3 \
  --device "${DEVICE:-cuda:0}" \
  "$@"
