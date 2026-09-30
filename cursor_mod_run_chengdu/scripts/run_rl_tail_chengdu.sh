#!/usr/bin/env bash
# RL tail selector on Chengdu v2 maxdtw: same pair list + split as match/diffusion; hydra cache must match.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CONDA_ENV="${CONDA_ENV:-traj_match}"
if [[ -n "${PYTHON:-}" ]]; then
  PY=("$PYTHON")
elif command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
  PY=(python3)
fi

OUT="${RL_OUT_DIR:-$REPO/cursor_mod_run_chengdu/rl_tail_scheme4_cuda}"
PAIRS="${PAIRS_PKL:-$REPO/cursor_mod_run_chengdu/data/all_pairs_v2_maxdtw.pkl}"
CACHE="${HYDRA_CACHE:-$REPO/cursor_mod_run_chengdu/match_ft_v2maxdtw_scheme4_cuda/hydra_only_cache_84.pt}"
# Must match the frozen MatchModel used for tail selection.
MATCH_CKPT="${MATCH_CKPT:-$REPO/cursor_mod_run_chengdu/match_ft_v2maxdtw_scheme4_cuda/best_match_s4_ep3_acc0.0533.pth}"

MAX_PAIRS="${MAX_PAIRS:-20000}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.1}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"

for f in "$PAIRS" "$CACHE" "$MATCH_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[run_rl_tail_chengdu] missing file: $f" >&2
    exit 1
  fi
done

mkdir -p "$OUT"

exec "${PY[@]}" "$REPO/cursor_mod_4/entrypoints/run_rl_tail_select.py" \
  --output-dir "$OUT" \
  --pairs-pkl "$PAIRS" \
  --max-pairs "$MAX_PAIRS" \
  --cache "$CACHE" \
  --match-ckpt "$MATCH_CKPT" \
  --match-scheme 4 \
  --hydra-tail 64 \
  --k-select 16 \
  --sample-ratio "$SAMPLE_RATIO" \
  --sample-seed "$SAMPLE_SEED" \
  --valid-sample-frac 0.15 \
  --valid-n-passes 3 \
  --device "$DEVICE" \
  "$@"
