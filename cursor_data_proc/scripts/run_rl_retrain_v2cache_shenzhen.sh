#!/usr/bin/env bash
# Retrain RL selector on Shenzhen v2 (7279) + match_ft_84 hydra cache; frozen scheme-4 match.
# Then re-run gap1~6 valid (gap A/B info + same v2 cache) with the new selector.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV:-traj_match}" python)
else
  PY=(python3)
fi

RL_OUT="${RL_OUT_DIR:-$ROOT/cursor_data_proc/out/rl_retrain_v2cache_s4}"
PAIRS_V2="${PAIRS_V2:-$ROOT/cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl}"
CACHE="${HYDRA_CACHE:-$ROOT/cursor_mod_runs/match_ft_84/hydra_only_cache_84.pt}"
MATCH_CKPT="${MATCH_CKPT:-$ROOT/cursor_mod_runs/match_ft_84/best_match_s4_ep41_acc0.1590.pth}"
DEVICE="${RL_DEVICE:-cuda:0}"

GAP_VALID_OUT="${GAP_VALID_OUT:-$ROOT/cursor_data_proc/out/gap_mod4_v2cache_valid_retrain}"

for f in "$PAIRS_V2" "$CACHE" "$MATCH_CKPT"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done

mkdir -p "$RL_OUT"
LOG="$RL_OUT/train.log"

echo "=== $(date -Is) RL retrain start ===" | tee "$LOG"
echo "PAIRS_V2=$PAIRS_V2" | tee -a "$LOG"
echo "CACHE=$CACHE MATCH_CKPT=$MATCH_CKPT" | tee -a "$LOG"
echo "RL_OUT=$RL_OUT DEVICE=$DEVICE" | tee -a "$LOG"

"${PY[@]}" "$ROOT/cursor_mod_4/entrypoints/run_rl_tail_select.py" \
  --output-dir "$RL_OUT" \
  --pairs-pkl "$PAIRS_V2" \
  --cache "$CACHE" \
  --match-ckpt "$MATCH_CKPT" \
  --match-scheme 4 \
  --hydra-tail 64 \
  --k-select 16 \
  --sample-ratio 0.1 \
  --sample-seed 42 \
  --valid-sample-frac 0.15 \
  --valid-n-passes 3 \
  --fixed-valid-seed 42 \
  --train-reward-mode bb_ce \
  --max-epochs 200 \
  --early-stop 15 \
  --device "$DEVICE" \
  2>&1 | tee -a "$LOG"

BEST_SEL="$(ls -t "$RL_OUT"/best_selector_k16_acc*.pt 2>/dev/null | head -1)"
if [[ -z "${BEST_SEL:-}" ]]; then
  echo "[rl_retrain] no selector ckpt under $RL_OUT" | tee -a "$LOG" >&2
  exit 1
fi
echo "BEST_SEL=$BEST_SEL" | tee -a "$LOG"

echo "=== $(date -Is) gap1~6 valid with retrained selector ===" | tee -a "$LOG"
export HYDRA_CACHE="$CACHE"
export MATCH_CKPT="$MATCH_CKPT"
export MATCH_CKPT_RL="$MATCH_CKPT"
export SELECTOR_CKPT="$BEST_SEL"
export GAP_MOD4_OUT="$GAP_VALID_OUT"
export GAP_MOD4_DEVICE="$DEVICE"
"$ROOT/cursor_data_proc/scripts/run_gap6_mod4_valid_v2cache.sh" 2>&1 | tee -a "$LOG"

GAP_MOD4_OUT="$GAP_VALID_OUT" "${PY[@]}" "$ROOT/cursor_data_proc/scripts/summarize_gap_mod4_v2cache_tables.py"

echo "=== $(date -Is) done RL_OUT=$RL_OUT GAP_VALID_OUT=$GAP_VALID_OUT ===" | tee -a "$LOG"
