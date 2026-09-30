#!/usr/bin/env bash
# Per gap (1..6): RL retrain on gap{N} pkl + shared v2 hydra cache, then valid with that gap's selector only.
# Semantics: train/valid both use gap A/B (info) + v2 cache (hydra); frozen Shenzhen scheme-4 match.
#
#   cursor_data_proc/out/gap{N}_rl_retrain_v2cache/     — selector ckpt + train.log
#   cursor_data_proc/out/gap_mod4_v2cache_rl_pergap/gap{N}/grid.json
#
#   tail -f cursor_data_proc/out/gap_mod4_v2cache_rl_pergap/master.log

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV:-traj_match}" python)
else
  PY=(python3)
fi

PAIRS_DIR="${GAP_PAIRS_DIR:-$ROOT/cursor_data_proc/out}"
CACHE="${HYDRA_CACHE:-$ROOT/cursor_mod_runs/match_ft_84/hydra_only_cache_84.pt}"
MATCH_CKPT="${MATCH_CKPT:-$ROOT/cursor_mod_runs/match_ft_84/best_match_s4_ep41_acc0.1590.pth}"
DEVICE="${GAP_RL_DEVICE:-cuda:0}"
RL_ROOT="${GAP_RL_ROOT:-$ROOT/cursor_data_proc/out}"
VALID_ROOT="${GAP_VALID_ROOT:-$ROOT/cursor_data_proc/out/gap_mod4_v2cache_rl_pergap}"

GAPS="${GAPS:-1 2 3 4 5 6}"

for f in "$CACHE" "$MATCH_CKPT"; do
  [[ -f "$f" ]] || { echo "[gap_rl_pergap] missing: $f" >&2; exit 1; }
done

mkdir -p "$VALID_ROOT"
MASTER_LOG="$VALID_ROOT/master.log"
: >"$MASTER_LOG"

echo "=== $(date -Is) per-gap RL retrain + valid start ===" | tee -a "$MASTER_LOG"
echo "CACHE=$CACHE MATCH_CKPT=$MATCH_CKPT DEVICE=$DEVICE" | tee -a "$MASTER_LOG"

for gap in $GAPS; do
  PAIRS="$PAIRS_DIR/all_pairs_v2_maxdtw_gap${gap}.pkl"
  RL_OUT="$RL_ROOT/gap${gap}_rl_retrain_v2cache"
  GAP_VALID_OUT="$VALID_ROOT/gap${gap}"
  mkdir -p "$RL_OUT" "$GAP_VALID_OUT"

  [[ -f "$PAIRS" ]] || { echo "[gap_rl_pergap] missing $PAIRS" | tee -a "$MASTER_LOG" >&2; exit 1; }

  echo "--- gap${gap} RL train $(date -Is) ---" | tee -a "$MASTER_LOG"
  RL_LOG="$RL_OUT/train.log"
  {
    echo "PAIRS=$PAIRS"
    echo "RL_OUT=$RL_OUT"
  } | tee -a "$MASTER_LOG"

  "${PY[@]}" "$ROOT/cursor_mod_4/entrypoints/run_rl_tail_select.py" \
    --output-dir "$RL_OUT" \
    --pairs-pkl "$PAIRS" \
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
    2>&1 | tee "$RL_LOG"

  BEST_SEL="$(ls -t "$RL_OUT"/best_selector_k16_acc*.pt 2>/dev/null | head -1)"
  if [[ -z "${BEST_SEL:-}" ]]; then
    echo "[gap_rl_pergap] no selector ckpt in $RL_OUT" | tee -a "$MASTER_LOG" >&2
    exit 1
  fi
  echo "gap${gap} BEST_SEL=$BEST_SEL" | tee -a "$MASTER_LOG"

  echo "--- gap${gap} valid $(date -Is) ---" | tee -a "$MASTER_LOG"
  VALID_LOG="$GAP_VALID_OUT/run.log"
  JSON="$GAP_VALID_OUT/grid.json"

  "${PY[@]}" "$ROOT/cursor_mod_4/entrypoints/eval_match_rl_noise_blend_grid.py" \
    --pairs-pkl "$PAIRS" \
    --cache "$CACHE" \
    --match-ckpt "$MATCH_CKPT" \
    --match-ckpt-rl "$MATCH_CKPT" \
    --selector-ckpt "$BEST_SEL" \
    --match-scheme 4 \
    --hydra-tail 64 \
    --k-select 16 \
    --sample-ratio 0.1 \
    --sample-seed 42 \
    --valid-n-passes 3 \
    --eval-base-seed 42 \
    --valid-batch 10 \
    --device "$DEVICE" \
    --alphas 1.0 \
    --valid-sample-fracs 0.15,0.10,0.05,0.02 \
    --modes match_vanilla,rl_selector \
    --output-json "$JSON" \
    2>&1 | tee "$VALID_LOG"

  echo "gap${gap} valid done -> $JSON" | tee -a "$MASTER_LOG"
done

GAP_MOD4_OUT="$VALID_ROOT" "${PY[@]}" "$ROOT/cursor_data_proc/scripts/summarize_gap_mod4_v2cache_tables.py" \
  | tee -a "$MASTER_LOG"
GAP_VALID_ROOT="$VALID_ROOT" GAP_RL_ROOT="$RL_ROOT" "${PY[@]}" \
  "$ROOT/cursor_data_proc/scripts/summarize_gap_rl_retrain_valid.py" | tee -a "$MASTER_LOG"

echo "=== $(date -Is) all per-gap RL+valid done ===" | tee -a "$MASTER_LOG"
