#!/usr/bin/env bash
# Gap1~6 valid with intentional semantics: gap A/B (info) + frozen v2 hydra cache (7279 rows).
# No training; load Chengdu match + RL selector ckpts.
#
# OUT: cursor_data_proc/out/gap_mod4_v2cache_valid/gap{N}/grid.json
# Logs: .../gap{N}/run.log
#   tail -f cursor_data_proc/out/gap_mod4_v2cache_valid/master.log

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${GAP_MOD4_OUT:-$ROOT/cursor_data_proc/out/gap_mod4_v2cache_valid}"
PAIRS_DIR="${GAP_PAIRS_DIR:-$ROOT/cursor_data_proc/out}"
CACHE="${HYDRA_CACHE:-$ROOT/cursor_mod_runs/match_ft_84/hydra_only_cache_84.pt}"
DEVICE="${GAP_MOD4_DEVICE:-cuda:0}"

MATCH_CKPT="${MATCH_CKPT:-$ROOT/cursor_mod_run_chengdu/match_ft_v2maxdtw_bb_ce_s4/best_match_s4_ep26_acc0.0956.pth}"
MATCH_CKPT_RL="${MATCH_CKPT_RL:-$ROOT/cursor_mod_run_chengdu/match_ft_v2maxdtw_scheme4_cuda/best_match_s4_ep3_acc0.0533.pth}"
SELECTOR_CKPT="${SELECTOR_CKPT:-$ROOT/cursor_mod_run_chengdu/rl_tail_scheme4_cuda/best_selector_k16_acc0.0578_ep7.pt}"

for f in "$CACHE" "$MATCH_CKPT" "$MATCH_CKPT_RL" "$SELECTOR_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[gap_mod4_v2cache] missing: $f" >&2
    exit 1
  fi
done

if command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV:-traj_match}" python)
else
  PY=(python3)
fi

mkdir -p "$OUT_ROOT"
MASTER_LOG="$OUT_ROOT/master.log"
: >"$MASTER_LOG"

echo "=== $(date -Is) gap6 mod4 (gap pkl + v2 cache) start ===" | tee -a "$MASTER_LOG"
echo "CACHE=$CACHE DEVICE=$DEVICE" | tee -a "$MASTER_LOG"
echo "MATCH_CKPT=$MATCH_CKPT" | tee -a "$MASTER_LOG"
echo "MATCH_CKPT_RL=$MATCH_CKPT_RL SELECTOR_CKPT=$SELECTOR_CKPT" | tee -a "$MASTER_LOG"

for gap in 1 2 3 4 5 6; do
  PAIRS="$PAIRS_DIR/all_pairs_v2_maxdtw_gap${gap}.pkl"
  GAP_OUT="$OUT_ROOT/gap${gap}"
  mkdir -p "$GAP_OUT"
  LOG="$GAP_OUT/run.log"
  JSON="$GAP_OUT/grid.json"

  if [[ ! -f "$PAIRS" ]]; then
    echo "[gap_mod4_v2cache] missing pairs: $PAIRS" | tee -a "$MASTER_LOG" >&2
    exit 1
  fi

  echo "--- gap${gap} $(date -Is) ---" | tee -a "$MASTER_LOG"
  {
    echo "PAIRS=$PAIRS"
    echo "JSON=$JSON"
  } | tee -a "$MASTER_LOG"

  "${PY[@]}" "$ROOT/cursor_mod_4/entrypoints/eval_match_rl_noise_blend_grid.py" \
    --pairs-pkl "$PAIRS" \
    --cache "$CACHE" \
    --match-ckpt "$MATCH_CKPT" \
    --match-ckpt-rl "$MATCH_CKPT_RL" \
    --selector-ckpt "$SELECTOR_CKPT" \
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
    2>&1 | tee "$LOG"

  echo "gap${gap} done -> $JSON" | tee -a "$MASTER_LOG"
done

echo "=== $(date -Is) all gaps done ===" | tee -a "$MASTER_LOG"
