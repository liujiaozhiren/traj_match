#!/usr/bin/env bash
# Valid-only on 6× gap-augmented v2 maxdtw pickles (沙河外 / 本机后台).
# Methods: Traj2SimVec, NeuTraj, DAN, DTW, Hausdorff (load best ckpt, no training).
# Protocol: fracs 15/10/5/2%%, valid_n_passes=3, alpha=1 (A/B only), noise_std_deg=0.
#
# Logs: cursor_data_proc/out/gap_valid_shahe_out/gap{N}/run.log
# JSON: cursor_data_proc/out/gap_valid_shahe_out/gap{N}/grid.json
#
# 沙河外看进度:
#   tail -f cursor_data_proc/out/gap_valid_shahe_out/gap1/run.log

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${GAP_VALID_OUT:-$ROOT/cursor_data_proc/out/gap_valid_shahe_out}"
mkdir -p "$OUT_ROOT"

PAIRS_DIR="${GAP_PAIRS_DIR:-$ROOT/cursor_data_proc/out}"
CKPT_T2S="${CKPT_T2S:-$ROOT/cursor_baseline/results/traj2simvec_run1/best_traj2simvec_ep18_acc0.0183.pth}"
CKPT_NT="${CKPT_NT:-$ROOT/cursor_baseline/results/neutraj_run1/best_neutraj_ep22_acc0.0245.pth}"
CKPT_DAN="${CKPT_DAN:-$ROOT/cursor_baseline/results/shahe_out_shenzhen_dan/best_dan_mot_simplified_ep1_acc0.0765.pth}"
DEVICE="${GAP_VALID_DEVICE:-cuda:0}"

for f in "$CKPT_T2S" "$CKPT_NT" "$CKPT_DAN"; do
  if [[ ! -f "$f" ]]; then
    echo "[gap_valid] missing ckpt: $f" >&2
    exit 1
  fi
done

if command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV:-traj_match}" python)
else
  PY=(python3)
fi

MASTER_LOG="$OUT_ROOT/master.log"
: >"$MASTER_LOG"

echo "=== $(date -Is) gap6 valid start ===" | tee -a "$MASTER_LOG"
echo "OUT_ROOT=$OUT_ROOT DEVICE=$DEVICE" | tee -a "$MASTER_LOG"

for gap in 1 2 3 4 5 6; do
  PAIRS="$PAIRS_DIR/all_pairs_v2_maxdtw_gap${gap}.pkl"
  GAP_OUT="$OUT_ROOT/gap${gap}"
  mkdir -p "$GAP_OUT"
  LOG="$GAP_OUT/run.log"
  JSON="$GAP_OUT/grid.json"

  if [[ ! -f "$PAIRS" ]]; then
    echo "[gap_valid] skip gap${gap}: missing $PAIRS" | tee -a "$MASTER_LOG"
    continue
  fi

  {
    echo "=== gap${gap} $(date -Is) ==="
    echo "PAIRS=$PAIRS"
    echo "JSON=$JSON"
  } | tee -a "$MASTER_LOG" "$LOG"

  "${PY[@]}" "$ROOT/cursor_baseline/run_noise_blend_eval_suite.py" \
    --pairs-pkl "$PAIRS" \
    --pairs-source pkl \
    --output-json "$JSON" \
    --device "$DEVICE" \
    --sample-ratio 0.1 \
    --sample-seed 42 \
    --valid-n-passes 3 \
    --eval-base-seed 42 \
    --n-points 12 \
    --noise-std-deg 0 \
    --alphas 1.0 \
    --repr-encoders traj2simvec,neutraj \
    --repr-ckpt-traj2simvec "$CKPT_T2S" \
    --repr-ckpt-neutraj "$CKPT_NT" \
    --dan-ckpt "$CKPT_DAN" \
    --distance-metrics dtw,hausdorff \
    --chunk-queries 32 \
    2>&1 | tee -a "$LOG"

  echo "=== gap${gap} done $(date -Is) -> $JSON ===" | tee -a "$MASTER_LOG"
done

echo "=== $(date -Is) all gaps finished ===" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG"
