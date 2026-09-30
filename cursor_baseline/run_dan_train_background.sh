#!/usr/bin/env bash
# 后台跑 train_dan_traj.py，日志固定到 output-dir/train.log
# 不用 conda run（重定向到文件时常整块缓冲，tail 看不到进度）；直接用 traj_match 环境的 python。
#
# 用法:
#   bash cursor_baseline/run_dan_train_background.sh [OUTPUT_DIR] [PAIRS_PKL]
# 例（成都 v2 maxdtw）:
#   bash cursor_baseline/run_dan_train_background.sh \
#     cursor_baseline/results/dan_chengdu_v2_maxdtw \
#     cursor_mod_run_chengdu/data/all_pairs_v2_maxdtw.pkl
#
# 若找不到环境，可:  export TRAJ_MATCH_PYTHON=/path/to/envs/traj_match/bin/python

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$REPO_ROOT/cursor_baseline/results/dan_chengdu_v2_maxdtw}"
PKL="${2:-$REPO_ROOT/cursor_mod_run_chengdu/data/all_pairs_v2_maxdtw.pkl}"
LOG="$OUT/train.log"
mkdir -p "$OUT"
: >"$LOG"
cd "$REPO_ROOT"

PY="${TRAJ_MATCH_PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in \
    "${HOME}/anaconda3/envs/traj_match/bin/python" \
    "${HOME}/miniconda3/envs/traj_match/bin/python" \
    "/home/jh/anaconda3/envs/traj_match/bin/python"; do
    if [[ -x "$cand" ]]; then
      PY="$cand"
      break
    fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "ERROR: traj_match python not found. Set TRAJ_MATCH_PYTHON=/.../envs/traj_match/bin/python" >&2
  exit 1
fi

nohup env PYTHONUNBUFFERED=1 "$PY" -u cursor_baseline/train_dan_traj.py \
  --pairs-pkl "$PKL" \
  --output-dir "$OUT" \
  >>"$LOG" 2>&1 &
echo $! >"$OUT/train.pid"
echo "Started PID $(cat "$OUT/train.pid")  (python: $PY)"
echo "Log file: $LOG"
echo "Watch:    tail -f $LOG"
echo "Epochs:   grep '\\[dan_mot_simplified\\] ep' \"$LOG\""
echo "Post-eval: grep '\\[dan_mot_post_eval\\]' \"$LOG\""
