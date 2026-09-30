#!/usr/bin/env bash
# 「沙河外」= 在你本机 / 任意 SSH 终端后台跑训练；日志落盘，给可复制的 tail CLI。
#
# 用法（在仓库根执行）:
#   bash cursor_baseline/run_shahe_out_train.sh [dan|tat] [OUTPUT_DIR] [PAIRS_PKL]
#
# 可选：给训练脚本追加参数（空格分隔），例如打开 AMP：
#   EXTRA_TRAIN_ARGS="--amp" bash cursor_baseline/run_shahe_out_train.sh dan ...
#
# 默认: dan + 成都 v2 maxdtw + 目录 cursor_baseline/results/shahe_out_chengdu_dan
#
# 沙河外只看日志（把下面 ABS_LOG 换成脚本打印的路径）:
#   tail -f ABS_LOG
#   grep '\[dan_mot_simplified\] ep' ABS_LOG
#   grep '\[tat_traj_baseline\] ep' ABS_LOG

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-dan}"
OUT="${2:-$REPO_ROOT/cursor_baseline/results/shahe_out_chengdu_${MODE}}"
PKL="${3:-$REPO_ROOT/cursor_mod_run_chengdu/data/all_pairs_v2_maxdtw.pkl}"
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
  echo "ERROR: set TRAJ_MATCH_PYTHON=/path/to/envs/traj_match/bin/python" >&2
  exit 1
fi

if [[ "$MODE" == "tat" ]]; then
  SCRIPT="cursor_baseline/train_tat_traj.py"
  GREP_TAG='\[tat_traj_baseline\] ep'
else
  SCRIPT="cursor_baseline/train_dan_traj.py"
  GREP_TAG='\[dan_mot_simplified\] ep'
fi

# shellcheck disable=SC2206
read -r -a EXTRA_ARR <<< "${EXTRA_TRAIN_ARGS:-}"
nohup env PYTHONUNBUFFERED=1 "$PY" -u "$SCRIPT" \
  --pairs-pkl "$PKL" \
  --output-dir "$OUT" \
  "${EXTRA_ARR[@]}" \
  >>"$LOG" 2>&1 &
echo $! >"$OUT/train.pid"

echo ""
echo "========== 沙河外 / 任意终端 =========="
echo "实时日志:"
echo "  tail -f $LOG"
echo ""
echo "只看 epoch 汇总:"
echo "  grep '$GREP_TAG' $LOG"
echo ""
echo "post-eval（仅 dan 日志 tag 为 dan_mot_post_eval；tat 同理）:"
echo "  grep '\\[dan_mot_post_eval\\]' $LOG"
echo ""
echo "PID: $(cat "$OUT/train.pid")  OUT: $OUT"
echo "========================================"
