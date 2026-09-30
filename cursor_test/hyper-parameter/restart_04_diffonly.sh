#!/usr/bin/env bash
# Kill the running sweep 04 (now mid useless cache-gen) and relaunch it diff-only.
# Run as `bash restart_04_diffonly.sh` so its own cmdline has no sweep id.
set -u
cd /home/jh/sjn/traj_match

pids=$(pgrep -f "04_lambda_x0" 2>/dev/null)
if [ -n "$pids" ]; then
  echo "killing 04 tree: $pids"
  kill -9 $pids 2>/dev/null
fi
sleep 3
echo "=== 04 leftover (should be empty) ==="
pgrep -af "04_lambda_x0" 2>/dev/null | grep -v restart_04 || echo "none"

ROOT=cursor_test/hyper-parameter
OUT="$ROOT/out"
mkdir -p "$OUT"
echo "=== relaunching 04 (diff-only) ==="
setsid nohup conda run --no-capture-output -n traj_match \
  python "$ROOT/scripts/run_sweep.py" --sweep 04_lambda_x0 --device cuda:0 \
  >> "$OUT/_sweep_04_lambda_x0.out" 2>&1 &
sleep 5
echo "launched; pid chain:"
pgrep -af "run_sweep.py --sweep 04" 2>/dev/null | grep -v restart_04
