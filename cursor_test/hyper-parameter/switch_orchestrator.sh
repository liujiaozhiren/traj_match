#!/usr/bin/env bash
# Retire the old after_master watcher (its job is now folded into queue_runner),
# verify 04 is running diff-only, then start the unified queue_runner.
set -u
cd /home/jh/sjn/traj_match
ROOT=cursor_test/hyper-parameter

# 1) kill old watcher if present
wpids=$(pgrep -f after_master.sh 2>/dev/null)
if [ -n "$wpids" ]; then echo "killing old watcher: $wpids"; kill -9 $wpids 2>/dev/null; fi

# 2) state snapshot
echo "=== 04 stage ==="
pgrep -af "04_lambda_x0" 2>/dev/null | grep -v pgrep \
  | grep -oE "run_tier.py --sweep 04_lambda_x0 --tier [0-9]|ft_diffusion_on_match_pairs.py|run_ft_match.py" | sort -u || echo "04 NOT running"
echo "04 tier dirs: $(ls $ROOT/out/sweep_04_lambda_x0/ 2>/dev/null | tr '\n' ' ')"
echo "=== master gen alive? ==="
pgrep -f "run_ft_match.py --pairs" 2>/dev/null | grep -v pgrep >/dev/null && echo YES || echo NO
echo "=== any scheduler already running? ==="
ps -eo pid,args | grep -E "bash[^|]*queue_runner\.sh" | grep -v grep || echo "none"

# 3) start unified queue_runner
echo "=== starting queue_runner ==="
setsid nohup bash "$ROOT/queue_runner.sh" >/dev/null 2>&1 &
sleep 3
echo "queue_runner: $(ps -eo pid,args | grep -E 'bash[^|]*queue_runner\.sh' | grep -v grep)"
echo "--- queue log ---"; tail -5 "$ROOT/out/_queue_runner.log" 2>/dev/null
