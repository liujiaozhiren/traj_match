#!/usr/bin/env bash
# Accurate state check. Runs as `bash check_state.sh` so its own cmdline does not
# contain the scheduler/sweep id strings -> no pgrep self-match.
set -u
echo "=== run_sweep sweeps running ==="
pgrep -af "run_sweep.py --sweep" 2>/dev/null | grep -oE -- "--sweep 0[1-5]_[a-z0-9_]+" | sort -u
echo "=== scheduler (real bash queue_runner.sh) ==="
ps -eo pid,args | grep -E "bash[^|]*queue_runner\.sh" | grep -v grep || echo "NONE"
echo "=== master cache gen ==="
ps -eo pid,args | grep "run_ft_match.py --pairs" | grep -v grep | grep -v "bash " | head -1 || echo "NONE"
echo "=== master cache file ==="
ls -lh cursor_test/hyper-parameter/scripts/registry/hydra_master_tail128.pt 2>/dev/null || echo "not yet"
