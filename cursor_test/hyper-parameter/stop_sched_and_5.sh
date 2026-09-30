#!/usr/bin/env bash
# Stop the queue scheduler FIRST (so the freed slot is not grabbed by sweep 01),
# then kill the sweep 05 process tree. Run as `bash stop_sched_and_5.sh` so this
# process's own command line contains neither "queue_runner.sh" nor a sweep id
# -> pgrep/kill below never self-match.
set -u

qpids=$(pgrep -f queue_runner.sh 2>/dev/null)
if [ -n "$qpids" ]; then
  echo "killing scheduler: $qpids"
  kill -9 $qpids 2>/dev/null
fi

spids=$(pgrep -f 05_hydra_tail 2>/dev/null)
if [ -n "$spids" ]; then
  echo "killing 05 tree: $spids"
  kill -9 $spids 2>/dev/null
fi

sleep 3
echo "=== scheduler leftover (should be empty) ==="
pgrep -af queue_runner.sh 2>/dev/null
echo "=== running sweeps now (expect only 04) ==="
pgrep -af "run_sweep.py --sweep" 2>/dev/null | grep -oE -- "--sweep 0[1-5]_[a-z0-9_]+" | sort -u
