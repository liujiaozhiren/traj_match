#!/usr/bin/env bash
# Stop sweeps 1/2/3 process trees. Runs as `bash stop_123.sh` so its own
# command line does NOT contain the sweep ids -> no self-kill.
set -u
for s in 01_diffusion_timestamp 02_unet_dim 03_num_res_blocks; do
  pids=$(pgrep -f "$s" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "killing $s: $pids"
    kill -9 $pids 2>/dev/null
  fi
done
sleep 3
echo "=== leftover 1/2/3 ==="
pgrep -af "01_diffusion_timestamp|02_unet_dim|03_num_res_blocks" 2>/dev/null
echo "=== running sweeps now ==="
pgrep -af "run_sweep.py --sweep" 2>/dev/null | grep -oE -- "--sweep 0[1-5]_[a-z0-9_]+" | sort -u
