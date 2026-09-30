#!/usr/bin/env bash
# Single orchestrator for the cuda:0 sweeps.
#
# State at launch: 04 (diff-only) is already running; the 128-candidate master
# hydra cache is being generated (counts as one GPU slot). 01/02/03 are diff-only
# and need NOTHING but the GPU; 05 needs the finished master cache.
#
# Policy: keep at most MAXJOBS (=2) GPU jobs running. Launch the pending sweeps in
# order; 05 is additionally gated on the master cache file being fully written.
#
# Run as `bash queue_runner.sh` so this process's own command line contains no
# sweep ids / 'queue_runner' -> the pgrep counting never self-matches.
set -u
cd /home/jh/sjn/traj_match

ROOT=cursor_test/hyper-parameter
OUT="$ROOT/out"
MASTER="$ROOT/scripts/registry/hydra_master_tail128.pt"
LOG="$OUT/_queue_runner.log"
mkdir -p "$OUT"

# diff-only 01/02/03 first (run alongside master), then 05 once master is ready.
PENDING=(01_diffusion_timestamp 02_unet_dim 03_num_res_blocks 05_hydra_tail)
MAXJOBS=2
POLL=60
SETTLE=30

log() { echo "[queue $(date '+%F %T')] $*" | tee -a "$LOG"; }

# Count GPU-occupying jobs: distinct run_sweep sweeps (deduped) + the master-cache
# generator (run_ft_match.py --pairs ... --skip-train), which is not a run_sweep.
count_running() {
  local n
  n=$(pgrep -af "run_sweep.py --sweep" 2>/dev/null \
    | grep -oE -- "--sweep 0[1-5]_[a-z0-9_]+" | sort -u | wc -l)
  if pgrep -f "run_ft_match.py --pairs" 2>/dev/null | grep -v pgrep >/dev/null; then
    n=$((n + 1))
  fi
  echo "$n"
}

# True once the master cache file exists and its size has stopped changing.
master_ready() {
  [ -f "$MASTER" ] || return 1
  local s1 s2
  s1=$(stat -c %s "$MASTER" 2>/dev/null); sleep 5; s2=$(stat -c %s "$MASTER" 2>/dev/null)
  [ -n "$s1" ] && [ "$s1" = "$s2" ] && [ "$s1" -gt 0 ]
}

log "scheduler start; pending=${PENDING[*]} maxjobs=$MAXJOBS"

for s in "${PENDING[@]}"; do
  if [ "$s" = "05_hydra_tail" ]; then
    log "05 waiting for master cache to finish: $MASTER"
    while ! master_ready; do sleep 120; done
    log "05 master cache ready ($(du -h "$MASTER" 2>/dev/null | cut -f1))"
  fi
  while [ "$(count_running)" -ge "$MAXJOBS" ]; do
    sleep "$POLL"
  done
  log "launching $s (running before launch=$(count_running))"
  nohup conda run --no-capture-output -n traj_match \
    python "$ROOT/scripts/run_sweep.py" --sweep "$s" --device cuda:0 \
    >> "$OUT/_sweep_${s}.out" 2>&1 &
  sleep "$SETTLE"
  log "launched $s; running now=$(count_running)"
done

log "all pending launched; scheduler exiting"
