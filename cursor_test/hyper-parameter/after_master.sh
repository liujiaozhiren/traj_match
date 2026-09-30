#!/usr/bin/env bash
# Auto-orchestrator: wait for the 128-candidate master hydra cache to finish
# generating, then (1) launch sweep 05 (now reuses the master, no regeneration)
# and (2) start the queue scheduler for sweeps 01/02/03. Keeps <=2 of {01..05}
# running on cuda:0 at once (04 is already running).
#
# Run as `bash after_master.sh` so its own cmdline does not contain sweep ids.
set -u
cd /home/jh/sjn/traj_match

ROOT=cursor_test/hyper-parameter
OUT="$ROOT/out"
MASTER="$ROOT/scripts/registry/hydra_master_tail128.pt"
LOG="$OUT/_after_master.log"
mkdir -p "$OUT"

log() { echo "[after_master $(date '+%F %T')] $*" | tee -a "$LOG"; }

master_gen_running() { pgrep -f "run_ft_match.py --pairs" 2>/dev/null | grep -v pgrep >/dev/null; }

log "waiting for master cache: $MASTER"
# Wait until the generator process has exited AND the file exists & is stable.
while master_gen_running; do
  sleep 120
done
if [ ! -f "$MASTER" ]; then
  log "ERROR: master gen exited but $MASTER missing — aborting (check gen_master.log)"
  exit 1
fi
# size-stability guard (in case file is still flushing)
s1=$(stat -c %s "$MASTER"); sleep 10; s2=$(stat -c %s "$MASTER")
while [ "$s1" != "$s2" ]; do s1=$s2; sleep 10; s2=$(stat -c %s "$MASTER"); done
log "master cache ready ($(du -h "$MASTER" | cut -f1))"

# 1) launch sweep 05 (reuses master via run_tier cache_master logic)
log "launching sweep 05_hydra_tail"
nohup conda run --no-capture-output -n traj_match \
  python "$ROOT/scripts/run_sweep.py" --sweep 05_hydra_tail --device cuda:0 \
  >> "$OUT/_sweep_05_hydra_tail.out" 2>&1 &
sleep 30
log "sweep 05 launched"

# 2) hand off to the queue scheduler for 01/02/03 (keeps total <=2 concurrent)
log "starting queue scheduler for 01/02/03"
setsid nohup bash "$ROOT/queue_runner.sh" >/dev/null 2>&1 &
log "scheduler started; after_master exiting"
