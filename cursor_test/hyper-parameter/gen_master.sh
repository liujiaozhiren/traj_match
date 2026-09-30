#!/usr/bin/env bash
# One-time generation of the 128-candidate master hydra cache for the hydra_tail
# sweep (05). Built from the fixed baseline diffusion ckpts over ALL pairs, so it
# aligns row-for-row with every tier; each tier then reuses it via --hydra-tail K.
# HYPER_* pin the diffusion arch to baseline so the baseline ckpt loads strictly.
set -u
cd /home/jh/sjn/traj_match

export HYPER_T=200
export HYPER_UNET_DIM=512
export HYPER_RES_BLOCKS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=cursor_test/hyper-parameter
OUT="$ROOT/out/_hydra_master"
mkdir -p "$OUT"
LOG="$OUT/gen_master.log"

echo "[gen_master $(date '+%F %T')] start" | tee -a "$LOG"
conda run --no-capture-output -n traj_match python \
  "$ROOT/scripts/run_ft_match.py" \
  --pairs-pkl cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl \
  --output-dir "$OUT" \
  --cache "$ROOT/scripts/registry/hydra_master_tail128.pt" \
  --regenerate-cache --hydra-tail 128 --cache-gen-batch 64 \
  --diff-pre-ckpt cursor_mod_runs/diff_ft_pairs131_meters_84/best_pre_ep289_rmse5.908m.pth \
  --diff-post-ckpt cursor_mod_runs/diff_ft_pairs131_meters_84/best_post_ep242_rmse4.678m.pth \
  --skip-train --device cuda:0 >> "$LOG" 2>&1
echo "[gen_master $(date '+%F %T')] exit=$?" | tee -a "$LOG"
