#!/usr/bin/env bash
# Smoke: all 14 methods, small N.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV:-traj_match}" python)
else
  PY=(python3)
fi

METHODS="dtw,hausdorff,frechet,sspd,t3s,traj2simvec,neutraj,trajgat,trajcl,st2vec,dan,tat,attnmove,mod4"

"${PY[@]}" cursor_test/match_efficiency/run_e2e.py \
  --methods "$METHODS" \
  --gallery-size "${GALLERY_SIZE:-32}" \
  --gallery-batch "${GALLERY_BATCH:-32}" \
  --n-queries 2 \
  --device "${DEVICE:-cuda:0}" \
  --output-json cursor_test/match_efficiency/out/e2e_smoke.json
