#!/usr/bin/env bash
# Hyper-parameter experiments (self-contained under scripts/).
set -euo pipefail
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPTS/../../.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "${CONDA_ENV:-traj_match}" python)
else
  PY=(python3)
fi

SWEEP="${1:-08_rl_lr}"
TIER="${2:-3}"
DEVICE="${DEVICE:-cuda:0}"
VALID_ONLY="${VALID_ONLY:-}"

ARGS=(--sweep "$SWEEP" --tier "$TIER" --device "$DEVICE")
if [[ -n "$VALID_ONLY" ]]; then
  ARGS+=(--valid-only)
fi

"${PY[@]}" "$SCRIPTS/run_tier.py" "${ARGS[@]}"
