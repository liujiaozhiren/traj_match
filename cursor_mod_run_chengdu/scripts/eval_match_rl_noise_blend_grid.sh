#!/usr/bin/env bash
# Valid-only: 6 α (noise–label blend) × 4 valid fracs × {match_vanilla | rl_selector}.
# 不训练；读最优 Match 与 RL selector 权重。
#
# 深圳数据：默认 PAIRS_PKL 指向 cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl（7279 条）。
# Hydra cache 行数必须与 pairs 列表长度一致；若尚无深圳 cache，请先对同一 pkl 跑
#   cursor_mod_4/entrypoints/run_ft_match.py --regenerate-cache …
# 或临时把 PAIRS_PKL 设为成都 data/all_pairs_v2_maxdtw.pkl 并使用现有 scheme4_cuda 的 cache。
#
# RL 训练时 frozen match 多为 scheme4_cuda；本脚本默认：
#   MATCH_CKPT        → bb_ce 最优（看「仅 match」）
#   MATCH_CKPT_RL     → scheme4_cuda（与 selector 一起的 frozen match）
# 若你 RL 与 bb_ce 用同一 match 训练，可 export MATCH_CKPT_RL=""  unset 后只设 MATCH_CKPT。

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

CONDA_ENV="${CONDA_ENV:-traj_match}"
if command -v conda >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "$CONDA_ENV" python)
else
  PY=(python3)
fi

OUT="${MATCH_RL_NOISE_BLEND_OUT:-$REPO/cursor_mod_run_chengdu/match_rl_noise_blend_eval}"
mkdir -p "$OUT"
LOG="${OUT}/run.log"
JSON="${OUT}/grid.json"

PAIRS_PKL="${PAIRS_PKL:-$REPO/cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl}"
CACHE="${HYDRA_CACHE:?请设置 HYDRA_CACHE 为与 PAIRS_PKL 对齐的 hydra_only_cache_84.pt}"

MATCH_CKPT="${MATCH_CKPT:-$REPO/cursor_mod_run_chengdu/match_ft_v2maxdtw_bb_ce_s4/best_match_s4_ep26_acc0.0956.pth}"
MATCH_CKPT_RL="${MATCH_CKPT_RL:-$REPO/cursor_mod_run_chengdu/match_ft_v2maxdtw_scheme4_cuda/best_match_s4_ep3_acc0.0533.pth}"
SELECTOR_CKPT="${SELECTOR_CKPT:-$REPO/cursor_mod_run_chengdu/rl_tail_scheme4_cuda/best_selector_k16_acc0.0578_ep7.pt}"

DEVICE="${DEVICE:-cuda:0}"
VALID_N_PASSES="${VALID_N_PASSES:-3}"

EXTRA_MATCH_RL=()
if [[ -n "${MATCH_CKPT_RL:-}" ]]; then
  EXTRA_MATCH_RL=(--match-ckpt-rl "$MATCH_CKPT_RL")
fi

{
  echo "=== $(date -Is) start ==="
  echo "PAIRS_PKL=$PAIRS_PKL"
  echo "CACHE=$CACHE"
  echo "MATCH_CKPT=$MATCH_CKPT"
  echo "MATCH_CKPT_RL=${MATCH_CKPT_RL:-}"
  echo "SELECTOR_CKPT=$SELECTOR_CKPT"
  echo "OUT=$OUT LOG=$LOG JSON=$JSON"
} | tee "$LOG"

"${PY[@]}" "$REPO/cursor_mod_4/entrypoints/eval_match_rl_noise_blend_grid.py" \
  --pairs-pkl "$PAIRS_PKL" \
  --cache "$CACHE" \
  --match-ckpt "$MATCH_CKPT" \
  "${EXTRA_MATCH_RL[@]}" \
  --selector-ckpt "$SELECTOR_CKPT" \
  --match-scheme 4 \
  --hydra-tail 64 \
  --k-select 16 \
  --sample-ratio "${SAMPLE_RATIO:-0.1}" \
  --sample-seed "${SAMPLE_SEED:-42}" \
  --valid-n-passes "$VALID_N_PASSES" \
  --eval-base-seed "${EVAL_BASE_SEED:-42}" \
  --valid-batch "${VALID_BATCH:-10}" \
  --device "$DEVICE" \
  --output-json "$JSON" \
  "$@" \
  2>&1 | tee -a "$LOG"

echo "=== $(date -Is) done === tail -f $LOG" | tee -a "$LOG"
