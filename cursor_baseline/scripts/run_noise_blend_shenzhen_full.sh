#!/usr/bin/env bash
# 深圳坐标系 7-tuple pairs（仓库内 v2 maxdtw）+ 已有 repr/DAN best 权重，跑 6×4 噪声混合 valid 网格。
# 日志: 同目录 run.log  |  结果: grid.json
# TrajCL：当前 results 下无 best_trajcl*.pth，本脚本不跑 trajcl。
# TrajGAT：旧 ckpt 与 v2 maxdtw 全量 pairs 构图节点数不一致会 load 失败，暂不纳入；需要时改用 --pairs-source confile 或重训后再加。
# v2 max-DTW：评估默认只校验 A↔F_A、B↔F_B；若需同时校验 L↔A，在下方 python 命令中加 --blend-check-label-ts（多数 v2 会失败）。

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/cursor_baseline/results/noise_blend_shenzhen_full_20260513"
mkdir -p "$OUT"
LOG="${OUT}/run.log"
JSON="${OUT}/grid.json"

# 与 cursor_mod_4 管线一致：归一化到深圳锚点的 v2 maxdtw pairs（本机已校验 len=7279）
PAIRS="${ROOT}/cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl"

# Baseline best（按文件名 acc 取各自目录内较优；与 train_repr / DAN 默认结构一致）
CKPT_T2S="${ROOT}/cursor_baseline/results/traj2simvec_run1/best_traj2simvec_ep18_acc0.0183.pth"
CKPT_NT="${ROOT}/cursor_baseline/results/neutraj_run1/best_neutraj_ep22_acc0.0245.pth"
CKPT_DAN="${ROOT}/cursor_baseline/results/shahe_out_shenzhen_dan/best_dan_mot_simplified_ep1_acc0.0765.pth"

for f in "$PAIRS" "$CKPT_T2S" "$CKPT_NT" "$CKPT_DAN"; do
  if [[ ! -f "$f" ]]; then
    echo "missing file: $f" >&2
    exit 1
  fi
done

{
  echo "=== $(date -Is) start ==="
  echo "ROOT=$ROOT"
  echo "PAIRS=$PAIRS"
  echo "OUT=$OUT"
  echo "LOG=$LOG"
  echo "JSON=$JSON"
  echo "device=${NOISE_BLEND_DEVICE:-cuda:0}"
} | tee -a "$LOG"

conda run --no-capture-output -n traj_match python "${ROOT}/cursor_baseline/run_noise_blend_eval_suite.py" \
  --pairs-pkl "$PAIRS" \
  --output-json "$JSON" \
  --device "${NOISE_BLEND_DEVICE:-cuda:0}" \
  --sample-ratio 0.1 \
  --sample-seed 42 \
  --valid-n-passes 3 \
  --eval-base-seed 42 \
  --n-points 12 \
  --noise-std-deg 5e-5 \
  --repr-encoders traj2simvec,neutraj \
  --repr-ckpt-traj2simvec "$CKPT_T2S" \
  --repr-ckpt-neutraj "$CKPT_NT" \
  --dan-ckpt "$CKPT_DAN" \
  --chunk-queries 32 \
  2>&1 | tee -a "$LOG"

{
  echo "=== $(date -Is) done ==="
  echo "wrote $JSON"
} | tee -a "$LOG"
