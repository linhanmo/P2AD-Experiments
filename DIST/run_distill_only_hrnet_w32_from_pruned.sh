#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/rivermind-data/PoseBH"
CFG="${ROOT_DIR}/experiments/DIST/hrnet_w32_distill_only_from_pruned_coco_256x192.py"
WORK_DIR="${ROOT_DIR}/experiments/DIST/work_dirs/hrnet_w32_distill_only_from_pruned_coco_256x192"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/prune_step_*.pth [extra train args...]"
  exit 2
fi

CKPT="$1"
shift

python "${ROOT_DIR}/tools/train.py" "${CFG}" \
  --work-dir "${WORK_DIR}" \
  --cfg-options "model.student_init_ckpt=${CKPT}" \
  "${@}"

