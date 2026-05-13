#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/root/rivermind-data/PoseBH"

CFG_REL="${1:-experiments/DIST_INNOV/full_innov_3methods.py}"
if [[ "${CFG_REL}" == /* ]]; then
  CFG="${CFG_REL}"
else
  CFG="${ROOT_DIR}/${CFG_REL}"
fi

CFG_BASENAME="$(basename "${CFG}")"
CFG_STEM="${CFG_BASENAME%.py}"
WORK_DIR="${ROOT_DIR}/experiments/DIST_INNOV/work_dirs/${WORK_DIR_NAME:-${CFG_STEM}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

shift || true

python "${ROOT_DIR}/tools/train.py" "${CFG}" --work-dir "${WORK_DIR}" "${@}"

