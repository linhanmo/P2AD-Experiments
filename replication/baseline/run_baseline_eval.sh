#!/usr/bin/env bash
# ==========================================
# 执行基线模型在所有数据集 (COCO, MPII, OCHuman, AP10K, WholeBody) 上的精度评估的主脚本。
# ==========================================
# 此脚本用于自动化评估 PoseBH 基线模型在 5 个核心数据集上的精度
# 执行前需要确保对应的环境和数据路径已经准备好
set -euo pipefail

ROOT_DIR="."
OUT_DIR="${ROOT_DIR}/experiments/stage1/baseline"
LOG_DIR="${OUT_DIR}/logs"
SPLIT_DIR="${OUT_DIR}/split_weights"

# 官方的预训练权重路径
WEIGHT_PATH="${ROOT_DIR}/weights/posebh/base.pth"

mkdir -p "${LOG_DIR}" "${SPLIT_DIR}"

# 1. 拆分模型权重
# PoseBH 的原始权重包含多个数据集的特定模块 (如多个 head)。
# 使用 tools/model_split.py 提取各个特定数据集的干净权重，以避免在评测时遇到 unexpected key 警告。
python "${ROOT_DIR}/tools/model_split.py" --source "${WEIGHT_PATH}" --target "${SPLIT_DIR}" --datasets human_animal |& tee "${LOG_DIR}/01_model_split.log"

# 2. 定义各数据集对应的配置文件
COCO_CFG="${ROOT_DIR}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py"
MPII_CFG="${ROOT_DIR}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/ViTPose_base_mpii_256x192.py"
OCHUMAN_CFG="${ROOT_DIR}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/ochuman/ViTPose_base_ochuman_256x192.py"
AP10K_CFG="${ROOT_DIR}/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py"
WHOLEBODY_CFG="${ROOT_DIR}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/ViTPose_base_wholebody_256x192.py"

# 3. 定义拆分后的权重文件
COCO_W="${SPLIT_DIR}/coco.pth"
MPII_W="${SPLIT_DIR}/mpii.pth"
AP10K_W="${SPLIT_DIR}/ap10k.pth"
WHOLEBODY_W="${SPLIT_DIR}/wholebody.pth"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# 官方的 COCO 人体检测框结果，用于 Top-Down 评估
DET_JSON="${ROOT_DIR}/data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json"

# 检查 CUDA 是否可用（在沙盒/CPU 环境中自动跳过实际推理）
CUDA_OK="$(python - <<'PY'
import torch
print('1' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else '0')
PY
)"
echo "cuda_available=${CUDA_OK}" |& tee "${LOG_DIR}/00_cuda_check.log"

# 定义测试运行函数
run_test() {
  local cfg="$1"       # 配置文件路径
  local weight="$2"    # 权重文件路径
  local log_name="$3"  # 日志文件名
  shift 3              # 提取后面的附加参数作为 cfg-options

  if [[ "${CUDA_OK}" != "1" ]]; then
    echo "SKIP ${log_name}: CUDA is not available (need at least 1 GPU)." |& tee "${LOG_DIR}/${log_name}"
    return 0
  fi

  set +e
  # 运行分布式测试脚本，1 表示 1 个 GPU
  bash "${ROOT_DIR}/tools/dist_test.sh" "${cfg}" "${weight}" 1 "$@" |& tee "${LOG_DIR}/${log_name}"
  local code=${PIPESTATUS[0]}
  set -e

  if [[ ${code} -ne 0 ]]; then
    echo "FAIL ${log_name}: exit_code=${code}" |& tee -a "${LOG_DIR}/${log_name}"
  fi
  return 0
}

# ----------------- 开始在各个数据集上执行评估 -----------------

# 评估 MPII (单人姿态，GT bbox)
run_test "${MPII_CFG}" "${MPII_W}" "02_mpii_test.log" \
  --cfg-options data.test_dataloader.samples_per_gpu=8 data.workers_per_gpu=4

# 评估 OCHuman (严重遮挡，直接复用 COCO 的预训练权重，GT bbox)
run_test "${OCHUMAN_CFG}" "${COCO_W}" "03_ochuman_test.log" \
  --cfg-options data.test_dataloader.samples_per_gpu=8 data.workers_per_gpu=4

# 评估 AP-10K (跨物种动物姿态，GT bbox)
run_test "${AP10K_CFG}" "${AP10K_W}" "04_ap10k_test.log" \
  --cfg-options data.test_dataloader.samples_per_gpu=8 data.workers_per_gpu=4

# 评估 COCO 和 COCO-WholeBody
# 根据是否提供了官方的人体检测框文件 (DET_JSON) 决定评估协议 (DET_BBOX 或 GT_BBOX)
if [[ "${CUDA_OK}" == "1" ]]; then
  if [[ -f "${DET_JSON}" ]]; then
    echo "protocol: DET_BBOX (${DET_JSON})" |& tee "${LOG_DIR}/05_wholebody_test.log" "${LOG_DIR}/06_coco_test.log"

    # 使用检测框文件 (DET_BBOX) 协议
    run_test "${WHOLEBODY_CFG}" "${WHOLEBODY_W}" "05_wholebody_test.log" \
      --cfg-options data.test_dataloader.samples_per_gpu=4 data.workers_per_gpu=4

    run_test "${COCO_CFG}" "${COCO_W}" "06_coco_test.log" \
      --cfg-options data.test_dataloader.samples_per_gpu=8 data.workers_per_gpu=4
  else
    echo "protocol: GT_BBOX (missing ${DET_JSON})" |& tee "${LOG_DIR}/05_wholebody_test.log" "${LOG_DIR}/06_coco_test.log"

    # 如果检测框文件缺失，退回到使用 Ground Truth bbox (GT_BBOX) 协议
    run_test "${WHOLEBODY_CFG}" "${WHOLEBODY_W}" "05_wholebody_test.log" \
      --cfg-options data.test.data_cfg.use_gt_bbox=True data.test.data_cfg.bbox_file='' data.test_dataloader.samples_per_gpu=4 data.workers_per_gpu=4

    run_test "${COCO_CFG}" "${COCO_W}" "06_coco_test.log" \
      --cfg-options data.test.data_cfg.use_gt_bbox=True data.test.data_cfg.bbox_file='' data.test_dataloader.samples_per_gpu=8 data.workers_per_gpu=4
  fi
else
  echo "protocol: SKIP (CUDA unavailable)" |& tee "${LOG_DIR}/05_wholebody_test.log" "${LOG_DIR}/06_coco_test.log"
fi

# 4. 生成对齐报告
# 调用总结脚本，从日志中解析所有 AP/AR/PCKh 指标并与官方指标对比，生成 CSV/JSON 报告。
python "${OUT_DIR}/summarize_baseline.py" --log-dir "${LOG_DIR}" --out-dir "${OUT_DIR}" |& tee "${LOG_DIR}/07_summarize.log"
