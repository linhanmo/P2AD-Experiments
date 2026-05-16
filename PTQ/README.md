PTQ (ONNXRuntime INT8)

1. Export FP32 ONNX

python experiments/PTQ/export_onnx_heatmap.py \
  --config experiments/PTQ/hrnet_w48_coco_256x192.py \
  --checkpoint /root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth \
  --out experiments/PTQ/work_dirs/hrnet_w48_fp32.onnx

2. PTQ to INT8

python experiments/PTQ/ptq_onnx_int8.py \
  --config experiments/PTQ/hrnet_w48_coco_256x192.py \
  --onnx-fp32 experiments/PTQ/work_dirs/hrnet_w48_fp32.onnx \
  --onnx-int8 experiments/PTQ/work_dirs/hrnet_w48_int8.onnx \
  --calib-split val \
  --calib-batches 200 \
  --per-channel

3. Evaluate INT8 ONNX

python experiments/PTQ/eval_onnx_int8.py \
  --config experiments/PTQ/hrnet_w48_coco_256x192.py \
  --checkpoint /root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth \
  --onnx experiments/PTQ/work_dirs/hrnet_w48_int8.onnx \
  --flip-test

