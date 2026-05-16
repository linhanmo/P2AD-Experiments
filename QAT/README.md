QAT (PyTorch INT8, CPU)

1. QAT fine-tuning + convert to INT8 checkpoint

python experiments/QAT/train_qat.py \
  --config experiments/QAT/hrnet_w48_coco_256x192_qat.py \
  --checkpoint /root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth \
  --work-dir experiments/QAT/work_dirs/hrnet_w48_qat \
  --epochs 10

2. Evaluate INT8 checkpoint

python experiments/QAT/eval_qat_int8.py \
  --config experiments/QAT/hrnet_w48_coco_256x192_qat.py \
  --float-checkpoint /root/rivermind-data/PoseBH/weights/pose_hrnet_w48_256x192.pth \
  --qat-int8 experiments/QAT/work_dirs/hrnet_w48_qat/qat_int8.pth \
  --flip-test

