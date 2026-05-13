_base_ = ['./base_innov_coco_256x192.py']

model = dict(
    distill_cfg=dict(
        dynamic_kd=dict(enable=False),
        adaptive_align=dict(enable=False),
        pose_prune=dict(enable=False),
    )
)

work_dir = 'experiments/DIST_INNOV/work_dirs/ablation_0_baseline_static'
