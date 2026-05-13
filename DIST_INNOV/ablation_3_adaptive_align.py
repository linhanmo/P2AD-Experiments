_base_ = ['./base_innov_coco_256x192.py']

model = dict(
    distill_cfg=dict(
        dynamic_kd=dict(enable=False),
        adaptive_align=dict(
            enable=True,
            hidden=256,
            eps=1.0e-5,
            normalize=True,
            normalize_teacher=True,
        ),
        pose_prune=dict(enable=False),
    )
)

work_dir = 'experiments/DIST_INNOV/work_dirs/ablation_3_adaptive_align'

