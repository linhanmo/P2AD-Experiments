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
        pose_prune=dict(
            enable=True,
            ema_momentum=0.95,
            attn_temperature=1.0,
            prune_branches_stage3=(2,),
            prune_branches_stage4=(2, 3),
        ),
    )
)

prune_hook = dict(importance_criterion='pose_proto')

work_dir = 'experiments/DIST_INNOV/work_dirs/ablation_6_pose_prune_plus_adaptive_align'

