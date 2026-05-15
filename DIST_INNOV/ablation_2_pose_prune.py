_base_ = ['./base_innov_coco_256x192.py']

model = dict(
    distill_cfg=dict(
        dynamic_kd=dict(enable=False),
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

work_dir = 'experiments/DIST_INNOV/work_dirs/ablation_2_pose_prune'
