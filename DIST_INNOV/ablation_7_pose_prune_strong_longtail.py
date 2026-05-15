_base_ = ['./base_innov_coco_256x192.py']

model = dict(
    distill_cfg=dict(
        dynamic_kd=dict(enable=False),
        pose_prune=dict(
            enable=True,
            ema_momentum=0.95,
            attn_temperature=1.0,
            protect_mode='long_tail_boost',
            protect_ratio=0.6,
            per_joint_min=1,
            tail_percentile=0.3,
            tail_boost=2.0,
            bn_mix=0.15,
            score_power=1.0,
        ),
    )
)

prune_hook = dict(importance_criterion='pose_proto')

work_dir = 'experiments/DIST_INNOV/work_dirs/ablation_7_pose_prune_strong_longtail'
