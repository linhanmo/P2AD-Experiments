_base_ = ['./base_innov_coco_256x192.py']

model = dict(
    distill_cfg=dict(
        dynamic_kd=dict(
            enable=True,
            prune_ratio_source='param_prune_rate',
            stage_boundaries=(0.2, 0.4),
            hm_stage_weights=(0.05, 0.10, 0.20),
            proto_stage_weights=(0.20, 0.15, 0.05),
            epoch_min_scale=0.5,
            epoch_power=1.0,
            hm_max=0.25,
            proto_max=0.25,
        ),
        adaptive_align=dict(enable=False),
        pose_prune=dict(enable=False),
    )
)

work_dir = 'experiments/DIST_INNOV/work_dirs/ablation_1_dynamic_weight'
