_base_ = ['./base_innov_coco_256x192.py']

model = dict(
    distill_cfg=dict(
        dynamic_kd=dict(
            enable=True,
            prune_ratio_source='param_prune_rate',
            stage_boundaries=(0.2, 0.5),
            hm_stage_weights=(0.03, 0.08, 0.20),
            proto_stage_weights=(0.25, 0.15, 0.05),
            epoch_min_scale=0.5,
            epoch_power=1.0,
            hm_max=0.25,
            proto_max=0.25,
        ),
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
            prune_branches_stage3=(2,),
            prune_branches_stage4=(2, 3),
        ),
        heatmap_supervision=dict(
            enable=True,
            target='gt',
            loss='mse',
            branches=(('stage3', 2, 0.08), ('stage4', 2, 0.06), ('stage4', 3, 0.04)),
        ),
        heatmap_guided_align=dict(
            enable=True,
            temperature=1.0,
            use_proto_energy=True,
            student_layers=(('stage3', 2), ('stage4', 2), ('stage4', 3)),
        ),
    )
)

prune_hook = dict(importance_criterion='pose_proto')

work_dir = 'experiments/DIST_INNOV/work_dirs/full_innov_3methods'
