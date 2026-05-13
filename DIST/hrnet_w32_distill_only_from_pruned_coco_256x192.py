_base_ = [
    '../../configs/_base_/default_runtime.py',
    '../../configs/_base_/datasets/coco.py',
]

import sys

sys.path.insert(0, './experiments/DIST')

custom_imports = dict(
    imports=[
        'experiments.DIST.distill_only',
        'experiments.DIST.custom_hooks',
    ],
    allow_failed_imports=False,
)

evaluation = dict(interval=10, metric='mAP', save_best='AP')

overall_log_name = 'DIST_distill_only_all.log'

optimizer = dict(type='Adam', lr=5e-5)
optimizer_config = dict(grad_clip=None)

fp16 = dict(loss_scale=512.0)

lr_config = dict(policy='step', step=[])
total_epochs = 200

target_type = 'GaussianHeatmap'
channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
)

data_cfg = dict(
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='/root/rivermind-data/PoseBH/data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(type='TopDownHalfBodyTransform', num_joints_half_body=8, prob_half_body=0.3),
    dict(type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2, encoding='UDP', target_type=target_type),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file',
            'joints_3d',
            'joints_3d_visible',
            'center',
            'scale',
            'rotation',
            'bbox_score',
            'flip_pairs',
        ],
    ),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine', use_udp=True),
    dict(type='ToTensor'),
    dict(type='NormalizeTensor', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['image_file', 'center', 'scale', 'rotation', 'bbox_score', 'flip_pairs'],
    ),
]

test_pipeline = val_pipeline

data_root = '/root/rivermind-data/PoseBH/data/coco'
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}},
    ),
)

teacher_model = dict(
    type='TopDownMoEProto',
    pretrained=None,
    multihead_pretrained=dict(
        checkpoint='/root/rivermind-data/PoseBH/weights/posebh/base.pth',
        strict=True,
    ),
    backbone=dict(
        type='ViTMoE',
        img_size=(256, 192),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ratio=1,
        use_checkpoint=False,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.3,
        num_expert=6,
        part_features=192,
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=768,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(final_conv_kernel=1),
        out_channels=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    associate_keypoint_head=[
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=14,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        ),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=16,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        ),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=17,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        ),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=17,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        ),
        dict(
            type='TopdownHeatmapSimpleHead',
            in_channels=768,
            num_deconv_layers=2,
            num_deconv_filters=(256, 256),
            num_deconv_kernels=(4, 4),
            extra=dict(final_conv_kernel=1),
            out_channels=133,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        ),
    ],
    proto_head=dict(
        type='KptProtoHead',
        in_channels=768,
        out_channels=64,
        num_deconv_layers=2,
        num_deconv_filters=(256, 256),
        num_deconv_kernels=(4, 4),
        extra=dict(
            channels=64,
            activation='silu',
            neck_type='res',
            head_kernel=3,
            num_keypoints=[
                channel_cfg['num_output_channels'],
                14,
                16,
                17,
                17,
                133,
            ],
            num_in_class_proto=3,
            gamma=0.999,
            loss_proto=dict(
                type='PixelPrototypeCELoss',
                cfg=dict(
                    weight=1.25e-5,
                    ppc_weight=0.01,
                    ppd_weight=0.001,
                    num_joints=214,
                    num_in_class_proto=3,
                ),
            ),
            phm_loss_weight=3.33e-6,
            cps_weight=1.0e-2,
            cp_size=96,
            css_weight=1.0e-3,
            css_conf_thr=0.25,
            css_update_weight=3.13e-9,
            css_match_dist=2.1,
        ),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True,
    ),
)

student_model = dict(
    type='TopDown',
    pretrained=None,
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(4, 4), num_channels=(32, 64)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC', num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC', num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256)),
        ),
    ),
    keypoint_head=dict(
        type='TopdownHeatmapSimpleHead',
        in_channels=32,
        out_channels=channel_cfg['num_output_channels'],
        num_deconv_layers=0,
        extra=dict(final_conv_kernel=1),
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=False,
        target_type=target_type,
        modulate_kernel=11,
        use_udp=True,
    ),
)

model = dict(
    type='TopDownDistillOnly',
    teacher=teacher_model,
    student=student_model,
    student_init_ckpt='',
    distill_cfg=dict(
        heatmap_loss_weight=1.8,
        proto_loss_weight=1.0,
        kd_hm_weight=0.1,
        kd_proto_weight=0.15,
        temperature=1.0,
        proto_adaptor_out_channels=64,
        sup_ratio_enforce_iters=100,
        min_sup_ratio=0.9,
    ),
)

custom_hooks = [
    dict(
        type='EarlyStopByMetricHook',
        monitor='AP',
        min_delta=0.001,
        patience=2,
        interval=10,
        priority='LOWEST',
    ),
]

work_dir = 'experiments/DIST/work_dirs/hrnet_w32_distill_only_from_pruned_coco_256x192'
