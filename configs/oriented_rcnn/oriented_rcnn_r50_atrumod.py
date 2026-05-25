# RGB baseline — single-stream with self-contained RotatedRetinaHead
_base_ = [
    '../_base_/datasets/atrumod.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

num_classes = 11
angle_version = 'le135'

model = dict(
    type='SingleStreamDetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        pad_size_divisor=32),
    backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RotatedRetinaHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            strides=[8, 16, 32, 64, 128],
            scales=[4],
            ratios=[1.0],
            angles=[0]),
        bbox_coder=dict(
            angle_range=angle_version,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    test_cfg=dict(
        nms_pre=2000,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

optim_wrapper = dict(optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

work_dir = 'logs/checkpoints/rgb_baseline'
