# DMM + S2ANet multimodal detector for ATR-UMOD
# Disparity-guided Multispectral Mamba (TGRS 2025)
#
# NOTE: Full DMM uses VMamba backbones (requires selective_scan_cuda).
# This config uses ResNet-50 as a practical fallback.
# To use VMamba: install https://github.com/MzeroMiko/VMamba and swap backbone types.
_base_ = [
    '../_base_/datasets/atrumod_paired.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

angle_version = 'le135'
num_classes = 11

# DCFModule channels — match backbone output channels
# ResNet-50: [256, 512, 1024, 2048]
# VMamba-Tiny: [96, 192, 384, 768]
fusblock_channels = [256, 512, 1024, 2048]

model = dict(
    type='DMMS2ANet',
    data_preprocessor=dict(
        type='DualInputDataPreprocessor',
        mean_rgb=[123.675, 116.28, 103.53],
        std_rgb=[58.395, 57.12, 57.375],
        mean_ir=[123.675, 116.28, 103.53],
        std_ir=[58.395, 57.12, 57.375],
        pad_size_divisor=32),
    backbone_vi=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    backbone_ir=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    # DMM cross-modal fusion
    fusblock=dict(
        type='DCFModule',
        channels=fusblock_channels,
        d_state=16,
        d_conv=3,
        expand=2,
        dropout=0.0,
        drop_path=0.1),
    # Multi-scale target-aware attention on VI
    mtablock=dict(
        type='MTAttentionBlock',
        channels_per_scale=fusblock_channels),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head_init=dict(
        type='mmrotate.RotatedRetinaHead',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        anchor_generator=dict(
            type='mmrotate.RotatedAnchorGenerator',
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='mmrotate.DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    bbox_head_refine=[
        dict(
            type='mmrotate.S2ARefineHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            anchor_generator=dict(
                type='mmdet.PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='mmrotate.DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=1,
                edge_swap=False,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='mmdet.SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    ],
    train_cfg=dict(
        fam_cfg=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmrotate.RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        odm_cfg=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='mmrotate.RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))

work_dir = 'logs/checkpoints/dmm_s2anet_atrumod'
