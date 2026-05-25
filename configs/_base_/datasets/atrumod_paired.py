"""
ATR-UMOD paired RGB-IR dataset config for multimodal training.
Loads both RGB and IR images, supports dual-stream backbone input.
"""
dataset_type = 'ATRUMODDataset'
data_root = 'data/'

backend_args = None

train_pipeline = [
    dict(type='LoadRGBIRPair', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(640, 512), keep_ratio=True),
    dict(
        type='mmdet.RandomRotate',
        prob=0.5,
        angle_range=180,
        rect_obj_labels=['car', 'suv', 'van', 'bus', 'freight_car', 'truck',
                         'motorcycle', 'trailer', 'tank_truck', 'excavator', 'crane']),
    dict(type='mmdet.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackPairedDetInputs'),
]

val_pipeline = [
    dict(type='LoadRGBIRPair', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(640, 512), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='PackPairedDetInputs',
         meta_keys=('img_id', 'img_path', 'ir_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/dota_labels/',
        data_prefix=dict(img_path='train/images/', ir_path='train/images_ir/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/dota_labels/',
        data_prefix=dict(img_path='val/images/', ir_path='val/images_ir/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))

val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.5])
