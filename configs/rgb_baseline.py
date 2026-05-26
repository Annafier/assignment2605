"""RGB baseline config — pure Python, no mm dependency."""

# Data
data_root = 'data'
train_ann = 'train/dota_labels'
train_img = 'train/images'
val_ann = 'val/dota_labels'
val_img = 'val/images'

# Training
batch_size = 8
num_workers = 8
max_epochs = 12
lr = 0.005
momentum = 0.9
weight_decay = 0.0001
lr_milestones = [8, 11]
lr_gamma = 0.1
use_amp = True
log_interval = 5
work_dir = 'logs/checkpoints/rgb_baseline'

# Backbone — ResNet-50
backbone = dict(
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
)

# Neck
neck = dict(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    start_level=0,
    num_outs=5,
)

# Head
head = dict(
    num_classes=11,
    in_channels=256,
    stacked_convs=4,
    feat_channels=256,
    anchor_generator=dict(
        strides=[8, 16, 32, 64, 128],
        scales=[4],
        ratios=[1.0],
        angles=[0],
    ),
    bbox_coder=dict(
        angle_range='le135',
        target_means=(0, 0, 0, 0, 0),
        target_stds=(1, 1, 1, 1, 1),
    ),
    loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
)
