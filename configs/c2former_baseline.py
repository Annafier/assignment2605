"""C2Former RGB+IR multimodal config — ICA cross-attention + AFS fusion."""

# Data
data_root = 'data'
train_ann = 'train/dota_labels'
train_img = 'train/images'
train_img_ir = 'train/images_ir'
val_ann = 'val/dota_labels'
val_img = 'val/images'
val_img_ir = 'val/images_ir'

# Training
batch_size = 2  # C2Former uses more VRAM (cross-attention)
num_workers = 8
max_epochs = 12
lr = 0.01
momentum = 0.9
weight_decay = 0.0001
lr_milestones = [8, 11]
lr_gamma = 0.1
use_amp = True
log_interval = 5
grad_clip = 35.0
work_dir = 'logs/checkpoints/c2former_baseline'

# C2Former backbone — ICA cross-attention fusion
backbone = dict(
    type='c2former',
    depth=50,
    out_indices=(0, 1, 2, 3),
)

# Neck (shared FPN on fused features)
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
