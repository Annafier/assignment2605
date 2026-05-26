"""DMM (Disparity-guided Multispectral Mamba) config — pure Python."""

# Data
data_root = 'data'
train_ann = 'train/dota_labels'
train_img = 'train/images'
train_img_ir = 'train/images_ir'
val_ann = 'val/dota_labels'
val_img = 'val/images'
val_img_ir = 'val/images_ir'

# Training
batch_size = 1
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
work_dir = 'logs/checkpoints/dmm_baseline'

# Dual ResNet-50 backbones (both ImageNet pretrained; IR is 3-channel too)
backbone = dict(type='dmm')

backbone_vi = dict(depth=50, out_indices=(0, 1, 2, 3), pretrained=True)
backbone_ir = dict(depth=50, out_indices=(0, 1, 2, 3), pretrained=True)

# DMM fusion modules
mtablock = dict(
    channels_per_scale=[256, 512, 1024, 2048],
)

fusblock = dict(
    channels=[256, 512, 1024, 2048],
    d_state=16,
    d_conv=3,
    expand=2,
    dropout=0.0,
    drop_path=0.0,
)

# Neck (FPN on fused features)
neck = dict(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    start_level=0,
    num_outs=5,
)

# S2ANet head
head = dict(
    num_classes=11,
    in_channels=256,
    feat_channels=256,
    stacked_convs=2,
    align_conv_strides=[8, 16, 32, 64, 128],
    align_conv_kernel=3,
    anchor_generator=dict(
        strides=[8, 16, 32, 64, 128],
        scales=[4],
        ratios=[1.0],
        angles=[0],
    ),
    bbox_coder=dict(angle_range='le135'),
    loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
    loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
)
