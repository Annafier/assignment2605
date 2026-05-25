# Oriented R-CNN — IR-only baseline on infrared images
_base_ = './oriented_rcnn_r50_atrumod.py'

# Override data root to use IR images
train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='train/images_ir/'),
        ann_file='train/dota_labels_ir/'))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(img_path='val/images_ir/'),
        ann_file='val/dota_labels_ir/'))

test_dataloader = val_dataloader

work_dir = 'logs/checkpoints/oriented_rcnn_r50_atrumod_ir'
