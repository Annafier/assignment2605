"""Training entry point — pure PyTorch, zero mm dependencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from atrumod.engine.config import load_config
from atrumod.engine.trainer import Trainer
from atrumod.engine.checkpoint import load_checkpoint
from atrumod.datasets.dota_dataset import DOTADataset, collate_fn
from atrumod.models.heads.rotated_retina_head import RotatedRetinaHead
from atrumod.models.necks.fpn import SimpleFPN


def build_backbone(cfg):
    """Build backbone from config. Supports torchvision ResNet or our custom ones."""
    bb_cfg = cfg.backbone
    bb_type = bb_cfg.get('type', 'resnet')

    if bb_type == 'resnet':
        import torchvision.models as tv_models
        depth = bb_cfg.get('depth', 50)
        pretrained = bb_cfg.get('pretrained', True)
        weights = 'DEFAULT' if pretrained else None
        if depth == 50:
            backbone = tv_models.resnet50(weights=weights)
        elif depth == 101:
            backbone = tv_models.resnet101(weights=weights)
        else:
            raise ValueError(f'Unknown resnet depth: {depth}')
        backbone.fc = torch.nn.Identity()
        return ResNetFeatureWrapper(backbone, out_indices=bb_cfg.get('out_indices', (0, 1, 2, 3)))

    elif bb_type == 'two_stream':
        from atrumod.models.backbones.ts_resnet import TwoStreamResNet
        return TwoStreamResNet(**{k: v for k, v in bb_cfg.items() if k != 'type'})

    elif bb_type == 'c2former':
        from atrumod.models.backbones.c2former_resnet import C2FormerResNet
        return C2FormerResNet(**{k: v for k, v in bb_cfg.items() if k != 'type'})

    raise ValueError(f'Unknown backbone type: {bb_type}')


class ResNetFeatureWrapper(torch.nn.Module):
    """Extract intermediate features from torchvision ResNet."""
    def __init__(self, backbone, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.backbone = backbone
        self.out_indices = out_indices

    def forward(self, x):
        feats = []
        # stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        # layers
        layers = [self.backbone.layer1, self.backbone.layer2,
                  self.backbone.layer3, self.backbone.layer4]
        for i, layer in enumerate(layers):
            x = layer(x)
            if i in self.out_indices:
                feats.append(x)
        return feats


class DetectionModel(torch.nn.Module):
    """Detection model: backbone + neck + head. Module-level so checkpoint resume works."""
    def __init__(self, backbone, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = head

    def forward(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return self.bbox_head(feats)


def build_model(cfg):
    """Build detection model from config."""
    backbone = build_backbone(cfg)
    head = RotatedRetinaHead(**cfg.head)
    neck = None
    if cfg.get('neck'):
        neck = SimpleFPN(**cfg.neck)
    return DetectionModel(backbone, neck, head)


def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py'
    cfg = load_config(cfg_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Datasets
    train_dataset = DOTADataset(
        data_root=cfg.data_root,
        ann_dir=cfg.train_ann,
        img_dir=cfg.train_img,
        training=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_dataset = DOTADataset(
        data_root=cfg.data_root,
        ann_dir=cfg.val_ann,
        img_dir=cfg.val_img,
        training=False,
    ) if cfg.get('val_ann') else None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=min(cfg.num_workers, 2),
        collate_fn=collate_fn,
        pin_memory=True,
    ) if val_dataset else None

    # Model
    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {params:.1f}M params')

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get('lr', 0.01),
        momentum=cfg.get('momentum', 0.9),
        weight_decay=cfg.get('weight_decay', 0.0001),
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get('lr_milestones', [8, 11]),
        gamma=cfg.get('lr_gamma', 0.1),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        max_epochs=cfg.get('max_epochs', 12),
        work_dir=cfg.get('work_dir', 'logs/checkpoints/run'),
        log_interval=cfg.get('log_interval', 50),
        device=device,
        use_amp=cfg.get('use_amp', False),
        resume=cfg.get('resume', None),
        grad_clip=cfg.get('grad_clip', 35.0),
    )

    trainer.fit()


if __name__ == '__main__':
    main()
