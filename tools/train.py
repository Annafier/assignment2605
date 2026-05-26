"""Training entry point — pure PyTorch, zero mm dependencies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from atrumod.engine.config import load_config
from atrumod.engine.trainer import Trainer
from atrumod.engine.checkpoint import load_checkpoint
from atrumod.datasets.dota_dataset import DOTADataset, DualInputDataset, collate_fn, dual_collate_fn
from atrumod.models.heads.rotated_retina_head import RotatedRetinaHead
from atrumod.models.necks.fpn import SimpleFPN


def build_backbone(cfg):
    """Build backbone from config. Supports torchvision ResNet, TwoStream, C2Former."""
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
        return TwoStreamResNet(depth=bb_cfg.get('depth', 50),
                               out_indices=bb_cfg.get('out_indices', (0, 1, 2, 3)))

    elif bb_type == 'c2former':
        from atrumod.models.backbones.c2former_resnet import C2FormerResNet
        return C2FormerResNet(depth=bb_cfg.get('depth', 50),
                              out_indices=bb_cfg.get('out_indices', (0, 1, 2, 3)))

    raise ValueError(f'Unknown backbone type: {bb_type}')


class ResNetFeatureWrapper(torch.nn.Module):
    """Extract intermediate features from torchvision ResNet."""
    def __init__(self, backbone, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.backbone = backbone
        self.out_indices = out_indices

    def forward(self, x):
        feats = []
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        layers = [self.backbone.layer1, self.backbone.layer2,
                  self.backbone.layer3, self.backbone.layer4]
        for i, layer in enumerate(layers):
            x = layer(x)
            if i in self.out_indices:
                feats.append(x)
        return feats


def build_model(cfg):
    """Build detection model from config — single, dual, or DMM."""
    bb_type = cfg.backbone.get('type', 'resnet')
    neck = None
    if cfg.get('neck'):
        neck = SimpleFPN(**cfg.neck)

    if bb_type == 'dmm':
        # DMM: dual backbones + Mamba fusion + S2ANet head
        from atrumod.models.heads.s2anet_head import S2ANetHead
        from atrumod.models.detectors.dmm_detector import DMMDetector

        backbone_vi = build_backbone_plain(cfg.backbone_vi)
        backbone_ir = build_backbone_plain(cfg.backbone_ir)
        head = S2ANetHead(**cfg.head)

        fusblock, mtablock = None, None
        if cfg.get('fusblock'):
            from atrumod.models.layers.dmm.rgbtmamba import DCFModule
            fusblock = DCFModule(**cfg.fusblock)
        if cfg.get('mtablock'):
            from atrumod.models.layers.dmm.rgbtmamba import MTAttentionBlock
            mtablock = MTAttentionBlock(**cfg.mtablock)

        return DMMDetector(backbone_vi, backbone_ir, neck, fusblock, mtablock, head)

    else:
        from atrumod.models.heads.rotated_retina_head import RotatedRetinaHead
        head = RotatedRetinaHead(**cfg.head)
        backbone = build_backbone(cfg)

        if bb_type in ('two_stream', 'c2former'):
            from atrumod.models.detectors.pure_detectors import TwoStreamDetector
            return TwoStreamDetector(backbone, neck, head)
        else:
            from atrumod.models.detectors.pure_detectors import SingleStreamDetector
            return SingleStreamDetector(backbone, neck, head)


def build_backbone_plain(bb_cfg):
    """Build a single-stream backbone from config dict."""
    import torchvision.models as tv_models
    depth = bb_cfg.get('depth', 50)
    pretrained = bb_cfg.get('pretrained', True)
    weights = 'DEFAULT' if pretrained else None
    if depth == 50:
        b = tv_models.resnet50(weights=weights)
    elif depth == 101:
        b = tv_models.resnet101(weights=weights)
    else:
        raise ValueError(f'Unknown depth: {depth}')
    b.fc = torch.nn.Identity()
    return ResNetFeatureWrapper(b, out_indices=bb_cfg.get('out_indices', (0, 1, 2, 3)))


def _build_dataloader(cfg, training):
    """Build DataLoader for single or dual modality."""
    is_dual = cfg.get('train_img_ir') is not None if training else cfg.get('val_img_ir') is not None
    if is_dual:
        dataset = DualInputDataset(
            data_root=cfg.data_root,
            ann_dir=cfg.train_ann if training else cfg.val_ann,
            img_dir_rgb=cfg.train_img if training else cfg.val_img,
            img_dir_ir=cfg.train_img_ir if training else cfg.val_img_ir,
            training=training,
        )
        collate = dual_collate_fn
    else:
        dataset = DOTADataset(
            data_root=cfg.data_root,
            ann_dir=cfg.train_ann if training else cfg.val_ann,
            img_dir=cfg.train_img if training else cfg.val_img,
            training=training,
        )
        collate = collate_fn

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=training,
        num_workers=cfg.num_workers if training else min(cfg.num_workers, 2),
        collate_fn=collate,
        pin_memory=True,
    )


def main():
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/rgb_baseline.py'
    cfg = load_config(cfg_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = _build_dataloader(cfg, training=True)
    val_loader = None
    if cfg.get('val_ann'):
        val_loader = _build_dataloader(cfg, training=False)

    model = build_model(cfg)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Model: {params:.1f}M params')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get('lr', 0.01),
        momentum=cfg.get('momentum', 0.9),
        weight_decay=cfg.get('weight_decay', 0.0001),
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.get('lr_milestones', [8, 11]),
        gamma=cfg.get('lr_gamma', 0.1),
    )

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
