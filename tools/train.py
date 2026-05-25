"""
Training script for ATR-UMOD multimodal object detection.
Wraps MMRotate's tools/train.py with project-specific setup.
"""
import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine import ConfigDict


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector on ATR-UMOD')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='directory to save logs and checkpoints')
    parser.add_argument('--amp', action='store_true', help='enable mixed precision training')
    parser.add_argument('--resume', action='store_true', help='resume from latest checkpoint')
    parser.add_argument('--auto-scale-lr', action='store_true', help='auto-scale learning rate')
    return parser.parse_args()


def main():
    args = parse_args()

    # Register all custom modules with MMRotate registries
    from atrumod.datasets.atrumod import ATRUMODDataset  # noqa: F401
    from atrumod.datasets.pipelines.loading import LoadRGBIRPair, PackPairedDetInputs  # noqa: F401
    from atrumod.models.backbones.c2former_resnet import C2FormerResNet  # noqa: F401
    from atrumod.models.backbones.ts_resnet import TwoStreamResNet  # noqa: F401
    from atrumod.models.detectors.two_stream_s2anet import TwoStreamS2ANet  # noqa: F401
    from atrumod.models.detectors.dmm_s2anet import DMMS2ANet  # noqa: F401
    from atrumod.models.data_preprocessor import DualInputDataPreprocessor  # noqa: F401
    from atrumod.models.layers.dmm import DCFModule, MTAttentionBlock  # noqa: F401

    cfg = Config.fromfile(args.config)

    # Set work directory with tensorboard
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = f'logs/checkpoints/{Path(args.config).stem}'

    # Ensure tensorboard logging
    if hasattr(cfg, 'visualizer') and hasattr(cfg.visualizer, 'vis_backends'):
        has_tb = any('Tensorboard' in str(b.get('type', '')) for b in cfg.visualizer.vis_backends)
        if not has_tb:
            cfg.visualizer.vis_backends.append(dict(type='TensorboardVisBackend'))

    # Auto-scale LR across GPUs
    if args.auto_scale_lr:
        cfg.auto_scale_lr.enable = True

    # Mixed precision
    if args.amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    # Resume
    if args.resume:
        cfg.resume = True

    os.makedirs(cfg.work_dir, exist_ok=True)
    print(f'Work dir: {cfg.work_dir}')
    print(f'Config: {cfg.pretty_text}')

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
