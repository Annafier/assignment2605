"""
Inference and evaluation script for ATR-UMOD.
Runs detection on val set and computes rotated mAP.
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector on ATR-UMOD')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file (pickle)')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', help='directory to save visualization images')
    return parser.parse_args()


def main():
    args = parse_args()

    from atrumod.datasets.atrumod import ATRUMODDataset  # noqa: F401
    from atrumod.datasets.pipelines.loading import LoadRGBIRPair, PackPairedDetInputs  # noqa: F401
    from atrumod.models.backbones.c2former_resnet import C2FormerResNet  # noqa: F401
    from atrumod.models.backbones.ts_resnet import TwoStreamResNet  # noqa: F401
    from atrumod.models.detectors.two_stream_s2anet import TwoStreamS2ANet  # noqa: F401
    from atrumod.models.data_preprocessor import DualInputDataPreprocessor  # noqa: F401

    cfg = Config.fromfile(args.config)
    cfg.load_from = args.checkpoint
    cfg.work_dir = str(Path(args.checkpoint).parent)

    runner = Runner.from_cfg(cfg)
    runner.test()


if __name__ == '__main__':
    main()
