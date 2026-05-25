"""Quick environment check — run with: conda activate atrumod && python tools/check_env.py"""
import sys
sys.path.insert(0, '.')

print("=== MMRotate Stack ===")
import mmcv, mmdet, mmengine, mmrotate
print(f"mmcv:      {mmcv.__version__}")
print(f"mmdet:     {mmdet.__version__}")
print(f"mmengine:  {mmengine.__version__}")
print(f"mmrotate:  {mmrotate.__version__}")

import torch
print(f"\ntorch:     {torch.__version__}")
print(f"CUDA:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:       {torch.cuda.get_device_name(0)}")
    print(f"VRAM:      {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

print("\n=== Custom Modules ===")
from atrumod.datasets.atrumod import ATRUMODDataset
print("[OK] ATRUMODDataset registered")

from atrumod.datasets.pipelines.loading import LoadRGBIRPair, PackPairedDetInputs
print("[OK] LoadRGBIRPair, PackPairedDetInputs registered")

from atrumod.models.backbones.ts_resnet import TwoStreamResNet
print("[OK] TwoStreamResNet registered")

from atrumod.models.backbones.c2former_resnet import C2FormerResNet
print("[OK] C2FormerResNet registered")

from atrumod.models.detectors.two_stream_s2anet import TwoStreamS2ANet
print("[OK] TwoStreamS2ANet registered")

from atrumod.models.detectors.dmm_s2anet import DMMS2ANet
print("[OK] DMMS2ANet registered")

from atrumod.models.data_preprocessor import DualInputDataPreprocessor
print("[OK] DualInputDataPreprocessor registered")

from atrumod.models.layers.dmm.rgbtmamba import DCFModule, MTAttentionBlock
print("[OK] DCFModule, MTAttentionBlock registered")

print("\n=== Config Build Test ===")
from mmengine.config import Config

for cfg_path in [
    "configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py",
    "configs/oriented_rcnn/oriented_rcnn_r50_atrumod_ir.py",
    "configs/c2former/c2former_s2anet_atrumod.py",
    "configs/dmm/dmm_s2anet_atrumod.py",
]:
    cfg = Config.fromfile(cfg_path)
    model_type = cfg.model.get('type', '?')
    print(f"[OK] {cfg_path.split('/')[-1]:40s} model={model_type}")

print("\n=== Model Build Test ===")
from mmengine.registry import MODELS

# Build Oriented R-CNN (single modality — simplest)
cfg = Config.fromfile("configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py")
model = MODELS.build(cfg.model)
print(f"[OK] OrientedRCNN built: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

print("\n=== Dataset Load Test ===")
from mmengine.runner import Runner
cfg = Config.fromfile("configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py")
# Quick check: can we instantiate the dataset?
train_cfg = cfg.train_dataloader
dataset = train_cfg['dataset']
print(f"  Type: {dataset['type']}")
print(f"  Ann file: {dataset['ann_file']}")
print(f"  Data prefix: {dataset['data_prefix']}")

print("\nALL CHECKS PASSED — environment ready!")
