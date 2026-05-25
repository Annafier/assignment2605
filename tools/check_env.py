"""Quick environment check — run with: conda activate atrumod && python tools/check_env.py"""
import sys
sys.path.insert(0, '.')

import atrumod.ops  # noqa: F401 — monkey-patch mmcv.ops before mmrotate

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
print("[OK] ATRUMODDataset")

from atrumod.models.backbones.ts_resnet import TwoStreamResNet
print("[OK] TwoStreamResNet")

from atrumod.models.backbones.c2former_resnet import C2FormerResNet
print("[OK] C2FormerResNet")

from atrumod.models.detectors.two_stream_detector import TwoStreamDetector, SingleStreamDetector
print("[OK] TwoStreamDetector, SingleStreamDetector")

from atrumod.models.heads.rotated_retina_head import RotatedRetinaHead
print("[OK] RotatedRetinaHead (self-contained)")

from atrumod.models.heads.rotated_anchor_generator import RotatedAnchorGenerator
print("[OK] RotatedAnchorGenerator")

from atrumod.models.heads.rotated_bbox_coder import DeltaXYWHAOBBoxCoder
print("[OK] DeltaXYWHAOBBoxCoder")

from atrumod.models.layers.dmm.rgbtmamba import DCFModule, MTAttentionBlock
print("[OK] DCFModule, MTAttentionBlock")

print("\n=== Config Build Test ===")
from mmengine.config import Config
cfg = Config.fromfile("configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py")
print(f"[OK] RGB baseline config: model={cfg.model['type']}")

print("\n=== Model Build Test ===")
from mmengine.registry import MODELS
model = MODELS.build(cfg.model)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[OK] Model built: {params:.1f}M params")

print("\n=== Forward Pass Test (CPU) ===")
model.eval()
dummy_rgb = torch.randn(2, 3, 512, 640)
with torch.no_grad():
    feats = model.extract_feat(dummy_rgb)
print(f"[OK] Forward pass: {[tuple(f.shape) for f in feats]}")

print("\nALL CHECKS PASSED — codebase ready!")
