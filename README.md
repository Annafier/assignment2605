# ATR-UMOD — Multimodal Oriented Object Detection

End-to-end training & evaluation codebase for UAV-based RGB-IR oriented object detection on the [ATR-UMOD](https://github.com/user-attachments/assets/20679e73-8f60-497e-a086-3b93d932579f) dataset (ICCV 2025). Built on **MMEngine + MMDetection** with **zero runtime mmrotate dependency** — all rotated detection ops and heads are self-contained pure PyTorch.

## Dataset

11,850 aligned RGB-IR image pairs at 640×512 resolution. 11 vehicle classes with **rotated bounding boxes** (cx, cy, w, h, angle). Each pair carries 6 condition attributes: UAV altitude, camera angle, weather, illumination, time, scene location.

| Class | car | SUV | van | bus | freight car | truck | motorcycle | trailer | tank truck | excavator | crane |
|---|-----|-----|-----|-----|---|-----|---|---|---|---|---|
| Count | 62k | 33k | 16k | 9k | 8k | 8k | 3k | 2k | 2k | 1k | 1k |

### Train / Val Split

Split by **scene location** — no same-scene leakage:

| Split | Locations | Pairs |
|-------|-----------|-------|
| Train | 0, 1, 3, 5 | 9,921 |
| Val | 2, 4, 6, 7, 8, 9, 10 | 1,929 |

## Setup

### Fresh Install

```bash
conda create -n atrumod python=3.10 -y && conda activate atrumod

# PyTorch — cu121 for pre-Blackwell, cu128 for RTX 50-series
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Base stack
pip install mmengine mmdet mmcv==2.2.0 mmrotate==1.0.0rc1
pip install tensorboard einops

# Patch version checks (mmcv 2.2 + mmdet 3.3 with mmrotate 1.0.0rc1)
python -c "
import mmdet; p=mmdet.__file__.replace('__init__.py','__init__.py')
f=open(p); c=f.read(); f.close()
c=c.replace(\"mmcv_maximum_version = '2.2.0'\",\"mmcv_maximum_version = '2.3.0'\")
open(p,'w').write(c)
"

# Install project
pip install -e .
```

### Verify

```bash
python tools/check_env.py
```

## Quick Start

```bash
# 1. RGB baseline (~6 hours on RTX 5060)
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py

# 2. IR baseline
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_atrumod_ir.py

# 3. C2Former multimodal
python tools/train.py configs/c2former/c2former_s2anet_atrumod.py

# 4. Monitor training
tensorboard --logdir logs/tensorboard
```

## Architecture

```
ATR-UMOD/atrumod/
├── ops/                        # Pure-PyTorch rotated ops (no CUDA compiler needed)
│   ├── rotated_iou.py          # box_iou_rotated — Sutherland-Hodgman clipping
│   ├── rotated_nms.py          # nms_rotated, batched_nms
│   ├── active_rotated_filter.py # ORConv kernel (rotation via grid_sample)
│   └── deform_conv.py          # DeformConv2d (torchvision delegate)
│
├── models/
│   ├── heads/                  # Self-contained rotated detection heads
│   │   ├── rotated_retina_head.py       # Full detection head with loss + inference
│   │   ├── rotated_anchor_generator.py  # Multi-scale rotated anchor generation
│   │   └── rotated_bbox_coder.py        # DeltaXYWHA encode/decode
│   │
│   ├── detectors/
│   │   ├── two_stream_detector.py  # SingleStreamDetector + TwoStreamDetector
│   │   ├── two_stream_s2anet.py    # C2Former S2ANet detector (legacy)
│   │   └── dmm_s2anet.py           # DMM Mamba-based detector
│   │
│   ├── backbones/
│   │   ├── ts_resnet.py         # TwoStreamResNet (dual RGB+IR stems)
│   │   ├── c2former_resnet.py   # C2Former: ICA cross-attention + AFS (TGRS 2024)
│   │   └── resnet_blocks.py     # Standalone BasicBlock / Bottleneck
│   │
│   └── layers/dmm/
│       ├── rgbtmamba.py    # DSSM, DCFM, DCFModule, MTAttentionBlock (TGRS 2025)
│       └── ssm_utils.py    # CrossScan/CrossMerge for RGB-T Mamba
│
├── datasets/
│   ├── atrumod.py            # ATRUMODDataset
│   └── pipelines/loading.py  # LoadRGBIRPair, PackPairedDetInputs
│
configs/
├── oriented_rcnn/            # RGB & IR single-modality baselines
├── c2former/                 # C2Former + S2ANet multimodal
└── dmm/                      # DMM + S2ANet multimodal
```

## Implemented Methods

| # | Method | Config | Venue | Fusion |
|---|--------|--------|-------|--------|
| 1 | RetinaNet-OBB (RGB) | `configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py` | — | None |
| 2 | RetinaNet-OBB (IR) | `configs/oriented_rcnn/oriented_rcnn_r50_atrumod_ir.py` | — | None |
| 3 | C2Former + S2ANet | `configs/c2former/c2former_s2anet_atrumod.py` | TGRS 2024 | ICA cross-attention + AFS |
| 4 | DMM + S2ANet | `configs/dmm/dmm_s2anet_atrumod.py` | TGRS 2025 | DCFM Mamba + MTAttention |

All heads use self-contained pure-PyTorch rotated ops — works on Blackwell (RTX 50-series), no CUDA extension compilation required.

## Evaluation

```bash
python tools/test.py configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py \
    logs/checkpoints/rgb_baseline/epoch_12.pth
```

Metrics: rotated mAP@0.5 (DOTA protocol), 11 classes.

## Tools

| Script | Purpose |
|--------|---------|
| `tools/train.py` | Training entry (MMEngine Runner, TensorBoard logging) |
| `tools/test.py` | Inference + evaluation |
| `tools/check_env.py` | Full environment verification |
| `tools/convert_labels.py` | XML → DOTA label format |
| `tools/visualize.py` | RGB+IR viewer with ground truth boxes |

## Transfer

```bash
# Pack environment for another machine
conda-pack -n atrumod -o atrumod_env.tar.gz

# Unpack on target
tar -xzf atrumod_env.tar.gz -C ~/miniconda3/envs/atrumod/
cp transfer/mmrotate_init_patched.py \
  $(~/miniconda3/envs/atrumod/python -c "import mmrotate; print(mmrotate.__path__[0])")/__init__.py
```

## Attribution

- Dataset: *"Fusion Meets Diverse Conditions"*, ICCV 2025
- C2Former: *"Calibrated and Complementary Transformer for RGB-Infrared Object Detection"*, TGRS 2024 https://github.com/yuanmaoxun/C2Former.
- DMM: *"Disparity-guided Multispectral Mamba for RGB-T Object Detection"*, TGRS 2025 https://github.com/Another-0/DMM
