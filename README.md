# ATR-UMOD — Multimodal Oriented Object Detection

Training & evaluation codebase for UAV-based RGB-IR oriented object detection on the [ATR-UMOD](https://github.com/user-attachments/assets/20679e73-8f60-497e-a086-3b93d932579f) dataset (ICCV 2025). Built on [MMRotate](https://github.com/open-mmlab/mmrotate).

## Dataset

11,850 aligned RGB-IR image pairs at 640×512 resolution. 11 vehicle classes annotated with **rotated bounding boxes** (cx, cy, w, h, angle) plus polygon corners. Each pair carries 6 condition attributes: UAV altitude, camera angle, weather, illumination, time, and scene location.

| Class | car | SUV | van | bus | freight car | truck | motorcycle | trailer | tank truck | excavator | crane |
|---|-----|-----|-----|-----|---|-----|---|---|---|---|---|
| Count | 62k | 33k | 16k | 9k | 8k | 8k | 3k | 2k | 2k | 1k | 1k |

### Train / Val Split

Split by **scene location** to prevent same-scene leakage between train and validation:

| Split | Locations | Pairs | Scenes |
|-------|-----------|-------|--------|
| Train | 0, 1, 3, 5 | 9,921 | Roads, neighborhoods, factories (urban villages & suburbs) |
| Val | 2, 4, 6, 7, 8, 9, 10 | 1,929 | Schools, parking lots, construction sites, urban center |

## Environment

### Option A — Transfer Pack (exact copy, 2.6 GB archive)

For RTX 50-series GPUs. Unpack on the target machine:

```bash
mkdir -p ~/miniconda3/envs/atrumod
tar -xzf atrumod_env.tar.gz -C ~/miniconda3/envs/atrumod
cp transfer/mmrotate_init_patched.py \
  $(~/miniconda3/envs/atrumod/python -c "import mmrotate; print(mmrotate.__path__[0])")/__init__.py
conda activate ~/miniconda3/envs/atrumod
pip install -e .
```

### Option B — Fresh Install

Requires Python ≥ 3.10, CUDA toolkit, and conda.

```bash
conda create -n atrumod python=3.10 -y
conda activate atrumod

# PyTorch — match your CUDA version (cu121 / cu124 / cu128)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# MMRotate stack
pip install mmengine mmdet mmrotate
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# Patch mmrotate version check (needed for mmcv ≥ 2.2.0)
cp transfer/mmrotate_init_patched.py \
  $(python -c "import mmrotate; print(mmrotate.__path__[0])")/__init__.py

# Project
pip install -e .
pip install tensorboard einops
```

For RTX 50-series (Blackwell) GPUs, use PyTorch ≥ 2.7 with CUDA 12.8 — see `transfer_setup.sh` for the exact commands.

## Quick Start

```bash
# Verify installation
python tools/check_env.py

# RGB baseline (fastest, ~6h on RTX 5060)
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py

# IR baseline
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_atrumod_ir.py

# C2Former multimodal
python tools/train.py configs/c2former/c2former_s2anet_atrumod.py

# DMM multimodal
python tools/train.py configs/dmm/dmm_s2anet_atrumod.py

# Monitor training
tensorboard --logdir logs/tensorboard
```

## Project Structure

```
├── atrumod/                         # Custom package (19 Python modules)
│   ├── datasets/
│   │   ├── atrumod.py               # ATRUMODDataset (MMRotate registry)
│   │   └── pipelines/loading.py     # LoadRGBIRPair, PackPairedDetInputs
│   └── models/
│       ├── backbones/
│       │   ├── resnet_blocks.py     # BasicBlock, Bottleneck (standalone)
│       │   ├── ts_resnet.py         # TwoStreamResNet (dual RGB+IR stems)
│       │   └── c2former_resnet.py   # C2Former: ICA cross-attention + AFS
│       ├── detectors/
│       │   ├── two_stream_s2anet.py # TwoStreamS2ANet (C2Former detector)
│       │   └── dmm_s2anet.py        # DMMS2ANet (Mamba-based detector)
│       ├── layers/dmm/
│       │   ├── rgbtmamba.py         # DCFModule, DSSM, MTAttentionBlock
│       │   └── ssm_utils.py         # CrossScan/CrossMerge utilities
│       └── data_preprocessor.py     # DualInputDataPreprocessor
├── configs/
│   ├── _base_/                      # Shared dataset, schedule, runtime
│   ├── oriented_rcnn/               # RGB & IR single-modality baselines
│   ├── c2former/                    # C2Former + S2ANet
│   └── dmm/                         # DMM + S2ANet
├── tools/
│   ├── train.py                     # Training entry point
│   ├── test.py                      # Inference & DOTA evaluation
│   ├── check_env.py                 # Environment verification
│   ├── convert_labels.py            # XML → DOTA format
│   └── visualize.py                 # RGB+IR viewer with ground truth
├── data/                            # Train: 9,921 / Val: 1,929
├── transfer/                        # Environment transfer files
└── logs/                            # TensorBoard + checkpoints per run
```

## Implemented Methods

| # | Method | Config | Venue | Fusion | Backbone |
|---|--------|--------|-------|--------|----------|
| 1 | Oriented R-CNN (RGB) | `configs/oriented_rcnn/oriented_rcnn_r50_atrumod.py` | ICCV 2021 | None | ResNet-50 |
| 2 | Oriented R-CNN (IR) | `configs/oriented_rcnn/oriented_rcnn_r50_atrumod_ir.py` | — | None | ResNet-50 |
| 3 | **C2Former + S2ANet** | `configs/c2former/c2former_s2anet_atrumod.py` | TGRS 2024 | ICA cross-attention + AFS | Dual ResNet-50 |
| 4 | **DMM + S2ANet** | `configs/dmm/dmm_s2anet_atrumod.py` | TGRS 2025 | DCFM Mamba + MTAttention | Dual ResNet-50 |

## Evaluation

```bash
python tools/test.py configs/c2former/c2former_s2anet_atrumod.py \
    logs/checkpoints/c2former_s2anet_atrumod/epoch_12.pth
```

Metrics reported: **rotated mAP@0.5** (DOTA evaluation protocol) over 11 vehicle classes.

## Attribution

Dataset — *"Fusion Meets Diverse Conditions: A High-diversity Benchmark and Baseline for UAV-based Multimodal Object Detection with Condition Cues"*, ICCV 2025.

C2Former — *"C2Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection"*, TGRS 2024.

DMM — *"Disparity-guided Multispectral Mamba for RGB-T Object Detection"*, TGRS 2025.
