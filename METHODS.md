# Detection Methods — ATR-UMOD

## Status Legend
- ✅ Pure PyTorch, tested, training verified
- 🔧 Pure PyTorch, tested (forward+backward)
- ⏳ Not yet ported (mm-dependent)

---

## 1. RGB Baseline (RetinaNet)

| | |
|---|---|
| **Status** | ✅ Fully working |
| **Backbone** | ResNet-50 (torchvision, ImageNet pretrained) |
| **Neck** | SimpleFPN |
| **Head** | RotatedRetinaHead (Focal Loss + SmoothL1) |
| **Config** | `configs/rgb_baseline.py` |
| **Params** | 32.2M |
| **Batch time** | ~0.45s (batch=8) |
| **Reference** | [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Lin et al., ICCV 2017 |

Single-modality RGB oriented object detection. Anchor-based RetinaNet architecture with rotated box regression (cx, cy, w, h, angle). Uses Focal Loss for class imbalance and SmoothL1 for bounding box regression.

---

## 2. TwoStream ResNet

| | |
|---|---|
| **Status** | ✅ Fully working |
| **Backbone** | TwoStreamResNet-50 (dual RGB+IR stems, from scratch) |
| **Fusion** | Element-wise sum `vis_x + lwir_x` |
| **Neck** | SimpleFPN (shared) |
| **Head** | RotatedRetinaHead |
| **Config** | `configs/two_stream_baseline.py` |
| **Params** | 55.7M |
| **Batch time** | ~0.29s (batch=4) |
| **Reference** | [C²Former](https://arxiv.org/abs/2306.16175) — Yuan & Wei, IEEE TGRS 2024 |

Dual-stream ResNet-50 with independent RGB and IR branches. Each modality has its own stem + 4 stages. Features are fused via simple element-wise addition after each stage. Serves as the base architecture for C²Former.

**Note:** No pretrained weights available for the IR stream; converges slower than RGB baseline.

---

## 3. C²Former (Calibrated and Complementary Transformer)

| | |
|---|---|
| **Status** | 🔧 Forward+backward verified, training test passed |
| **Backbone** | C2FormerResNet-50 (extends TwoStreamResNet) |
| **Fusion** | ICA (Inter-modality Cross-Attention) + AFS (Adaptive Feature Sampling) |
| **Neck** | SimpleFPN (shared) |
| **Head** | RotatedRetinaHead |
| **Config** | `configs/c2former_baseline.py` |
| **Params** | 114.3M |
| **Batch time** | ~3.8s (batch=2) |
| **Reference** | [C²Former: Calibrated and Complementary Transformer for RGB-Infrared Object Detection](https://arxiv.org/abs/2306.16175) — Yuan & Wei, IEEE TGRS 2024 |
| **Code** | https://github.com/yuanmaoxun/C2Former |

Key innovations:
- **ICA:** RGB queries attend to IR keys/values (bidirectional), learning cross-modal correspondence to fix sensor misalignment
- **AFS:** Deformable spatial sampling via predicted offsets, replacing dense global attention with sparse local attention
- **ModalityNorm:** IR feature statistics normalize and recalibrate RGB features (and vice versa)

**Note:** ~27× slower per sample than TwoStream due to cross-attention + deformable grid_sample. Batch=2 fits on RTX 5060 8.5GB.

---

## 4. DMM (Disparity-guided Multispectral Mamba)

| | |
|---|---|
| **Status** | ⏳ Not yet ported (mm-dependent) |
| **Backbone** | TwoStreamResNet + DCFM fusion |
| **Fusion** | DCFM (Disparity-guided Cross-modal Fusion Mamba) |
| **Head** | S2ANet (AlignDet-style FAM + ODM) |
| **Params** | TBD |
| **Reference** | [DMM: Disparity-guided Multispectral Mamba for Oriented Object Detection in Remote Sensing](https://arxiv.org/abs/2407.08132) — Zhou et al., IEEE TGRS 2025 |
| **Code** | https://github.com/Another-0/DMM |

Key innovations:
- **DCFM:** Mamba-based state-space model for cross-modal fusion, guided by RGB-IR disparity
- **MTA:** Multi-scale Target-aware Attention to enhance intra-modal features
- **TPA:** Target-Prior Aware auxiliary task for additional supervision
- First application of Mamba architecture to multispectral oriented detection

**Note:** Requires porting the S2ANet head (FAM + ODM refinement stages) and DCFM Mamba modules to pure PyTorch. The DMM-specific layers (`rgbtmamba.py`, `ssm_utils.py`) are already pure PyTorch but depend on the mm-detector wrapper.

---

## 5. S2ANet (Align Deep Features)

| | |
|---|---|
| **Status** | ⏳ Not yet ported (mm-dependent) |
| **Head** | FAM (Feature Alignment Module) + ODM (Oriented Detection Module) |
| **Reference** | [Align Deep Features for Oriented Object Detection](https://ieeexplore.ieee.org/document/9377550) — Han et al., IEEE TGRS 2022 |

Two-stage rotated detector: FAM aligns features to oriented anchors via alignment convolutions, ODM refines predictions. Used as the head for DMM. The mmrotate implementation (`s2anet.py`, `two_stream_s2anet.py`) needs porting to pure PyTorch.

---

## Shared Components (all pure PyTorch)

| Component | File |
|-----------|------|
| RotatedRetinaHead | `atrumod/models/heads/rotated_retina_head.py` |
| RotatedAnchorGenerator | `atrumod/models/heads/rotated_anchor_generator.py` |
| DeltaXYWHAOBBoxCoder | `atrumod/models/heads/rotated_bbox_coder.py` |
| SimpleFPN | `atrumod/models/necks/fpn.py` |
| Rotated IoU (vectorized) | `atrumod/ops/rotated_iou.py` |
| Rotated NMS | `atrumod/ops/rotated_nms.py` |
| DOTADataset | `atrumod/datasets/dota_dataset.py` |
| DualInputDataset | `atrumod/datasets/dota_dataset.py` |
| Trainer | `atrumod/engine/trainer.py` |
| Config loader | `atrumod/engine/config.py` |
