# ATR-UMOD Multimodal Oriented Object Detection Codebase

## Goal
Build a training + evaluation pipeline for RGB-IR oriented (rotated) object detection on the ATR-UMOD dataset, using MMRotate framework. Target: class competition with hidden test set.

## Dataset
- 11,850 RGB-IR paired images, 640x512, 11 vehicle classes
- Labels: XML with rotated bounding boxes (robndbox: cx, cy, w, h, angle) + polygon corners
- 6 condition attributes per pair: UAV angle, height, weather, illumination, time, location
- Train/val split by location/scenario: train on locations {0,1,3,5} (9,921 pairs), val on {2,4,6,7,8,9,10} (1,929 pairs)

## Architecture

### Framework
MMRotate (MMEngine + MMCV + MMDetection ecosystem). Config-driven, standard rotated detection pipeline.

### Baselines (ordered by implementation priority)
1. **Oriented R-CNN (ResNet-50)** — single-modality RGB baseline, no fusion. Proves the data pipeline works.
2. **C2Former + S2ANet** — Transformer cross-attention fusion (ICA + AFS), open-source, plug-and-play. Primary multimodal baseline.
3. **DMM + S2ANet** — Mamba-based fusion (DCFM + MTA), higher ceiling. Advanced.

### Data Format
Convert XML labels to DOTA format (the MMRotate standard for rotated detection). DOTA uses: `x1 y1 x2 y2 x3 y3 x4 y4 classname difficult`.

### Fusion Strategy
Mid-fusion via dual-stream backbone. RGB and IR go through separate stem + early layers, then fuse at intermediate stages via cross-attention (C2Former) or Mamba selective scan (DMM). Fused features feed standard FPN neck + rotated detection head.

### Logging
TensorBoard for all training metrics (loss curves, mAP progression). Logs stored in `logs/tensorboard/` per experiment. Checkpoints in `logs/checkpoints/`.

## Project Structure
```
ATR-UMOD/
├── data/train/  data/val/           # Split dataset
├── configs/                         # MMRotate config files
│   ├── _base_/datasets/atrumod.py
│   ├── _base_/schedules/
│   └── c2former/  dmm/  oriented_rcnn/
├── atrumod/                         # Custom package
│   ├── datasets/                    # ATRUMODDataset, XML→DOTA converter
│   ├── models/                      # Custom backbones, necks, detectors
│   └── evaluation/
├── tools/                           # train.py, test.py, convert_labels.py
├── logs/tensorboard/  logs/checkpoints/
├── requirements.txt
└── setup.py
```

## Success Criteria
- [ ] XML → DOTA conversion with all 11 classes
- [ ] Oriented R-CNN baseline training converges, val mAP > 0.5
- [ ] C2Former multimodal training: val mAP exceeds single-modality baseline
- [ ] TensorBoard logs capture loss, mAP per epoch
- [ ] Inference script outputs DOTA-format predictions for test set
