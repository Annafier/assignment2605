"""Pure-PyTorch detector wrappers — no mmdet dependency.

Each detector is a self-contained nn.Module: backbone → neck → head.
Drop-in replacements for the old mmdet.BaseDetector subclasses.
"""
import torch
import torch.nn as nn


class SingleStreamDetector(nn.Module):
    """Single-input (RGB or IR only) rotated object detector.

    Backbone → Neck (optional) → RotatedRetinaHead.

    Works with the pure-PyTorch Trainer: trainer calls model(images) to get
    predictions, then model.bbox_head.loss() to compute losses.
    """

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

    def extract_feat(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward_train(self, x, gt_bboxes, gt_labels):
        """Training forward: returns (cls_scores, bbox_preds) and loss dict."""
        feats = self.extract_feat(x)
        cls_scores, bbox_preds = self.bbox_head(feats)
        losses = self.bbox_head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)
        return losses

    def forward_test(self, x):
        """Inference forward: returns list of (dets, labels, scores) per image."""
        feats = self.extract_feat(x)
        cls_scores, bbox_preds = self.bbox_head(feats)
        return self.bbox_head.get_bboxes(cls_scores, bbox_preds, cfg=self.test_cfg)

    def forward(self, x):
        """Returns (cls_scores, bbox_preds) for use with Trainer."""
        feats = self.extract_feat(x)
        return self.bbox_head(feats)


class TwoStreamDetector(nn.Module):
    """Dual-input (RGB + IR) rotated object detector.

    Two-stream backbone → fusion → Neck → RotatedRetinaHead.

    The backbone receives (rgb, ir) as a tuple/list and returns fused features.
    Compatible with TwoStreamResNet, C2FormerResNet, etc.
    """

    def __init__(self, backbone, neck=None, bbox_head=None, train_cfg=None, test_cfg=None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.train_cfg = train_cfg or {}
        self.test_cfg = test_cfg or {}

    def extract_feat(self, rgb, ir):
        feats = self.backbone(rgb, ir)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward_train(self, rgb, ir, gt_bboxes, gt_labels):
        """Training forward with dual input."""
        feats = self.extract_feat(rgb, ir)
        cls_scores, bbox_preds = self.bbox_head(feats)
        losses = self.bbox_head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)
        return losses

    def forward_test(self, rgb, ir):
        """Inference forward with dual input."""
        feats = self.extract_feat(rgb, ir)
        cls_scores, bbox_preds = self.bbox_head(feats)
        return self.bbox_head.get_bboxes(cls_scores, bbox_preds, cfg=self.test_cfg)

    def forward(self, rgb, ir):
        """Returns (cls_scores, bbox_preds) from dual input."""
        feats = self.extract_feat(rgb, ir)
        return self.bbox_head(feats)
