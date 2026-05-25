"""Self-contained two-stream detector — zero mmrotate model imports."""
import torch
from mmengine.registry import MODELS
from mmdet.models.detectors.base import BaseDetector


@MODELS.register_module()
class TwoStreamDetector(BaseDetector):
    """Dual-input (RGB+IR) rotated object detector.

    Backbone (C2Former/TwoStreamResNet) → Neck (FPN) → Head (RotatedRetinaHead).
    Fully self-contained — no dependency on mmrotate model internals.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs, batch_inputs_ir=None):
        """Extract features from backbone + optional neck.

        If batch_inputs_ir is provided, does dual-stream forward.
        Otherwise falls back to single-stream (RGB only).
        """
        if batch_inputs_ir is not None:
            x = self.backbone(batch_inputs, batch_inputs_ir)
        else:
            # Single stream — backbone takes only one input
            x = self.backbone(batch_inputs)
            if isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], list):
                # If backbone returns (vis_feats, ir_feats), take vis
                x = x[0] if isinstance(x[0], (list, tuple)) else x

        if hasattr(self, 'neck'):
            x = self.neck(x)
        return x

    def loss(self, batch_inputs, batch_inputs_ir=None,
             batch_data_samples=None, **kwargs):
        """Compute losses.

        batch_data_samples contains 'gt_bboxes' (rotated 5-column) and 'gt_labels'.
        """
        if batch_data_samples is None:
            batch_data_samples = kwargs.get('data_samples', [])

        x = self.extract_feat(batch_inputs, batch_inputs_ir)

        # Extract GT from data samples
        device = batch_inputs.device
        gt_bboxes = []
        gt_labels = []
        for ds in batch_data_samples:
            if hasattr(ds, 'gt_instances'):
                bbox = ds.gt_instances.bboxes
                label = ds.gt_instances.labels
            elif hasattr(ds, 'gt_bboxes'):
                bbox = ds.gt_bboxes
                label = ds.gt_labels
            elif isinstance(ds, dict):
                bbox = ds.get('gt_bboxes', torch.zeros(0, 5, device=device))
                label = ds.get('gt_labels', torch.zeros(0, dtype=torch.long, device=device))
            else:
                bbox = torch.zeros(0, 5, device=device)
                label = torch.zeros(0, dtype=torch.long, device=device)
            gt_bboxes.append(bbox.to(device))
            gt_labels.append(label.to(device))

        cls_scores, bbox_preds = self.bbox_head(x)
        losses = self.bbox_head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)
        return losses

    def predict(self, batch_inputs, batch_inputs_ir=None,
                batch_data_samples=None, rescale=True):
        """Inference."""
        x = self.extract_feat(batch_inputs, batch_inputs_ir)
        cls_scores, bbox_preds = self.bbox_head(x)
        results = self.bbox_head.get_bboxes(cls_scores, bbox_preds, cfg=self.test_cfg)
        return results

    def _forward(self, batch_inputs):
        return self.extract_feat(batch_inputs)

    def forward(self, batch_inputs, batch_inputs_ir=None,
                batch_data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(batch_inputs, batch_inputs_ir, batch_data_samples)
        elif mode == 'predict':
            return self.predict(batch_inputs, batch_inputs_ir, batch_data_samples)
        elif mode == 'tensor':
            return self._forward(batch_inputs)
        raise RuntimeError(f'Invalid mode "{mode}"')


@MODELS.register_module()
class SingleStreamDetector(BaseDetector):
    """Single-input (RGB or IR only) rotated detector.

    Backbone (ResNet) → Neck (FPN) → Head (RotatedRetinaHead).
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, batch_inputs):
        x = self.backbone(batch_inputs)
        if hasattr(self, 'neck'):
            x = self.neck(x)
        return x

    def loss(self, batch_inputs, batch_data_samples=None, **kwargs):
        if batch_data_samples is None:
            batch_data_samples = kwargs.get('data_samples', [])

        x = self.extract_feat(batch_inputs)
        device = batch_inputs.device

        gt_bboxes, gt_labels = [], []
        for ds in batch_data_samples:
            if hasattr(ds, 'gt_instances'):
                bbox = ds.gt_instances.bboxes
                label = ds.gt_instances.labels
            elif hasattr(ds, 'gt_bboxes'):
                bbox = ds.gt_bboxes
                label = ds.gt_labels
            elif isinstance(ds, dict):
                bbox = ds.get('gt_bboxes', torch.zeros(0, 5, device=device))
                label = ds.get('gt_labels', torch.zeros(0, dtype=torch.long, device=device))
            else:
                bbox = torch.zeros(0, 5, device=device)
                label = torch.zeros(0, dtype=torch.long, device=device)
            gt_bboxes.append(bbox.to(device))
            gt_labels.append(label.to(device))

        cls_scores, bbox_preds = self.bbox_head(x)
        losses = self.bbox_head.loss(cls_scores, bbox_preds, gt_bboxes, gt_labels)
        return losses

    def predict(self, batch_inputs, batch_data_samples=None, rescale=True):
        x = self.extract_feat(batch_inputs)
        cls_scores, bbox_preds = self.bbox_head(x)
        return self.bbox_head.get_bboxes(cls_scores, bbox_preds, cfg=self.test_cfg)

    def _forward(self, batch_inputs):
        return self.extract_feat(batch_inputs)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        raise RuntimeError(f'Invalid mode "{mode}"')
