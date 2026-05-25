"""
DMM (Disparity-guided Multispectral Mamba) detector for RGB-IR oriented detection.

Fuses VI and IR features via DCFModule (cross-modal Mamba SSM),
with MTAttentionBlock for intra-modal target enhancement.
Uses dual VMamba backbones + S2ANet detection heads.

Port from: https://github.com/Another-0/DMM
IEEE TGRS 2025
"""
from typing import List, Optional
import torch
import torch.nn as nn
from mmengine.registry import MODELS
from mmdet.models.detectors.base import BaseDetector


@MODELS.register_module()
class DMMS2ANet(BaseDetector):
    """DMM detector: dual backbone → MTA → DCFM → FPN → S2ANet.

    Architecture:
    1. VI and IR images processed by separate VMamba (or ResNet) backbones
    2. MTAttentionBlock enhances VI features with multi-scale spatial attention
    3. DCFModule fuses VI and IR features via cross-modal Mamba scanning
    4. Standard FPN neck on fused features
    5. S2ANet init head + refine head for rotated detection
    """

    def __init__(self,
                 backbone_vi: dict,
                 backbone_ir: dict,
                 fusblock: Optional[dict] = None,
                 mtablock: Optional[dict] = None,
                 neck: Optional[dict] = None,
                 bbox_head_init: Optional[dict] = None,
                 bbox_head_refine: Optional[List[dict]] = None,
                 aux_neck: Optional[dict] = None,
                 aux_rpn_head: Optional[dict] = None,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Dual backbones
        self.backbone_vi = MODELS.build(backbone_vi)
        self.backbone_ir = MODELS.build(backbone_ir)

        # DMM fusion modules
        if fusblock is not None:
            self.fusblock = MODELS.build(fusblock)
        else:
            self.fusblock = None

        if mtablock is not None:
            self.mtablock = MODELS.build(mtablock)
        else:
            self.mtablock = None

        # FPN neck
        if neck is not None:
            self.neck = MODELS.build(neck)
        else:
            self.neck = None

        # S2ANet heads
        if bbox_head_init is not None:
            self.bbox_head_init = MODELS.build(bbox_head_init)
        if bbox_head_refine is not None:
            self.bbox_head_refine = nn.ModuleList()
            for refine_cfg in bbox_head_refine:
                self.bbox_head_refine.append(MODELS.build(refine_cfg))

        # Optional auxiliary RPN (on VI features, for target-prior-aware task)
        if aux_neck is not None:
            self.aux_neck = MODELS.build(aux_neck)
            self.aux_rpn_head = MODELS.build(aux_rpn_head)
        else:
            self.aux_neck = None
            self.aux_rpn_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _forward_dual_backbone(self, batch_inputs_vi, batch_inputs_ir):
        """Extract multi-scale features from both backbones."""
        vi_feats = self.backbone_vi(batch_inputs_vi)
        ir_feats = self.backbone_ir(batch_inputs_ir)
        return vi_feats, ir_feats

    def extract_feat(self, batch_inputs_vi, batch_inputs_ir=None):
        """Extract fused features through: backbone → MTA → DCFM → FPN."""
        if batch_inputs_ir is None:
            # Single modality fallback
            feats = self.backbone_vi(batch_inputs_vi)
            if self.neck is not None:
                feats = self.neck(feats)
            return feats

        # Dual backbone forward
        vi_feats, ir_feats = self._forward_dual_backbone(
            batch_inputs_vi, batch_inputs_ir)

        # MTAttention on VI features
        if self.mtablock is not None:
            vi_feats = self.mtablock(vi_feats)

        # DCFM cross-modal fusion
        if self.fusblock is not None:
            feats = self.fusblock(vi_feats, ir_feats)
        else:
            # Simple sum fusion fallback
            feats = [vi + ir for vi, ir in zip(vi_feats, ir_feats)]

        # FPN neck
        if self.neck is not None:
            feats = self.neck(feats)

        return feats

    def loss(self, batch_inputs, batch_inputs_ir,
             batch_data_samples, **kwargs):
        """Compute S2ANet losses with DMM-fused features."""
        x = self.extract_feat(batch_inputs, batch_inputs_ir)

        losses = dict()

        # S2ANet init head (FAM)
        fam_outs = self.bbox_head_init(x)
        loss_inputs = fam_outs + (batch_data_samples,)
        fam_losses = self.bbox_head_init.loss_by_feat(*loss_inputs)
        for name, value in fam_losses.items():
            losses[f'fam.{name}'] = value

        # Refine bboxes + ODM head
        rois = self.bbox_head_init.refine_bboxes(*fam_outs)
        # Use align_conv if available, otherwise pass through
        if hasattr(self, 'align_conv'):
            align_feat = self.align_conv(x, rois)
        else:
            align_feat = x
        odm_outs = self.bbox_head_refine[0](align_feat)
        loss_inputs = odm_outs + (batch_data_samples,)
        odm_losses = self.bbox_head_refine[0].loss_by_feat(
            *loss_inputs, rois=rois)
        for name, value in odm_losses.items():
            losses[f'odm.{name}'] = value

        return losses

    def predict(self, batch_inputs, batch_inputs_ir,
                batch_data_samples, rescale=True):
        """Inference with DMM-fused features."""
        x = self.extract_feat(batch_inputs, batch_inputs_ir)

        fam_outs = self.bbox_head_init(x)
        rois = self.bbox_head_init.refine_bboxes(*fam_outs)
        if hasattr(self, 'align_conv'):
            align_feat = self.align_conv(x, rois)
        else:
            align_feat = x
        odm_outs = self.bbox_head_refine[0](align_feat)

        results_list = self.bbox_head_refine[0].predict_by_feat(
            *odm_outs, batch_data_samples=batch_data_samples,
            rescale=rescale, rois=rois)
        return results_list

    def forward(self, batch_inputs, batch_inputs_ir=None,
                batch_data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(batch_inputs, batch_inputs_ir,
                             batch_data_samples)
        elif mode == 'predict':
            return self.predict(batch_inputs, batch_inputs_ir,
                                batch_data_samples)
        elif mode == 'tensor':
            return self.extract_feat(batch_inputs, batch_inputs_ir)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
