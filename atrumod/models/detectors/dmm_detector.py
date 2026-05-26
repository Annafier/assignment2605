"""DMM detector — pure PyTorch, no mmdet dependency.

Dual backbone → MTA → DCFM → FPN → S2ANet (FAM → AlignConv → ODM).
"""
import torch
import torch.nn as nn
from atrumod.models.heads.s2anet_head import S2ANetHead


class DMMDetector(nn.Module):
    """DMM: Disparity-guided Multispectral Mamba detector.

    Architecture:
    1. Dual backbones process RGB and IR separately
    2. MTAttentionBlock enhances RGB features (multi-scale spatial attention)
    3. DCFModule fuses RGB and IR via cross-modal Mamba scanning
    4. FPN neck on fused features
    5. S2ANet head: FAM → AlignConv → ODM
    """

    def __init__(self, backbone_vi, backbone_ir, neck=None,
                 fusblock=None, mtablock=None, bbox_head=None):
        super().__init__()
        self.backbone_vi = backbone_vi
        self.backbone_ir = backbone_ir
        self.neck = neck
        self.fusblock = fusblock
        self.mtablock = mtablock
        self.bbox_head = bbox_head

    def extract_feat(self, rgb, ir):
        """Extract fused features from dual inputs."""
        vi_feats = self.backbone_vi(rgb)
        ir_feats = self.backbone_ir(ir)

        if self.mtablock is not None:
            vi_feats = self.mtablock(vi_feats)

        if self.fusblock is not None and ir_feats is not None:
            feats = self.fusblock(vi_feats, ir_feats)
        else:
            feats = [v + i for v, i in zip(vi_feats, ir_feats)]

        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(self, rgb, ir):
        """Forward pass: returns FAM predictions from S2ANet head."""
        feats = self.extract_feat(rgb, ir)
        return self.bbox_head(feats)

    def forward_train(self, rgb, ir, gt_bboxes, gt_labels):
        """Full training forward: FAM → AlignConv → ODM → loss."""
        feats = self.extract_feat(rgb, ir)
        fam_cls, fam_reg = self.bbox_head(feats)

        # Generate proposals from FAM
        proposals = self.bbox_head.get_proposals(fam_cls, fam_reg)

        # AlignConv on proposals
        aligned = self.bbox_head.align_conv(feats, proposals)

        if aligned is not None and aligned.shape[1] > 0:
            odm_cls, odm_reg = self.bbox_head._odm_forward(aligned)
        else:
            odm_cls, odm_reg = None, None

        # Losses
        losses = self.bbox_head._fam_loss(fam_cls, fam_reg, gt_bboxes, gt_labels)
        return losses

    def forward_test(self, rgb, ir, cfg=None):
        """Inference: FAM → proposals → AlignConv → ODM → NMS."""
        feats = self.extract_feat(rgb, ir)
        fam_cls, fam_reg = self.bbox_head(feats)

        # Generate proposals and run ODM
        proposals = self.bbox_head.get_proposals(fam_cls, fam_reg)
        aligned = self.bbox_head.align_conv(feats, proposals)

        if aligned is not None and aligned.shape[1] > 0:
            odm_cls, odm_reg = self.bbox_head._odm_forward(aligned)
        else:
            odm_cls, odm_reg = None, None

        return self.bbox_head.get_bboxes(fam_cls, fam_reg, odm_cls, odm_reg, cfg=cfg)
