"""
Two-Stream S2ANet detector for RGB-IR multimodal oriented object detection.
Port from: https://github.com/yuanmaoxun/C2Former
"""
import torch
import warnings
from mmengine.registry import MODELS
from mmrotate.models.detectors.s2anet import S2ANet
from mmrotate.structures import rbbox2result


@MODELS.register_module()
class TwoStreamS2ANet(S2ANet):
    """S2ANet variant that accepts paired RGB + IR inputs.

    Uses a dual-stream backbone (e.g., C2FormerResNet) for multimodal fusion.
    Inherits the standard S2ANet FAM/ODM detection heads.
    """

    def extract_feat(self, batch_inputs, batch_inputs_ir=None):
        """Extract features from dual-stream backbone + neck."""
        if batch_inputs_ir is None:
            return super().extract_feat(batch_inputs)
        x = self.backbone(batch_inputs, batch_inputs_ir)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs, batch_inputs_ir=None,
             batch_data_samples=None):
        """Forward for training with dual input."""
        if batch_inputs_ir is None:
            return super().loss(batch_inputs, batch_data_samples)

        x = self.extract_feat(batch_inputs, batch_inputs_ir)

        losses = dict()

        # FAM head forward
        fam_outs = self.fam_head(x)
        loss_inputs = fam_outs + (batch_data_samples,)
        fam_losses = self.fam_head.loss_by_feat(*loss_inputs)
        for name, value in fam_losses.items():
            losses[f'fam.{name}'] = value

        # Refine bboxes
        rois = self.fam_head.refine_bboxes(*fam_outs)
        align_feat = self.align_conv(x, rois)
        odm_outs = self.odm_head(align_feat)
        loss_inputs = odm_outs + (batch_data_samples,)
        odm_losses = self.odm_head.loss_by_feat(*loss_inputs, rois=rois)
        for name, value in odm_losses.items():
            losses[f'odm.{name}'] = value

        return losses

    def predict(self, batch_inputs, batch_inputs_ir=None,
                batch_data_samples=None, rescale=True):
        """Forward for inference with dual input."""
        if batch_inputs_ir is None:
            return super().predict(batch_inputs, batch_data_samples, rescale=rescale)

        x = self.extract_feat(batch_inputs, batch_inputs_ir)
        fam_outs = self.fam_head(x)
        rois = self.fam_head.refine_bboxes(*fam_outs)
        align_feat = self.align_conv(x, rois)
        odm_outs = self.odm_head(align_feat)

        results_list = self.odm_head.predict_by_feat(
            *odm_outs, batch_data_samples=batch_data_samples,
            rescale=rescale, rois=rois)
        return results_list

    def forward(self, batch_inputs, batch_inputs_ir=None,
                batch_data_samples=None, mode='tensor'):
        if batch_inputs_ir is None:
            return super().forward(batch_inputs, batch_data_samples, mode=mode)
        if mode == 'loss':
            return self.loss(batch_inputs, batch_inputs_ir, batch_data_samples)
        elif mode == 'predict':
            return self.predict(batch_inputs, batch_inputs_ir, batch_data_samples)
        elif mode == 'tensor':
            return self.extract_feat(batch_inputs, batch_inputs_ir)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
