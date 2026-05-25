"""
Dual-input data preprocessor for RGB-IR multimodal detection.
Normalizes RGB and IR separately and stacks them for model input.
"""
from typing import Optional, Sequence
import torch
from mmengine.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor


@MODELS.register_module()
class DualInputDataPreprocessor(DetDataPreprocessor):
    """Data preprocessor for paired RGB + IR inputs.

    Applies separate normalization to RGB and IR channels,
    then stacks them as (batch_inputs, batch_inputs_ir) for the detector.
    """

    def __init__(self,
                 mean_rgb=None,
                 std_rgb=None,
                 mean_ir=None,
                 std_ir=None,
                 **kwargs):
        super().__init__(**kwargs)
        if mean_rgb is None:
            mean_rgb = [123.675, 116.28, 103.53]
        if std_rgb is None:
            std_rgb = [58.395, 57.12, 57.375]
        if mean_ir is None:
            mean_ir = [123.675, 116.28, 103.53]
        if std_ir is None:
            std_ir = [58.395, 57.12, 57.375]

        self.mean_rgb = torch.tensor(mean_rgb).view(1, -1, 1, 1)
        self.std_rgb = torch.tensor(std_rgb).view(1, -1, 1, 1)
        self.mean_ir = torch.tensor(mean_ir).view(1, -1, 1, 1)
        self.std_ir = torch.tensor(std_ir).view(1, -1, 1, 1)

    def forward(self, data, training=False):
        """Normalize RGB and IR separately, return as tuple."""
        # Standard preprocessing from parent
        data = super().forward(data, training)

        # If paired data available, split and normalize
        if 'inputs_ir' in data:
            ir = data['inputs_ir']
            if ir.size(1) == 3:
                self.mean_ir = self.mean_ir.to(ir.device)
                self.std_ir = self.std_ir.to(ir.device)
                ir = (ir - self.mean_ir) / self.std_ir
            elif ir.size(1) == 1:
                ir = (ir - self.mean_ir[:, :1]) / self.std_ir[:, :1]
            data['inputs_ir'] = ir

        return data
