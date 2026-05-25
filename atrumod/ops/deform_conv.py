"""
Pure-PyTorch DeformConv2d — replaces mmcv.ops.DeformConv2d.

Uses torchvision.ops.deform_conv2d (available since torchvision 0.13).
Falls back to regular Conv2d if torchvision is not installed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d
    _HAS_TORCHVISION_DCN = True
except ImportError:
    _HAS_TORCHVISION_DCN = False


class DeformConv2d(nn.Module):
    """Deformable Convolution v2 — pure-PyTorch implementation.

    API-compatible with mmcv.ops.DeformConv2d.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.deform_groups = deform_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups,
                        self.kernel_size[0], self.kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, offset):
        """Forward with offset-based deformable sampling.

        Args:
            x: (B, C, H, W) input feature map
            offset: (B, 2*deform_groups*kH*kW, H_out, W_out) sampling offsets
        Returns:
            output: (B, out_channels, H_out, W_out)
        """
        if _HAS_TORCHVISION_DCN:
            mask = None
            return _tv_deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, mask=mask)
        else:
            return _fallback_dcn_forward(x, offset, self)


def _fallback_dcn_forward(x, offset, module):
    """Fallback DCN using regular conv (no offset)."""
    return F.conv2d(x, module.weight, module.bias,
                    stride=module.stride, padding=module.padding,
                    dilation=module.dilation, groups=module.groups)


import math
