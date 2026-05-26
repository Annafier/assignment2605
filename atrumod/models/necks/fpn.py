"""Simple FPN neck — pure PyTorch."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFPN(nn.Module):
    """Minimal Feature Pyramid Network for rotated detection."""

    def __init__(self, in_channels=None, out_channels=256, start_level=0,
                 num_outs=5, add_extra_convs=False):
        super().__init__()
        if in_channels is None:
            in_channels = [256, 512, 1024, 2048]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.start_level = start_level
        self.num_outs = num_outs
        self.num_ins = len(in_channels)

        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels:
            self.lateral_convs.append(
                nn.Conv2d(ch, out_channels, 1))

        # Output convolutions (3x3 after upsample + add)
        self.fpn_convs = nn.ModuleList()
        for _ in range(self.num_ins):
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, 3, padding=1))

        # Extra output levels
        self.extra_convs = nn.ModuleList()
        if num_outs > self.num_ins:
            for i in range(num_outs - self.num_ins):
                extra = nn.Conv2d(
                    out_channels if i == 0 else out_channels,
                    out_channels, 3, stride=2, padding=1)
                self.extra_convs.append(extra)

    def forward(self, feats):
        """FPN forward pass.

        Args:
            feats: list of feature maps from backbone [P2, P3, P4, P5]
        Returns:
            list of FPN outputs [P2_out, P3_out, P4_out, P5_out, (P6_out, ...)]
        """
        feats = feats[self.start_level:self.start_level + self.num_ins]

        # Top-down path
        laterals = []
        for i, feat in enumerate(feats):
            laterals.append(self.lateral_convs[i](feat))

        for i in range(len(laterals) - 1, 0, -1):
            prev = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:],
                                 mode='nearest')
            laterals[i-1] = laterals[i-1] + prev

        outs = []
        for i in range(len(laterals)):
            outs.append(self.fpn_convs[i](laterals[i]))

        # Extra levels via stride-2 conv
        if self.extra_convs:
            last = outs[-1]
            for conv in self.extra_convs:
                last = conv(F.relu(last))
                outs.append(last)

        return outs
