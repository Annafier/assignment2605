"""Rotated anchor generator — self-contained, no mmrotate dependency."""
import torch
import math


class RotatedAnchorGenerator:
    """Generate rotated anchors at multiple scales for each FPN level.

    Args:
        strides: list of stride values per FPN level, e.g. [8, 16, 32, 64, 128]
        scales: anchor scales (relative to stride), e.g. [4]
        ratios: aspect ratios, e.g. [1.0]
        angles: anchor rotation angles in degrees, default [0]
    """

    def __init__(self, strides, scales=None, ratios=None, angles=None):
        self.strides = strides
        self.scales = scales or [4]
        self.ratios = ratios or [1.0]
        self.angles = angles or [0]
        self.num_anchors = len(self.scales) * len(self.ratios) * len(self.angles)
        self._cache = {}

    def _gen_single_level(self, feat_h, feat_w, stride, device, dtype):
        """Generate anchors for one FPN level."""
        key = (feat_h, feat_w, stride)
        if key in self._cache:
            anchors = self._cache[key].to(device=device, dtype=dtype)
        else:
            shift_x = (torch.arange(0, feat_w, device='cpu') + 0.5) * stride
            shift_y = (torch.arange(0, feat_h, device='cpu') + 0.5) * stride
            shift_yy, shift_xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
            shifts = torch.stack([shift_xx, shift_yy], dim=-1)  # (H, W, 2)

            base_anchors = []
            for scale in self.scales:
                ws = stride * scale
                for ratio in self.ratios:
                    hs = ws / ratio
                    for angle in self.angles:
                        base_anchors.append([0, 0, ws, hs, angle])
            base_anchors = torch.tensor(base_anchors, device='cpu')  # (N, 5)

            K = base_anchors.shape[0]
            shifts = shifts.unsqueeze(2).expand(-1, -1, K, -1)  # (H, W, K, 2)
            base_anchors = base_anchors.view(1, 1, K, 5).expand(feat_h, feat_w, K, 5)

            anchors = torch.cat([shifts, base_anchors[..., 2:]], dim=-1)  # (H, W, K, 5)
            self._cache[key] = anchors

        return anchors.to(device=device, dtype=dtype)

    def __call__(self, featmap_sizes, device):
        """Generate anchors for all FPN levels.

        Args:
            featmap_sizes: list of (H, W) per level
            device: target device
        Returns:
            anchors_list: list of (H_i*W_i*K, 5) per level
            valid_flags: list of (H_i*W_i*K,) per level
        """
        anchors_list = []
        valid_flags = []
        dtype = torch.float32
        for i, (h, w) in enumerate(featmap_sizes):
            anchors = self._gen_single_level(h, w, self.strides[i], device, dtype)
            anchors = anchors.reshape(-1, 5)
            valid = torch.ones(anchors.shape[0], dtype=torch.bool, device=device)

            # Filter anchors outside image boundaries (for training stability)
            anchors_list.append(anchors)
            valid_flags.append(valid)

        return anchors_list, valid_flags
