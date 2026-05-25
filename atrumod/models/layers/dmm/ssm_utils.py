"""
Cross-scan and cross-merge utilities for RGB-T Mamba fusion (DMM).
Creates 4-pattern sequences combining VI, IR, and subtraction features
for cross-modal selective scanning.

Port from: https://github.com/Another-0/DMM
"""
import torch
import torch.nn as nn


class CrossScanRGBTK4(torch.autograd.Function):
    """Cross-scan for RGBT: creates 4 directional scan patterns.

    Pattern 0: [VI | diff]  forward
    Pattern 1: [VI | diff]  reverse (flip)
    Pattern 2: [IR | diff]  forward
    Pattern 3: [IR | diff]  reverse (flip)

    Input:  (B, C, 2*H*W) from concat [sub, VI, IR] along spatial dim
    Output: (B, 4, C, H*W) — 4 scan directions
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, N = x.shape  # N = 2*H*W
        ctx.shape = (B, C, N)

        x_0 = x.view(B, C, -1)
        xs = torch.stack([
            x_0,
            x_0.flip(dims=[2]),
            x_0.flip(dims=[2]),
            x_0,
        ], dim=1)  # (B, 4, C, N)
        return xs

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        B, C, N = ctx.shape
        dy_0 = dy[:, 0] + dy[:, 1].flip(dims=[2]) + dy[:, 2].flip(dims=[2]) + dy[:, 3]
        return dy_0.view(B, C, N)


class CrossMergeRGBTK4(torch.autograd.Function):
    """Reverse of CrossScanRGBTK4: merges 4 directions back to flat tensor."""

    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, N = ys.shape  # K=4
        ctx.shape = (B, K, C, N)

        y_0 = ys[:, 0] + ys[:, 1].flip(dims=[2]) + ys[:, 2].flip(dims=[2]) + ys[:, 3]
        return y_0.view(B, C, -1)

    @staticmethod
    def backward(ctx, dx: torch.Tensor):
        B, K, C, N = ctx.shape
        dx = dx.view(B, C, -1)
        dxs = torch.stack([
            dx,
            dx.flip(dims=[2]),
            dx.flip(dims=[2]),
            dx,
        ], dim=1)
        return dxs


class CrossScanMamba(nn.Module):
    """Wraps CrossScanRGBTK4 for use in nn.Module forward pass."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return CrossScanRGBTK4.apply(x)


class CrossMergeMamba(nn.Module):
    """Wraps CrossMergeRGBTK4 for use in nn.Module forward pass."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return CrossMergeRGBTK4.apply(x)
