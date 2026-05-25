"""
DMM: Disparity-guided Multispectral Mamba — core fusion modules.

DCFM (Dynamic Cross-modal Fusion Mamba) — fuses VI and IR features
via cross-modal interleaved token mixing at each feature scale.

MTAttentionBlock — Multi-scale Target-aware Attention on VI features.

Port from: https://github.com/Another-0/DMM
IEEE TGRS 2025
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """SE-like channel attention for flattened tokens (B, N, C)."""

    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.fc2 = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        # x: (B, N, C)
        attn = x.mean(dim=1, keepdim=True)
        attn = self.fc1(attn)
        attn = F.gelu(attn)
        attn = self.fc2(attn)
        attn = attn.sigmoid()
        return x * attn


class MultiScaleSpatialAttention(nn.Module):
    """Multi-kernel spatial attention: 7x7, 5x5, 3x3 depthwise convs."""

    def __init__(self, dim):
        super().__init__()
        self.conv7 = nn.Conv2d(dim, 1, 7, padding=3, groups=1)
        self.conv5 = nn.Conv2d(dim, 1, 5, padding=2, groups=1)
        self.conv3 = nn.Conv2d(dim, 1, 3, padding=1, groups=1)
        self.proj = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        a7 = self.conv7(x)
        a5 = self.conv5(x)
        a3 = self.conv3(x)
        attn = torch.cat([a7, a5, a3], dim=1)
        attn = self.proj(attn).sigmoid()
        return x * attn


class MultiScaleAttentionLayer(nn.Module):
    """Conv-Norm-GELU + MultiScaleSpatialAttention."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.msa = MultiScaleSpatialAttention(dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.msa(x)
        return x


class MTAttentionBlock(nn.Module):
    """Multi-scale Target-aware Attention block.

    Applies per-scale spatial-channel attention to VI features
    before they enter the fusion module.
    """

    def __init__(self, channels_per_scale):
        super().__init__()
        self.attn_layers = nn.ModuleList([
            MultiScaleAttentionLayer(c) for c in channels_per_scale
        ])

    def forward(self, features):
        """features: list of (B, C_i, H_i, W_i) per scale."""
        return [layer(f) for layer, f in zip(self.attn_layers, features)]


class DSSM(nn.Module):
    """Dynamic State Space Model — cross-modal fusion operator.

    Projects VI, IR, and their difference into a shared space,
    applies interleaved cross-modal 1D conv mixing,
    gates via channel attention, and outputs fused features.
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        d_inner = int(d_model * expand)

        # Input projections for VI, IR, and their subtracted difference
        self.in_proj_vi = nn.Linear(d_model, d_inner * 2)
        self.in_proj_ir = nn.Linear(d_model, d_inner * 2)
        self.in_proj_sub = nn.Linear(d_model, d_inner * 2)

        # Depth-wise convolutions (and the mixing conv)
        self.conv_vi = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv // 2, groups=d_inner)
        self.conv_ir = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv // 2, groups=d_inner)
        self.conv_sub = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv // 2, groups=d_inner)

        self.act = nn.SiLU()

        # Channel attention gates
        self.ca_vi = ChannelAttention(d_inner)
        self.ca_ir = ChannelAttention(d_inner)

        # Output projections
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _cross_modal_mix(self, vi_x, ir_x, sub_x):
        """Cross-modal token mixing via conv1d over interleaved sequences.

        Creates sequence: [VI[0], sub[0], IR[0], sub[1], VI[1], sub[2], ...]
        and applies depthwise conv to mix information across modalities.
        This approximates the cross-modal Mamba scan in a simpler form.
        """
        # Interleave: VI, sub, IR, sub → (B, d_inner, 4*N)
        combined = torch.cat([
            vi_x.transpose(1, 2),
            sub_x.transpose(1, 2),
            ir_x.transpose(1, 2),
            sub_x.transpose(1, 2),
        ], dim=-1)  # (B, d_inner, 4*N)

        # Apply 1D depthwise conv over the interleaved sequence
        y = self.conv_vi(combined)  # reuse conv_vi as mixer
        y = self.act(y)

        # De-interleave back to VI, IR components
        B_d, C_d, Len = y.shape
        seg = Len // 4
        y_vi = y[..., :seg] + 0.5 * y[..., 3*seg:]
        y_sub = 0.5 * (y[..., seg:2*seg] + y[..., 3*seg:])
        y_ir = y[..., 2*seg:3*seg] + 0.5 * y[..., seg:2*seg]

        return y_vi, y_ir

    def forward(self, vi, ir):
        """Fuse VI and IR features at one scale.

        Args:
            vi: (B, C, H, W) visible image features
            ir: (B, C, H, W) infrared image features
        Returns:
            fused: (B, C, H, W)
        """
        B, C, H, W = vi.shape
        N = H * W

        # Flatten to (B, N, C)
        vi_flat = vi.flatten(2).transpose(1, 2)
        ir_flat = ir.flatten(2).transpose(1, 2)
        sub_flat = vi_flat - ir_flat

        # Input projections
        vi_proj = self.in_proj_vi(vi_flat)  # (B, N, 2*d_inner)
        ir_proj = self.in_proj_ir(ir_flat)
        sub_proj = self.in_proj_sub(sub_flat)

        d_inner = vi_proj.shape[-1] // 2
        vi_x, vi_z = vi_proj[..., :d_inner], vi_proj[..., d_inner:]
        ir_x, ir_z = ir_proj[..., :d_inner], ir_proj[..., d_inner:]
        sub_x, _ = sub_proj[..., :d_inner], sub_proj[..., d_inner:]

        # Depthwise conv on each modality
        vi_x = self.conv_vi(vi_x.transpose(1, 2)).transpose(1, 2)
        ir_x = self.conv_ir(ir_x.transpose(1, 2)).transpose(1, 2)
        sub_x = self.conv_sub(sub_x.transpose(1, 2)).transpose(1, 2)

        vi_x = self.act(vi_x)
        ir_x = self.act(ir_x)
        sub_x = self.act(sub_x)

        # Cross-modal mixing — returns (B, d_inner, N)
        y_vi, y_ir = self._cross_modal_mix(vi_x, ir_x, sub_x)

        # Transpose to (B, N, d_inner) for gating and output projection
        y_vi = y_vi.transpose(1, 2)
        y_ir = y_ir.transpose(1, 2)

        # Channel gating
        y_vi = y_vi * self.ca_vi(vi_z).sigmoid()
        y_ir = y_ir * self.ca_ir(ir_z).sigmoid()

        # Output projection + dropout
        y_vi = self.out_proj(y_vi)
        y_ir = self.out_proj(y_ir)
        y_vi = self.dropout(y_vi)
        y_ir = self.dropout(y_ir)

        # Sum into single fused output
        fused = y_vi + y_ir  # (B, N, C)

        return fused.transpose(1, 2).reshape(B, C, H, W)


class DCFM(nn.Module):
    """Dynamic Cross-modal Fusion Mamba block — DSSM + DropPath."""

    def __init__(self, dim, d_state=16, d_conv=3, expand=2,
                 dropout=0.0, drop_path=0.0):
        super().__init__()
        self.dssm = DSSM(dim, d_state, d_conv, expand, dropout)
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, vi, ir):
        fused = self.dssm(vi, ir)
        return self.drop_path(fused) + vi + ir


class DCFModule(nn.Module):
    """Multi-scale DCFM wrapper — one DCFM block per feature level.

    Args:
        channels: list of channel dimensions per scale, e.g. [96, 192, 384, 768]
        d_state: SSM state dimension
    """

    def __init__(self, channels, d_state=16, d_conv=3, expand=2,
                 dropout=0.0, drop_path=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            DCFM(c, d_state, d_conv, expand, dropout, drop_path)
            for c in channels
        ])

    def forward(self, vi_features, ir_features):
        """Fuse multi-scale VI and IR features.

        Args:
            vi_features: list of (B, C_i, H_i, W_i)
            ir_features: list of (B, C_i, H_i, W_i)
        Returns:
            list of fused (B, C_i, H_i, W_i) per scale
        """
        return [block(vi, ir) for block, vi, ir in zip(self.blocks, vi_features, ir_features)]
