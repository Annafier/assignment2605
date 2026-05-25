"""
C2Former: Calibrated and Complementary Transformer for RGB-IR fusion.
Port from: https://github.com/yuanmaoxun/C2Former

ICA (Inter-modality Cross-Attention) + AFS (Adaptive Feature Sampling)
for calibrated cross-modal fusion at each ResNet stage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .ts_resnet import TwoStreamResNet

try:
    from mmengine.registry import MODELS
    _HAS_MMENGINE = True
except ImportError:
    _HAS_MMENGINE = False

def _register(cls):
    if _HAS_MMENGINE:
        return MODELS.register_module()(cls)
    return cls


class LayerNormProxy(nn.Module):
    """LayerNorm applied in channel-last then back to channel-first."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class ModalityNorm(nn.Module):
    """Modality-aware normalization: normalizes `ref` and scales/shifts
    it by statistics derived from `lr`. Enables cross-modal calibration."""

    def __init__(self, nf, use_residual=True, learnable=True):
        super().__init__()
        self.learnable = learnable
        self.use_residual = use_residual
        self.norm_layer = nn.InstanceNorm2d(nf, affine=False)

        if self.learnable:
            self.conv = nn.Sequential(
                nn.Conv2d(nf, nf, 3, 1, 1, bias=True),
                nn.ReLU(inplace=True))
            self.conv_gamma = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_beta = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            nn.init.zeros_(self.conv_gamma.weight)
            nn.init.zeros_(self.conv_beta.weight)
            nn.init.zeros_(self.conv_gamma.bias)
            nn.init.zeros_(self.conv_beta.bias)

    def forward(self, lr, ref):
        ref_normed = self.norm_layer(ref)
        if self.learnable:
            x = self.conv(lr)
            gamma = self.conv_gamma(x)
            beta = self.conv_beta(x)
        b, c, h, w = lr.size()
        lr_mean = lr.view(b, c, -1).mean(-1, keepdim=True).unsqueeze(3)
        lr_std = lr.view(b, c, -1).std(-1, keepdim=True).unsqueeze(3)

        if self.learnable:
            if self.use_residual:
                gamma = gamma + lr_std
                beta = beta + lr_mean
            else:
                gamma = 1 + gamma
        else:
            gamma = lr_std
            beta = lr_mean

        return ref_normed * gamma + beta


class C2Former(nn.Module):
    """Core C2Former fusion block: cross-attention with deformable sampling.

    RGB queries attend to IR keys/values (and vice versa) after adaptive
    spatial offset prediction, enabling calibrated cross-modal fusion.
    """

    def __init__(self, q_size, kv_size, n_heads, n_head_channels, n_groups,
                 attn_drop, proj_drop, stride, offset_range_factor,
                 no_off, stage_idx):
        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = kv_size
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor

        ksizes = [9, 7, 5, 3]
        kk = ksizes[stage_idx]

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.n_group_channels, self.n_group_channels, kk,
                      stride, kk // 2, groups=self.n_group_channels),
            LayerNormProxy(self.n_group_channels),
            nn.GELU(),
            nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False))

        self.proj_q_lwir = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_q_vis = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_combinq = nn.Conv2d(self.nc * 2, self.nc, 1, 1, 0)
        self.proj_k_lwir = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_k_vis = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_v_lwir = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_v_vis = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_out_lwir = nn.Conv2d(self.nc, self.nc, 1, 1, 0)
        self.proj_out_vis = nn.Conv2d(self.nc, self.nc, 1, 1, 0)

        self.vis_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.lwir_proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.vis_attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.lwir_attn_drop = nn.Dropout(attn_drop, inplace=True)

        self.vis_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)
        self.lwir_MN = ModalityNorm(self.nc, use_residual=True, learnable=True)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij')
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key).mul_(2).sub_(1)
        ref[..., 0].div_(H_key).mul_(2).sub_(1)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)
        return ref

    def forward(self, vis_x, lwir_x):
        B, C, H, W = vis_x.size()
        dtype, device = vis_x.dtype, vis_x.device

        x = torch.cat([vis_x, lwir_x], 1)
        combin_q = self.proj_combinq(x)

        q_off = einops.rearrange(combin_q, 'b (g c) h w -> (b g) c h w',
                                 g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off)
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / Hk, 1.0 / Wk], device=device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        vis_reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        lwir_reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            vis_pos = vis_reference + offset
            lwir_pos = lwir_reference
        else:
            vis_pos = (vis_reference + offset).tanh()
            lwir_pos = lwir_reference.tanh()

        vis_x_sampled = F.grid_sample(
            input=vis_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=vis_pos[..., (1, 0)], mode='bilinear', align_corners=True)
        lwir_x_sampled = F.grid_sample(
            input=lwir_x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=lwir_pos[..., (1, 0)], mode='bilinear', align_corners=True)

        vis_x_sampled = vis_x_sampled.reshape(B, C, 1, n_sample)
        lwir_x_sampled = lwir_x_sampled.reshape(B, C, 1, n_sample)

        # IR queries attend to RGB keys
        q_lwir = self.proj_q_lwir(self.vis_MN(vis_x, lwir_x))
        q_lwir = q_lwir.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_vis = self.proj_k_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_vis = self.proj_v_vis(vis_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn_vis = torch.einsum('b c m, b c n -> b m n', q_lwir, k_vis)
        attn_vis = attn_vis.mul(self.scale)
        attn_vis = F.softmax(attn_vis, dim=2)
        attn_vis = self.vis_attn_drop(attn_vis)
        out_vis = torch.einsum('b m n, b c n -> b c m', attn_vis, v_vis)
        out_vis = out_vis.reshape(B, C, H, W)
        out_vis = self.vis_proj_drop(self.proj_out_vis(out_vis))

        # RGB queries attend to IR keys
        q_vis = self.proj_q_vis(self.lwir_MN(lwir_x, vis_x))
        q_vis = q_vis.reshape(B * self.n_heads, self.n_head_channels, H * W)
        k_lwir = self.proj_k_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v_lwir = self.proj_v_lwir(lwir_x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn_lwir = torch.einsum('b c m, b c n -> b m n', q_vis, k_lwir)
        attn_lwir = attn_lwir.mul(self.scale)
        attn_lwir = F.softmax(attn_lwir, dim=2)
        attn_lwir = self.lwir_attn_drop(attn_lwir)
        out_lwir = torch.einsum('b m n, b c n -> b c m', attn_lwir, v_lwir)
        out_lwir = out_lwir.reshape(B, C, H, W)
        out_lwir = self.lwir_proj_drop(self.proj_out_lwir(out_lwir))

        return out_vis, out_lwir


@_register
class C2FormerResNet(TwoStreamResNet):
    """C2Former backbone: two-stream ResNet with C2Former fusion blocks.

    After each ResNet stage, RGB and IR features are fused via inter-modality
    cross-attention (ICA) with adaptive feature sampling (AFS).
    """

    def __init__(self,
                 fmap_size=(80, 64),
                 dims_in=None,
                 dims_out=None,
                 num_heads=None,
                 cca_strides=None,
                 groups=None,
                 offset_range_factor=None,
                 no_offs=None,
                 attn_drop_rate=0.0,
                 drop_rate=0.0,
                 **kwargs):
        super().__init__(**kwargs)

        if dims_in is None:
            dims_in = [256, 512, 1024, 2048]
        if dims_out is None:
            dims_out = [96, 192, 384, 768]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if cca_strides is None:
            cca_strides = [3, 3, 3, 3]
        if groups is None:
            groups = [1, 2, 3, 6]
        if offset_range_factor is None:
            offset_range_factor = [2, 2, 2, 2]
        if no_offs is None:
            no_offs = [False, False, False, False]

        self.num_heads = num_heads
        self.fmap_size = fmap_size

        self.c2formers = nn.ModuleList()
        self.vis_convlist1 = nn.ModuleList()
        self.lwir_convlist1 = nn.ModuleList()
        self.vis_convlist2 = nn.ModuleList()
        self.lwir_convlist2 = nn.ModuleList()

        for i in range(self.num_stages):
            hc = dims_out[i] // num_heads[i]
            self.c2formers.append(
                C2Former(self.fmap_size, self.fmap_size, num_heads[i],
                         hc, groups[i], attn_drop_rate, drop_rate,
                         cca_strides[i], offset_range_factor[i],
                         no_offs[i], i))
            self.vis_convlist1.append(
                nn.Sequential(nn.Conv2d(dims_in[i], dims_out[i], 1, 1), nn.ReLU()))
            self.lwir_convlist1.append(
                nn.Sequential(nn.Conv2d(dims_in[i], dims_out[i], 1, 1), nn.ReLU()))
            self.vis_convlist2.append(
                nn.Sequential(nn.Conv2d(dims_out[i], dims_in[i], 1, 1), nn.ReLU()))
            self.lwir_convlist2.append(
                nn.Sequential(nn.Conv2d(dims_out[i], dims_in[i], 1, 1), nn.ReLU()))
            self.fmap_size = (self.fmap_size[0] // 2, self.fmap_size[1] // 2)

    def forward(self, vis_x, lwir_x):
        if self.deep_stem:
            vis_x = self.vis_stem(vis_x)
            lwir_x = self.lwir_stem(lwir_x)
        else:
            vis_x = self.vis_conv1(vis_x)
            vis_x = self.vis_norm1(vis_x)
            vis_x = self.relu(vis_x)
            lwir_x = self.lwir_conv1(lwir_x)
            lwir_x = self.lwir_norm1(lwir_x)
            lwir_x = self.relu(lwir_x)

        vis_x = self.maxpool(vis_x)
        lwir_x = self.maxpool(lwir_x)

        outs = []
        for i in range(self.num_stages):
            vis_layer = getattr(self, self.vis_res_layers[i])
            lwir_layer = getattr(self, self.lwir_res_layers[i])
            vis_x = vis_layer(vis_x)
            lwir_x = lwir_layer(lwir_x)

            input_vis_x = self.vis_convlist1[i](vis_x)
            input_lwir_x = self.lwir_convlist1[i](lwir_x)

            out_vis, out_lwir = self.c2formers[i](input_vis_x, input_lwir_x)
            out_vis = self.vis_convlist2[i](out_vis)
            out_lwir = self.lwir_convlist2[i](out_lwir)
            vis_x = vis_x + out_lwir
            lwir_x = lwir_x + out_vis

            if i in self.out_indices:
                out = vis_x + lwir_x
                outs.append(out)

        return tuple(outs)
