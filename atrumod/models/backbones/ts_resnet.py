"""
Two-Stream ResNet backbone for RGB-IR multimodal detection.
Each modality gets its own ResNet stem + 4 stages.
Based on: https://github.com/yuanmaoxun/C2Former

Works both standalone and as an MMRotate registered module.
"""
import torch.nn as nn
from .resnet_blocks import BasicBlock, Bottleneck

try:
    from mmdet.registry import MODELS
    _HAS_MMDET = True
except ImportError:
    _HAS_MMDET = False


def _register(cls):
    if _HAS_MMDET:
        return MODELS.register_module()(cls)
    return cls


@_register
class TwoStreamResNet(nn.Module):
    """Two-stream ResNet with separate RGB and IR branches.

    Architecture matches standard ResNet but duplicates stem + stages.
    The C2FormerResNet subclass adds cross-modal fusion between stages.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth=50,
                 num_stages=4,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.depth = depth
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.zero_init_residual = zero_init_residual
        self.block, self.stage_blocks = self.arch_settings[depth]
        self.stage_blocks = self.stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer(style, norm_cfg)

        self.vis_res_layers = []
        self.lwir_res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = 2 if i > 0 else 1
            planes = 64 * 2 ** i

            # Build VIS layer — save and restore inplanes so streams are independent
            _saved_inplanes = self.inplanes
            vis_layer = self._make_res_layer(
                self.block, planes, num_blocks, stride=stride,
                norm_cfg=norm_cfg, avg_down=self.avg_down,
                stage_idx=i)
            self.inplanes = _saved_inplanes  # restore for IR stream
            lwir_layer = self._make_res_layer(
                self.block, planes, num_blocks, stride=stride,
                norm_cfg=norm_cfg, avg_down=self.avg_down,
                stage_idx=i)

            vis_name = f'vis_layer{i + 1}'
            lwir_name = f'lwir_layer{i + 1}'
            self.add_module(vis_name, vis_layer)
            self.add_module(lwir_name, lwir_layer)
            self.vis_res_layers.append(vis_name)
            self.lwir_res_layers.append(lwir_name)

        self._freeze_stages()

        if pretrained:
            self.init_weights(pretrained)

    def _make_stem_layer(self, style, norm_cfg):
        if isinstance(norm_cfg, dict):
            norm_cfg = norm_cfg.copy()

        def _make_norm(num_features):
            if isinstance(norm_cfg, dict):
                bn = nn.BatchNorm2d(num_features)
                if norm_cfg.get('requires_grad', True) is False:
                    for p in bn.parameters():
                        p.requires_grad = False
                return bn
            return norm_cfg(num_features) if callable(norm_cfg) else nn.BatchNorm2d(num_features)

        if self.deep_stem:
            self.vis_stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                _make_norm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                _make_norm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            )
            self.lwir_stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                _make_norm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
                _make_norm(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            )
        else:
            self.vis_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.lwir_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.vis_norm1 = _make_norm(64)
            self.lwir_norm1 = _make_norm(64)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _make_norm(self, num_features, norm_cfg):
        if isinstance(norm_cfg, dict):
            bn = nn.BatchNorm2d(num_features)
            if norm_cfg.get('requires_grad', True) is False:
                for p in bn.parameters():
                    p.requires_grad = False
            return bn
        return norm_cfg(num_features) if callable(norm_cfg) else nn.BatchNorm2d(num_features)

    def _make_res_layer(self, block, planes, num_blocks, stride, norm_cfg, avg_down, stage_idx):
        norm_layer = self._make_norm
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                    norm_layer(planes * block.expansion, norm_cfg),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion, norm_cfg),
                )

        layers = []
        layers.append(block(
            self.inplanes, planes, stride=stride, downsample=downsample,
            norm_cfg=norm_cfg))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, norm_cfg=norm_cfg))

        if self.zero_init_residual:
            for m in layers:
                if hasattr(m, 'bn3'):
                    nn.init.constant_(m.bn3.weight, 0)
                elif hasattr(m, 'bn2') and m.__class__.__name__ == 'BasicBlock':
                    nn.init.constant_(m.bn2.weight, 0)

        return nn.Sequential(*layers)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.vis_stem if self.deep_stem else [self.vis_conv1, self.vis_norm1],
                      self.lwir_stem if self.deep_stem else [self.lwir_conv1, self.lwir_norm1]]:
                modules = m if isinstance(m, list) else [m]
                for mod in modules:
                    for param in mod.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            for layer in [getattr(self, f'vis_layer{i}'), getattr(self, f'lwir_layer{i}')]:
                for param in layer.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        if pretrained:
            import torch
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            # Map single-stream weights to both streams
            new_state = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    k = k[9:]
                new_state[f'vis_{k}'] = v
                new_state[f'lwir_{k}'] = v
            self.load_state_dict(new_state, strict=False)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

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

            if i in self.out_indices:
                out = vis_x + lwir_x  # simple sum fusion by default
                outs.append(out)

        return tuple(outs)
