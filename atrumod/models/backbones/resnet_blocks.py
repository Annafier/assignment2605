"""Self-contained ResNet BasicBlock and Bottleneck — no MMRotate dependency."""
import torch.nn as nn


class BasicBlock(nn.Module):
    """ResNet BasicBlock (used by ResNet-18, 34)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_cfg=None, **kwargs):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = self._make_norm(planes, norm_cfg)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = self._make_norm(planes, norm_cfg)
        self.downsample = downsample
        self.stride = stride

    @staticmethod
    def _make_norm(num_features, norm_cfg):
        if isinstance(norm_cfg, dict):
            bn = nn.BatchNorm2d(num_features)
            if norm_cfg.get('requires_grad', True) is False:
                for p in bn.parameters():
                    p.requires_grad = False
            return bn
        return norm_cfg(num_features)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """ResNet Bottleneck (used by ResNet-50, 101, 152)."""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_cfg=None, **kwargs):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = self._make_norm(planes, norm_cfg)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = self._make_norm(planes, norm_cfg)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = self._make_norm(planes * self.expansion, norm_cfg)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @staticmethod
    def _make_norm(num_features, norm_cfg):
        if isinstance(norm_cfg, dict):
            bn = nn.BatchNorm2d(num_features)
            if norm_cfg.get('requires_grad', True) is False:
                for p in bn.parameters():
                    p.requires_grad = False
            return bn
        return norm_cfg(num_features)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)
