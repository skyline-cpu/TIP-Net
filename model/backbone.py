"""
model/backbone.py  —  Frame-level backbone registry.

Supported:
    xception        — Custom Xception (2048-d output)
    vit_base        — timm ViT-B/16 (768-d output)
    vit_large       — timm ViT-L/16 (1024-d output)
    dino_vit        — Facebook DINO ViT-S/8 via torch.hub (384-d output)
    efficientnet_b4 — timm EfficientNet-B4 (1792-d output)

All backbones return a flat feature vector per frame: [B, output_dim].
Use `build_backbone(name)` to get (backbone, output_dim).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# Xception
# ══════════════════════════════════════════════════════════════════════════════
class _SepConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, d=1, bias=False):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, d, groups=in_ch, bias=bias)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        return self.pw(self.dw(x))


class _XBlock(nn.Module):
    def __init__(self, in_f, out_f, reps, stride=1, start_relu=True, grow_first=True):
        super().__init__()
        self.skip = (nn.Sequential(
            nn.Conv2d(in_f, out_f, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_f))
            if (in_f != out_f or stride != 1) else None)

        rep = []
        f = in_f
        if grow_first:
            rep += [nn.ReLU(inplace=False),
                    _SepConv2d(in_f, out_f, 3, 1, 1),
                    nn.BatchNorm2d(out_f)]
            f = out_f
        for _ in range(reps - 1):
            rep += [nn.ReLU(inplace=True),
                    _SepConv2d(f, f, 3, 1, 1),
                    nn.BatchNorm2d(f)]
        if not grow_first:
            rep += [nn.ReLU(inplace=True),
                    _SepConv2d(in_f, out_f, 3, 1, 1),
                    nn.BatchNorm2d(out_f)]
        if not start_relu:
            rep = rep[1:]
        if stride != 1:
            rep.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        skip = self.skip(x) if self.skip else x
        return self.rep(x) + skip


class XceptionBackbone(nn.Module):
    """Xception → 2048-d global-average-pooled feature."""

    output_dim = 2048

    def __init__(self, pretrained: bool = False, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        self.block1  = _XBlock(64,  128,  2, 2, start_relu=False, grow_first=True)
        self.block2  = _XBlock(128, 256,  2, 2, start_relu=True,  grow_first=True)
        self.block3  = _XBlock(256, 728,  2, 2, start_relu=True,  grow_first=True)
        self.block4  = _XBlock(728, 728,  3, 1)
        self.block5  = _XBlock(728, 728,  3, 1)
        self.block6  = _XBlock(728, 728,  3, 1)
        self.block7  = _XBlock(728, 728,  3, 1)
        self.block8  = _XBlock(728, 728,  3, 1)
        self.block9  = _XBlock(728, 728,  3, 1)
        self.block10 = _XBlock(728, 728,  3, 1)
        self.block11 = _XBlock(728, 728,  3, 1)
        self.block12 = _XBlock(728, 1024, 2, 2, start_relu=True, grow_first=False)

        self.conv3 = _SepConv2d(1024, 1536, 3, 1, 1)
        self.bn3   = nn.BatchNorm2d(1536)
        self.conv4 = _SepConv2d(1536, 2048, 3, 1, 1)
        self.bn4   = nn.BatchNorm2d(2048)

        self._init_weights()

        if pretrained:
            self._load_imagenet_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1); m.bias.data.zero_()

    def _load_imagenet_weights(self):
        try:
            import pretrainedmodels  # pip install pretrainedmodels
            pm = pretrainedmodels.__dict__['xception'](
                num_classes=1000, pretrained='imagenet')
            state = {k: v for k, v in pm.state_dict().items()
                     if k in self.state_dict() and
                     self.state_dict()[k].shape == v.shape}
            missing = self.load_state_dict(state, strict=False)
            print(f'Xception pretrained weights loaded '
                  f'(missing={len(missing.missing_keys)})')
        except Exception as e:
            print(f'Could not load Xception pretrained weights: {e}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        for blk in [self.block1, self.block2, self.block3,
                    self.block4, self.block5, self.block6,
                    self.block7, self.block8, self.block9,
                    self.block10, self.block11, self.block12]:
            x = blk(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.dropout(x)   # [B, 2048]


# ══════════════════════════════════════════════════════════════════════════════
# ViT / DINO wrappers via timm & torch.hub
# ══════════════════════════════════════════════════════════════════════════════
class TimmBackbone(nn.Module):
    """Wraps any timm model as a feature extractor (no classification head)."""

    def __init__(self, name: str, pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        try:
            import timm
        except ImportError:
            raise ImportError('Please install timm: pip install timm')

        self.backbone = timm.create_model(name, pretrained=pretrained,
                                          num_classes=0,  # no head
                                          global_pool='avg')
        self.output_dim = self.backbone.num_features
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.backbone(x))


class DinoViTBackbone(nn.Module):
    """Facebook DINO ViT via torch.hub (vits8, vitb8, etc.)."""

    def __init__(self, arch: str = 'vit_small', patch_size: int = 8,
                 pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        if pretrained:
            self.backbone = torch.hub.load(
                'facebookresearch/dino:main',
                f'dino_{arch.replace("_", "")}{patch_size}',
                pretrained=True)
        else:
            try:
                import timm
                self.backbone = timm.create_model(
                    f'{arch}_patch{patch_size}_224', pretrained=False, num_classes=0)
            except Exception:
                self.backbone = torch.hub.load(
                    'facebookresearch/dino:main',
                    f'dino_{arch.replace("_", "")}{patch_size}',
                    pretrained=False)

        # Detect output dim
        dummy = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(dummy)
        self.output_dim = out.shape[-1]
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        # Some DINO models return [CLS, patch1, patch2, ...]; take CLS
        if feats.dim() == 3:
            feats = feats[:, 0]
        return self.dropout(feats)


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════
def build_backbone(name: str,
                   pretrained: bool = True,
                   dropout: float = 0.0) -> tuple[nn.Module, int]:
    """
    Returns (backbone, output_dim).

    Supported names:
        xception
        efficientnet_b4 | efficientnet_b0 | efficientnet_b7 | ...
        vit_base_patch16_224 | vit_large_patch16_224  (any timm name)
        dino_vit_small | dino_vit_base
    """
    name_lower = name.lower().strip()

    if name_lower == 'xception':
        bb = XceptionBackbone(pretrained=pretrained, dropout=dropout)
        return bb, bb.output_dim

    if name_lower.startswith('dino_'):
        # e.g. dino_vit_small  →  arch=vit_small, patch_size=8 (default)
        parts = name_lower.split('_')   # ['dino', 'vit', 'small']
        arch  = '_'.join(parts[1:])     # 'vit_small'
        bb = DinoViTBackbone(arch=arch, pretrained=pretrained, dropout=dropout)
        return bb, bb.output_dim

    # Fall back to timm for everything else
    bb = TimmBackbone(name, pretrained=pretrained, dropout=dropout)
    return bb, bb.output_dim