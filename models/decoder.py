"""
Feature Pyramid Network (FPN) decoder with CBAM attention.

Takes multi-scale backbone features and produces a unified feature map
that task heads consume for per-pixel predictions.
"""

from typing import Dict  # used in type hints

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CBAM Attention ────────────────────────────────────────────────────────────


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention using channel-wise statistics."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg, mx], dim=1))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ── FPN Decoder ───────────────────────────────────────────────────────────────


class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network that fuses multi-scale backbone features
    into a single high-resolution feature map.

    Produces:
        - fpn_out: (B, out_channels, H/4, W/4) fused feature map
        - Individual scale outputs for skip connections

    Args:
        in_channels: dict mapping scale names to their channel counts
        out_channels: unified channel dimension for FPN outputs
    """

    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: int = 256,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Lateral 1×1 convolutions (project each backbone level to out_channels)
        self.laterals = nn.ModuleDict()
        for name, ch in in_channels.items():
            self.laterals[name] = nn.Conv2d(ch, out_channels, 1)

        # Top-down refinement 3×3 convolutions
        self.smoothing = nn.ModuleDict()
        for name in in_channels:
            self.smoothing[name] = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        # CBAM attention at each level
        self.attention = nn.ModuleDict()
        for name in in_channels:
            self.attention[name] = CBAM(out_channels)

        # Final fusion
        n_levels = len(in_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * n_levels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: dict of backbone feature maps {name: (B, C, H, W)}

        Returns:
            (B, out_channels, H_max, W_max) fused feature map
        """
        # Sort by spatial resolution (largest stride → smallest = coarse to fine)
        sorted_names = sorted(
            features.keys(),
            key=lambda k: features[k].shape[2],  # sort by height (ascending)
        )

        # Build lateral projections
        laterals = {}
        for name in sorted_names:
            if name in self.laterals:
                laterals[name] = self.laterals[name](features[name])
            else:
                # If key not in laterals, skip
                continue

        # Top-down pathway (coarse to fine)
        processed = {}
        prev = None

        # Process coarse-to-fine (reversed order)
        for name in reversed(sorted_names):
            if name not in laterals:
                continue
            lat = laterals[name]
            if prev is not None:
                # Upsample previous coarser level and add
                prev_up = F.interpolate(
                    prev, size=lat.shape[2:], mode="bilinear", align_corners=False
                )
                lat = lat + prev_up
            lat = self.smoothing[name](lat)
            lat = self.attention[name](lat)
            processed[name] = lat
            prev = lat

        # Upsample all levels to finest resolution and concatenate
        target_size = max(processed.values(), key=lambda t: t.shape[2]).shape[2:]

        upsampled = []
        for name in sorted_names:
            if name not in processed:
                continue
            feat = processed[name]
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            upsampled.append(feat)

        fused = torch.cat(upsampled, dim=1)
        return self.fuse(fused)


# ── Task-Group Refinement ─────────────────────────────────────────────────────


class TaskGroupRefinement(nn.Module):
    """
    Shared refinement block for a group of related tasks.
    E.g., all water-related tasks share one refinement.
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.refine(x)
        out = self.cbam(out + x)  # residual + attention
        return out
