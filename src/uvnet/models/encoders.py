"""UV-grid CNN encoders: 2D surface grids, 1D curve grids, and a topology MLP."""

from __future__ import annotations

from torch import nn

from ..data.features import GDV_DIM


class UVGridEncoder(nn.Module):
    """UV-Net style UV-grid CNN: three conv blocks, global pooling, linear projection."""

    def __init__(self, channels=7, width=64):
        super().__init__()
        layers, current = [], channels
        for out_channels in (64, 128, 256):
            layers.extend((nn.Conv2d(current, out_channels, 3, padding=1, bias=False),
                           nn.GroupNorm(8, out_channels), nn.LeakyReLU()))
            current = out_channels
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(256, width), nn.LayerNorm(width),
                                     nn.LeakyReLU())

    def forward(self, grid):
        return self.network(grid.permute(0, 3, 1, 2).contiguous())


class SurfaceEncoder(UVGridEncoder):
    """10x10x7 face grids: xyz, normal xyz and the trimming mask."""

    def __init__(self, width=64):
        super().__init__(channels=7, width=width)


class CurveEncoder(nn.Module):
    """Trim curves: 10 samples x 6 channels (xyz, normal xyz), encoded with a 1D CNN."""

    def __init__(self, width=64, channels=6):
        super().__init__()
        layers, current = [], channels
        for out_channels in (64, 128, 256):
            layers.extend((nn.Conv1d(current, out_channels, 3, padding=1, bias=False),
                           nn.GroupNorm(8, out_channels), nn.LeakyReLU()))
            current = out_channels
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                     nn.Linear(256, width), nn.LayerNorm(width),
                                     nn.LeakyReLU())

    def forward(self, grid):
        return self.network(grid.transpose(1, 2).contiguous())


class TopologyEncoder(nn.Module):
    """MLP over the 15-dim GDV node descriptor; no surface geometry is used."""

    def __init__(self, width=64, in_dim=GDV_DIM):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(in_dim, width), nn.LayerNorm(width),
                                     nn.LeakyReLU())

    def forward(self, gdv):
        return self.network(gdv)


__all__ = ["UVGridEncoder", "SurfaceEncoder", "CurveEncoder", "TopologyEncoder"]
