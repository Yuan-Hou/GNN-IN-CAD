"""Curve UV-grid encoder from UV-Net Section 3.2."""

from torch import nn


class EdgeCNN(nn.Module):
    """Encode (N, C, U) edge grids as (N, embedding_dim) features.

    Six input channels contain xyz and unit curve tangents.
    Three channels can be used for xyz alone. The default
    64-dimensional embedding matches the paper. Layer settings and weight
    initialization follow AutodeskAILab/UV-Net's UVNetCurveEncoder.
    """

    def __init__(self, embedding_dim: int = 64, in_channels: int = 6):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
        )

        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected edge grids (N, {self.in_channels}, U).")
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h).flatten(start_dim=1)
        return self.fc(h)
