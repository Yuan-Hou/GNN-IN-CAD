"""Surface UV-grid encoder from UV-Net Section 3.2."""

from torch import nn


class FaceCNN(nn.Module):
    """Encode (N, C, U, V) face grids as (N, embedding_dim) features.

    Seven input channels contain xyz, outward unit normals, and a trimming
    mask. Four channels (xyz and mask) can also be used. The default
    64-dimensional embedding matches the paper. Layer settings and weight
    initialization follow AutodeskAILab/UV-Net's UVNetSurfaceEncoder.
    """

    def __init__(self, embedding_dim: int = 64, in_channels: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
        )

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(layer.weight)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected face grids (N, {self.in_channels}, U, V).")
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.pool(h).flatten(start_dim=1)
        return self.fc(h)
