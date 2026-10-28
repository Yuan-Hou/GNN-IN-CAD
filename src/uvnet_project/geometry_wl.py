"""Two-layer classifier on geometry plus categorical per-face WL features."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class FaceWLBatch:
    geometry: torch.Tensor  # cached embeddings (frozen) or UV-grids (joint)
    colors: torch.Tensor

    def to(self, device, non_blocking=False):
        geometry = self.geometry.to(device, non_blocking=non_blocking)
        if geometry.ndim == 4 and torch.device(device).type == "cuda":
            geometry = geometry.contiguous(memory_format=torch.channels_last)
        return FaceWLBatch(geometry, self.colors.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        return FaceWLBatch(self.geometry.pin_memory(), self.colors.pin_memory())


class WLClassifier(nn.Module):
    """Linear([geometry, one-hot WL history]) -> ReLU -> class logits.

    Embedding lookup and sum exactly implement the categorical part of the
    first linear layer without allocating dense one-hot feature matrices.
    They are classifier weights, not a separate WL embedding network.
    Unknown colours contribute zero, including after training.
    """

    def __init__(self, embedding_dim, hidden_dim, vocabulary, num_classes=16, face_cnn=None):
        super().__init__()
        self.face_cnn = face_cnn
        self.geometry_projection = nn.Linear(embedding_dim, hidden_dim)
        self.color_projection = nn.Embedding(vocabulary["size"], hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("unknown_indices", torch.tensor(vocabulary["unknown_indices"], dtype=torch.long))
        # Same initialization range as a linear layer on the full concatenation.
        bound = 1 / math.sqrt(embedding_dim + vocabulary["size"])
        nn.init.uniform_(self.geometry_projection.weight, -bound, bound)
        nn.init.uniform_(self.geometry_projection.bias, -bound, bound)
        nn.init.uniform_(self.color_projection.weight, -bound, bound)
        with torch.no_grad():
            self.color_projection.weight[self.unknown_indices] = 0

    def forward(self, inputs):
        geometry = self.face_cnn(inputs.geometry) if self.face_cnn is not None else inputs.geometry
        color_values = self.color_projection(inputs.colors)
        known = (inputs.colors != self.unknown_indices).unsqueeze(-1)
        hidden = self.geometry_projection(geometry) + (color_values * known).sum(dim=1)
        return self.linear2(torch.relu(hidden))
