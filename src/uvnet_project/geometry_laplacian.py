"""Joint face CNN and fixed-Laplacian classifier, without message passing."""

from dataclasses import dataclass

import torch
from torch import nn

@dataclass
class FaceLaplacianBatch:
    """Tensor batch compatible with DataLoader pinning and our evaluator."""

    grids: torch.Tensor
    laplacian: torch.Tensor

    def to(self, device, non_blocking=False):
        grids = self.grids.to(device, non_blocking=non_blocking)
        if torch.device(device).type == "cuda":
            grids = grids.contiguous(memory_format=torch.channels_last)
        return FaceLaplacianBatch(
            grids, self.laplacian.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self):
        return FaceLaplacianBatch(self.grids.pin_memory(), self.laplacian.pin_memory())


class GeometryLaplacianClassifier(nn.Module):
    """Recompute geometry features on every forward pass for joint training."""

    def __init__(self, face_cnn, classifier):
        super().__init__()
        self.face_cnn = face_cnn
        self.classifier = classifier

    def forward(self, inputs):
        geometry = self.face_cnn(inputs.grids)
        return self.classifier(torch.cat((geometry, inputs.laplacian), dim=1))
