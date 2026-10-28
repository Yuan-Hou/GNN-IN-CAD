"""Joint geometry and fixed graphlet-feature classifier; no message passing."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FaceGraphletBatch:
    grids: torch.Tensor
    graphlets: torch.Tensor

    def to(self, device, non_blocking=False):
        grids = self.grids.to(device, non_blocking=non_blocking)
        if torch.device(device).type == "cuda":
            grids = grids.contiguous(memory_format=torch.channels_last)
        return FaceGraphletBatch(grids, self.graphlets.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        return FaceGraphletBatch(self.grids.pin_memory(), self.graphlets.pin_memory())


class GeometryGraphletClassifier(nn.Module):
    def __init__(self, face_cnn, classifier):
        super().__init__()
        self.face_cnn = face_cnn
        self.classifier = classifier

    def forward(self, inputs):
        geometry = self.face_cnn(inputs.grids)
        return self.classifier(torch.cat((geometry, inputs.graphlets), dim=1))
