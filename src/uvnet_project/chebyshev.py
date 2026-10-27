"""Localized Chebyshev filters following Graph Convnets Lab 01."""

from dataclasses import dataclass

import torch
from torch import nn


def scaled_laplacian(graph):
    """Return sparse L_tilde = 2 L / 2 - I = -D^-1/2 A D^-1/2.

    Use the normalized-Laplacian spectral upper bound 2, avoiding an
    eigendecomposition. L = I - normalized A, including isolated vertices.
    Collapse parallel edges and remove loops, as in the shallow themes.
    The zero-order Chebyshev term already preserves self information.
    """
    n = graph.num_nodes()
    src, dst = (values.cpu().long() for values in graph.edges())
    edges = torch.stack((src, dst))
    edges = edges[:, src != dst]
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    edges = torch.unique(edges, dim=1, sorted=True)
    degree = torch.bincount(edges[0], minlength=n).float()
    inv = degree.clamp_min(1).rsqrt()
    values = -inv[edges[0]] * inv[edges[1]]
    return torch.sparse_coo_tensor(edges, values, (n, n)).coalesce()


@dataclass
class FaceChebBatch:
    geometry: torch.Tensor
    laplacian: torch.Tensor

    def to(self, device, non_blocking=False):
        geometry = self.geometry.to(device, non_blocking=non_blocking)
        if geometry.ndim == 4 and torch.device(device).type == "cuda":
            geometry = geometry.contiguous(memory_format=torch.channels_last)
        return FaceChebBatch(geometry, self.laplacian.to(device, non_blocking=non_blocking))

    def pin_memory(self):
        # Pin sparse components explicitly; generic sparse pinning varies by Torch version.
        lap = self.laplacian.coalesce()
        pinned = torch.sparse_coo_tensor(lap.indices().pin_memory(), lap.values().pin_memory(),
                                         lap.shape, is_coalesced=True)
        return FaceChebBatch(self.geometry.pin_memory(), pinned)


def collate_cheb(batch):
    """Batch independent solids using a sparse block-diagonal Laplacian."""
    geometries, indices, values, labels = [], [], [], []
    offset = 0
    for geometry, laplacian, target in batch:
        laplacian = laplacian.coalesce()
        n = geometry.shape[0]
        if laplacian.shape != (n, n) or target.shape != (n,):
            raise ValueError("Geometry, Laplacian and labels must align by face.")
        geometries.append(geometry)
        indices.append(laplacian.indices() + offset)
        values.append(laplacian.values())
        labels.append(target)
        offset += n
    laplacian = torch.sparse_coo_tensor(torch.cat(indices, dim=1), torch.cat(values),
                                       (offset, offset)).coalesce()
    return FaceChebBatch(torch.cat(geometries), laplacian), torch.cat(labels)


class ChebyshevConv(nn.Module):
    """Concatenate T_0(L)X through T_hops(L)X, then apply one Linear.

    Matches the tutorial's feature-major, order-minor flattening. ``hops``
    is polynomial degree; the tutorial's K (number of terms) is hops + 1.
    """

    def __init__(self, in_features, out_features, hops):
        super().__init__()
        if not isinstance(hops, int) or isinstance(hops, bool) or hops < 0:
            raise ValueError("hops must be a nonnegative integer.")
        self.hops = hops
        self.linear = nn.Linear((hops + 1) * in_features, out_features)

    def forward(self, x, laplacian):
        # Keep the recurrence and sparse multiplication in float32 under CNN AMP.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x0 = x.float()
            terms = [x0]
            if self.hops > 0:
                x1 = torch.sparse.mm(laplacian, x0)
                terms.append(x1)
            for _ in range(2, self.hops + 1):
                x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
                terms.append(x2)
                x0, x1 = x1, x2
            basis = torch.stack(terms, dim=-1).flatten(1)
            return self.linear(basis)


class ChebFaceClassifier(nn.Module):
    """One graph layer, then a pointwise two-layer per-face classifier."""

    def __init__(self, embedding_dim, graph_dim, hidden_dim, hops, num_classes=16,
                 face_cnn=None):
        super().__init__()
        self.face_cnn = face_cnn
        self.conv = ChebyshevConv(embedding_dim, graph_dim, hops)
        self.classifier = nn.Sequential(nn.Linear(graph_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, num_classes))

    def forward(self, inputs):
        x = self.face_cnn(inputs.geometry) if self.face_cnn is not None else inputs.geometry
        x = torch.relu(self.conv(x, inputs.laplacian))
        return self.classifier(x)
