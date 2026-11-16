from __future__ import annotations

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


"""Classification head for face embeddings."""

import torch
from torch import nn


class Classifier(nn.Module):
    """Two-layer MLP: embedding -> hidden -> class logits.

    Follows the two-layer description in UV-Net Section 4.2 and the
    Linear/ReLU pattern in the course notebooks. The released UV-Net code
    instead uses three linear layers with batch normalization and dropout.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int, num_classes: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings):
        """Map an (N, embedding_dim) tensor to (N, num_classes) logits."""
        h = torch.relu(self.linear1(embeddings))
        return self.linear2(h)


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


"""Edge-gated face message passing adapted from Graph Convnets Lab 04/05."""

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F




@dataclass
class FaceEdgeBatch:
    geometry: torch.Tensor
    edge_grids: torch.Tensor
    edge_index: torch.Tensor  # source, destination in original DGL edge-ID order
    node_scale: torch.Tensor
    edge_scale: torch.Tensor

    def to(self, device, non_blocking=False):
        tensors = {field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
                   for field in fields(self)}
        if tensors["geometry"].ndim == 4 and torch.device(device).type == "cuda":
            tensors["geometry"] = tensors["geometry"].contiguous(memory_format=torch.channels_last)
        return FaceEdgeBatch(**tensors)

    def pin_memory(self):
        return FaceEdgeBatch(**{field.name: getattr(self, field.name).pin_memory()
                               for field in fields(self)})


def collate_face_edges(batch):
    geometry, grids, indices, labels, ns, es = [], [], [], [], [], []
    offset = 0
    for face_features, edge_grids, edge_index, target in batch:
        n, e = face_features.shape[0], edge_grids.shape[0]
        if edge_index.shape != (2, e) or target.shape != (n,):
            raise ValueError("Edge grids, edge indices and face labels must align.")
        if n < 1 or (e and (edge_index.min() < 0 or edge_index.max() >= n)):
            raise ValueError("Invalid local face indices.")
        geometry.append(face_features)
        grids.append(edge_grids)
        indices.append(edge_index + offset)
        labels.append(target)
        ns.append(torch.full((n, 1), n ** -0.5))
        es.append(torch.full((e, 1), max(e, 1) ** -0.5))
        offset += n
    return FaceEdgeBatch(torch.cat(geometry), torch.cat(grids), torch.cat(indices, dim=1),
                         torch.cat(ns), torch.cat(es)), torch.cat(labels)


def _batch_norm(module, x):
    # BatchNorm has no sample variance with 0/1 rows. Use saved statistics.
    if x.shape[0] < 2 and module.training:
        return F.batch_norm(x, module.running_mean, module.running_var,
                            module.weight, module.bias, training=False, eps=module.eps)
    return module(x)


class GatedFaceLayer(nn.Module):
    """Tutorial residual gated update; static edges omit endpoint/state updates.

    Gates are feature-wise sigmoid weights, normalized over incoming edges.
    The dynamic variant uses q_ij=C e_ij+D h_i+E h_j for the current gate,
    then carries e+ReLU(BN(q*graph_size_scale)) to the next graph layer.
    """

    def __init__(self, width, edge_mode, carry_edges=True):
        super().__init__()
        if edge_mode not in ("static", "updated"):
            raise ValueError("edge_mode must be static or updated.")
        self.edge_mode = edge_mode
        self.carry_edges = edge_mode == "updated" and carry_edges
        self.A = nn.Linear(width, width)
        self.B = nn.Linear(width, width)
        self.C = nn.Linear(width, width)
        self.bn_node = nn.BatchNorm1d(width)
        if edge_mode == "updated":
            self.D = nn.Linear(width, width)
            self.E = nn.Linear(width, width)
        if self.carry_edges:
            self.bn_edge = nn.BatchNorm1d(width)

    def forward(self, h, e, inputs):
        # Float32 gate normalization and reductions avoid half-precision underflow.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h, e = h.float(), e.float()
            src, dst = inputs.edge_index
            q = self.C(e)
            if self.edge_mode == "updated":
                q = q + self.D(h)[dst] + self.E(h)[src]
            gate = torch.sigmoid(q)
            messages = gate * self.B(h)[src]
            numerator = h.new_zeros(h.shape).index_add(0, dst, messages)
            denominator = h.new_zeros(h.shape).index_add(0, dst, gate)
            node_value = self.A(h) + numerator / denominator.clamp_min(1e-6)
            updated_h = h + torch.relu(_batch_norm(self.bn_node, node_value * inputs.node_scale))
            if self.carry_edges:
                e = e + torch.relu(_batch_norm(self.bn_edge, q * inputs.edge_scale))
            return updated_h, e


class AnisotropicFaceClassifier(nn.Module):
    """Two residual gated layers and a pointwise MLP; no solid-level pooling.

    The EdgeCNN is learned in both variants. Static means its encoded state
    is not recurrently updated by the graph layers, not frozen CNN weights.
    """

    def __init__(self, embedding_dim, graph_dim, hidden_dim, edge_mode,
                 num_classes=16, edge_in_channels=6, face_cnn=None):
        super().__init__()
        self.face_cnn = face_cnn
        self.edge_cnn = EdgeCNN(embedding_dim=graph_dim, in_channels=edge_in_channels)
        self.face_projection = nn.Linear(embedding_dim, graph_dim)
        self.layers = nn.ModuleList([
            GatedFaceLayer(graph_dim, edge_mode, carry_edges=True),
            # Final edge carry would never reach the face loss: omit that dead branch.
            GatedFaceLayer(graph_dim, edge_mode, carry_edges=False),
        ])
        self.classifier = nn.Sequential(nn.Linear(graph_dim, hidden_dim), nn.ReLU(),
                                        nn.Linear(hidden_dim, num_classes))

    def forward(self, inputs):
        geometry = self.face_cnn(inputs.geometry) if self.face_cnn is not None else inputs.geometry
        h = self.face_projection(geometry)
        if inputs.edge_grids.shape[0] == 0:
            e = h.new_empty((0, h.shape[1]))
        elif inputs.edge_grids.shape[0] == 1 and self.edge_cnn.training:
            # Only the final edge FC BatchNorm needs a fallback with one curve.
            bn = self.edge_cnn.fc[1]
            was_training = bn.training
            try:
                bn.eval()
                e = self.edge_cnn(inputs.edge_grids)
            finally:
                bn.train(was_training)
        else:
            e = self.edge_cnn(inputs.edge_grids)
        for layer in self.layers:
            h, e = layer(h, e, inputs)
        return self.classifier(h)
