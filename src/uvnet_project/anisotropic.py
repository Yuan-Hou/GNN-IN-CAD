"""Edge-gated face message passing adapted from Graph Convnets Lab 04/05."""

from dataclasses import dataclass, fields

import torch
from torch import nn
from torch.nn import functional as F

from .edge_cnn import EdgeCNN


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
