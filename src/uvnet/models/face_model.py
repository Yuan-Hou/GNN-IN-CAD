"""Per-face classification models: geometry-only, topology-only and full UV-Net."""

from __future__ import annotations

import dgl
import torch
from torch import nn
from dgl.nn import GraphConv

from ..data.graphs import NUM_CLASSES
from .convs import EdgeConditionedConv
from .encoders import CurveEncoder, SurfaceEncoder, TopologyEncoder

MODEL_NAMES = ("cnn", "complete_gcn", "adjacency_gcn", "topo_gdv",
               "adjacency_gcn_lape", "uvnet", "uvnet_lape")

# Models that need an extra per-node or per-edge input from the dataset.
MODEL_FEATURES = {"topo_gdv": ("gdv",),
                  "adjacency_gcn_lape": ("pe",),
                  "uvnet": ("edge",),
                  "uvnet_lape": ("edge", "pe")}


def complete_graph(graph):
    """Explicit disjoint complete graphs, including self-loops (reference path)."""
    graphs = []
    for count in graph.batch_num_nodes().tolist():
        nodes = torch.arange(count, device=graph.device, dtype=graph.idtype)
        graphs.append(dgl.graph((nodes.repeat_interleave(count), nodes.repeat(count)),
                                num_nodes=count, device=graph.device))
    return dgl.batch(graphs)


def complete_convolution(graph, features, conv):
    """Exactly GraphConv(norm='both') on each complete graph, without F² edges."""
    with graph.local_scope():
        graph.ndata["_features"] = features
        means = dgl.mean_nodes(graph, "_features")
    pooled = means @ conv.weight
    if conv.bias is not None:
        pooled = pooled + conv.bias
    return dgl.broadcast_nodes(graph, pooled)


class FaceModel(nn.Module):
    """One class per information source; every model returns per-face logits."""

    def __init__(self, model="cnn", width=64, num_classes=NUM_CLASSES, dropout=0.1,
                 pe_dim=8):
        super().__init__()
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model!r}")
        self.model_name = model
        self.encoder = TopologyEncoder(width) if model == "topo_gdv" else SurfaceEncoder(width)
        # Construct before graph layers so the shared modules initialize identically per seed.
        self.classifier = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))
        if model.startswith("uvnet"):
            self.convs = nn.ModuleList([EdgeConditionedConv(width, width, width)
                                        for _ in range(2)])
        else:
            self.convs = nn.ModuleList([] if model == "cnn" else
                                       [GraphConv(width, width, norm="both") for _ in range(2)])
        self.curve_encoder = CurveEncoder(width) if model.startswith("uvnet") else None
        # Built after ``convs`` so the shared encoder/classifier/graph layers keep the
        # exact initialization of the baselines for a given seed.
        self.pe_projection = (nn.Linear(width + pe_dim, width)
                              if model in ("adjacency_gcn_lape", "uvnet_lape") else None)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        if self.model_name == "topo_gdv":
            features = self.encoder(graph.ndata["gdv"].float())
        else:
            features = self.encoder(graph.ndata["x"])
            if self.pe_projection is not None:
                features = self.pe_projection(
                    torch.cat((features, graph.ndata["pe"].float()), dim=-1))
        if self.curve_encoder is not None:
            curves = self.curve_encoder(graph.edata["xs"].float())
            for conv in self.convs:
                features = features + self.dropout(self.activation(conv(graph, features, curves)))
            return self.classifier(features)
        for conv in self.convs:
            message = (complete_convolution(graph, features, conv)
                       if self.model_name == "complete_gcn" else conv(graph, features))
            features = features + self.dropout(self.activation(message))
        return self.classifier(features)


__all__ = ["MODEL_NAMES", "MODEL_FEATURES", "FaceModel", "complete_graph",
           "complete_convolution"]
