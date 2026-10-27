"""Three UV-grid baselines with a common DGLGraph -> per-face logits API."""

import dgl
import torch
from torch import nn
from dgl.nn import GraphConv

MODEL_NAMES = ("cnn", "complete_gcn", "adjacency_gcn")


class SurfaceEncoder(nn.Module):
    """UV-Net style surface CNN; per-face normalization also handles singletons."""

    def __init__(self, width=64):
        super().__init__()
        layers = []
        channels = 7
        for out_channels in (64, 128, 256):
            layers.extend((nn.Conv2d(channels, out_channels, 3, padding=1, bias=False),
                           nn.GroupNorm(8, out_channels), nn.LeakyReLU()))
            channels = out_channels
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                     nn.Linear(256, width), nn.LayerNorm(width), nn.LeakyReLU())

    def forward(self, uv):
        return self.network(uv.permute(0, 3, 1, 2).contiguous())


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
    def __init__(self, model="cnn", width=64, num_classes=16, dropout=0.1):
        super().__init__()
        if model not in MODEL_NAMES:
            raise ValueError(f"Unknown model {model!r}")
        self.model_name = model
        self.encoder = SurfaceEncoder(width)
        # Construct before graph layers so the shared modules initialize identically per seed.
        self.classifier = nn.Sequential(nn.Linear(width, 128), nn.ReLU(),
                                        nn.Dropout(dropout), nn.Linear(128, num_classes))
        self.convs = nn.ModuleList([] if model == "cnn" else
                                   [GraphConv(width, width, norm="both") for _ in range(2)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        features = self.encoder(graph.ndata["x"])
        for conv in self.convs:
            message = (complete_convolution(graph, features, conv)
                       if self.model_name == "complete_gcn" else conv(graph, features))
            features = features + self.dropout(self.activation(message))
        return self.classifier(features)
