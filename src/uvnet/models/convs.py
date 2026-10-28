"""Edge-conditioned convolution used by UV-Net."""

from __future__ import annotations

import torch
import dgl.function as fn
from torch import nn


class EdgeConditionedConv(nn.Module):
    """Edge-conditioned convolution (Simonovsky & Komodakis, 2017).

    ``h_i' = W_self h_i + (1/|N(i)|) sum_{j in N(i)} F(e_ij) h_j`` where ``F`` maps an
    edge embedding to a ``in_dim x out_dim`` matrix. This is the graph operator used by
    UV-Net for propagating surface features along trim-curve (edge) features.
    """

    def __init__(self, in_dim, out_dim, edge_dim, hidden=128):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.edge_network = nn.Sequential(nn.Linear(edge_dim, hidden), nn.ReLU(),
                                          nn.Linear(hidden, in_dim * out_dim))
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, graph, node_feat, edge_feat):
        with graph.local_scope():
            weights = self.edge_network(edge_feat).view(-1, self.in_dim, self.out_dim)
            src, dst = graph.edges()
            messages = torch.bmm(node_feat[src].unsqueeze(1), weights).squeeze(1)
            graph.edata["_message"] = messages
            graph.update_all(fn.copy_e("_message", "_message"), fn.sum("_message", "_sum"))
            aggregated = graph.ndata["_sum"]
        degree = graph.in_degrees().clamp_min(1).to(node_feat.dtype).unsqueeze(-1)
        return self.self_linear(node_feat) + aggregated / degree


__all__ = ["EdgeConditionedConv"]
