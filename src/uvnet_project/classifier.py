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
