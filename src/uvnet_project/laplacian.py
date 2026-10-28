"""Per-solid Laplacian features, following the course's LapEig encoding."""

import torch


def laplacian_positional_encoding(graph, pos_enc_dim: int) -> torch.Tensor:
    """Return (faces, pos_enc_dim) nontrivial normalized-Laplacian eigenvectors.

    Call on one solid, never a batch of disconnected solids. Preserve the
    stored adjacency multiplicity and self-loops, as in the course's dense
    adjacency construction. The graph must have symmetric directed adjacency.
    ``eigh`` gives real orthonormal eigenvectors sorted by eigenvalue for this
    symmetric Laplacian. Drop the zero eigenspace; pad missing columns with 0.
    Sign augmentation belongs in training, not in this cached computation.
    Repeated-eigenvalue basis ambiguity is not resolved by this baseline.
    """
    if pos_enc_dim < 1:
        raise ValueError("pos_enc_dim must be positive.")
    if graph.batch_size != 1:
        raise ValueError("Compute Laplacian features separately for each solid.")
    n = graph.num_nodes()
    src, dst = graph.edges()
    adjacency = torch.bincount(
        src.cpu() * n + dst.cpu(), minlength=n * n
    ).reshape(n, n).double()
    if not torch.equal(adjacency, adjacency.T):
        raise ValueError("Expected symmetric adjacency for an undirected solid graph.")
    degree = adjacency.sum(1)
    inv_sqrt = degree.clamp_min(1).pow(-0.5)
    laplacian = torch.eye(n, dtype=torch.float64) - (
        inv_sqrt[:, None] * adjacency * inv_sqrt[None, :]
    )
    # Treat isolated nodes as a zero block of the normalized Laplacian.
    laplacian[degree == 0, degree == 0] = 0
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    nontrivial = eigenvectors[:, eigenvalues > 1e-8][:, :pos_enc_dim]
    encoding = torch.zeros(n, pos_enc_dim, dtype=torch.float32)
    encoding[:, :nontrivial.shape[1]] = nontrivial.float()
    return encoding
