"""Topology-only node descriptors (graphlet orbits, Laplacian eigenvectors).

``graphlet_degree_vectors``
    15-dimensional GDV from ``05_Shallow_Learning/code01``. Orbits 0-3 (edge, induced
    P3 endpoint, induced P3 centre, triangle) are exact; orbits 4-14 are the automorphism
    orbits of the connected 4-node graphlets. The course samples 500 rooted 4-node
    subsets per node and rescales; here all 4-subsets are enumerated exactly when the
    solid is small enough (``exact_limit``), otherwise 4-subsets are sampled uniformly
    and the per-node orbit counts are turned into frequencies.

``laplacian_positional_encoding``
    The first ``dim`` non-trivial eigenvectors of the symmetric normalized Laplacian
    ``L = I - D^{-1/2} A D^{-1/2}`` from ``07_Graph_Transformers/code02`` (the course
    skips the trivial eigenvector). Columns are sign-canonicalized so the encoding is
    deterministic; repeated eigenvalues still leave an arbitrary rotation inside each
    eigenspace, which is inherent to Laplacian eigenvectors.
"""

from __future__ import annotations

import itertools
import math

import dgl
import numpy as np
import torch

GDV_DIM = 15

ORBIT_NAMES = (
    "edge",
    "P3 endpoint",
    "P3 centre",
    "triangle",
    "P4 endpoint",
    "P4 internal",
    "3-star leaf",
    "3-star centre",
    "4-cycle",
    "tailed-triangle tail",
    "tailed-triangle degree-2",
    "tailed-triangle attachment",
    "diamond degree-2",
    "diamond degree-3",
    "K4",
)

# Connected 4-node graphlets: sorted degree sequence -> {root degree: orbit index}.
_FOUR_NODE_ORBITS = {
    (1, 1, 2, 2): {1: 4, 2: 5},      # P4
    (1, 1, 1, 3): {1: 6, 3: 7},      # K1,3
    (2, 2, 2, 2): {2: 8},            # C4
    (1, 2, 2, 3): {1: 9, 2: 10, 3: 11},  # tailed triangle
    (2, 2, 3, 3): {2: 12, 3: 13},    # diamond
    (3, 3, 3, 3): {3: 14},           # K4
}
_DEGREE_WEIGHTS = np.array([64, 16, 4, 1], dtype=np.int64)


def simple_adjacency(graph) -> np.ndarray:
    """Dense symmetric 0/1 adjacency of one solid, without self-loops."""
    simple = dgl.to_simple(dgl.remove_self_loop(graph))
    count = simple.num_nodes()
    adjacency = np.zeros((count, count), dtype=np.int8)
    if simple.num_edges():
        src, dst = simple.edges()
        adjacency[src.numpy(), dst.numpy()] = 1
        adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


def _four_node_orbits(adjacency: np.ndarray, subsets: np.ndarray) -> np.ndarray:
    """Orbit index (4-14) of every root of every 4-node subset; -1 if disconnected."""
    block = adjacency[subsets[:, :, None], subsets[:, None, :]].astype(np.float64)
    degrees = block.sum(axis=2)
    squared = block @ block
    reachable = block + squared + squared @ block
    connected = (reachable > 0).all(axis=(1, 2))
    key = (np.sort(degrees, axis=1) * _DEGREE_WEIGHTS).sum(axis=1)
    orbits = np.full(subsets.shape, -1, dtype=np.int64)
    for pattern, mapping in _FOUR_NODE_ORBITS.items():
        mask = connected & (key == (np.array(pattern) * _DEGREE_WEIGHTS).sum())
        for root_degree, orbit in mapping.items():
            orbits[mask] = np.where(degrees[mask] == root_degree, orbit, orbits[mask])
    return orbits


def _four_node_subsets(count: int, samples: int, exact_limit: int, seed: int) -> np.ndarray:
    if math.comb(count, 4) <= exact_limit:
        return np.array(list(itertools.combinations(range(count), 4)), dtype=np.int64)
    draws = max(1, min(samples, math.comb(count - 1, 3)))
    rng = np.random.default_rng(seed)
    # Uniform 4-subset: keep the four smallest of ``count`` random scores.
    return rng.random((draws, count)).argpartition(4, axis=1)[:, :4]


def graphlet_degree_vectors(graph, samples: int = 20000, seed: int = 0,
                            exact_limit: int = 100000) -> torch.Tensor:
    """15-dim GDV per node, scaled the same way as the course lab (0-1 ranges)."""
    adjacency = simple_adjacency(graph)
    count = adjacency.shape[0]
    dense = adjacency.astype(np.float64)
    degree = dense.sum(axis=1)
    common = dense @ dense
    triangles = (dense * common).sum(axis=1) / 2.0
    p3_centre = np.maximum(degree * (degree - 1.0) / 2.0 - triangles, 0.0)
    p3_endpoint = dense @ (degree - 1.0) - (dense * common).sum(axis=1)

    counts = np.zeros((count, GDV_DIM), dtype=np.float64)
    counts[:, 0] = degree
    counts[:, 1] = p3_endpoint
    counts[:, 2] = p3_centre
    counts[:, 3] = triangles

    if count >= 4:
        subsets = _four_node_subsets(count, samples, exact_limit, seed)
        orbits = _four_node_orbits(adjacency, subsets)
        valid = orbits >= 0
        subset_index, root_index = np.nonzero(valid)
        roots = subsets[subset_index, root_index]
        np.add.at(counts, (roots, orbits[subset_index, root_index]), 1.0)
        appearances = np.bincount(roots, minlength=count).astype(np.float64)
        seen = appearances > 0
        counts[seen, 4:] /= appearances[seen, None]

    counts[:, 0] /= max(count - 1, 1)
    counts[:, 1:4] /= max(math.comb(count - 1, 2), 1)
    return torch.from_numpy(counts.astype(np.float32))


def laplacian_positional_encoding(graph, dim: int = 8) -> torch.Tensor:
    """First ``dim`` non-trivial normalized-Laplacian eigenvectors, sign-canonicalized."""
    adjacency = simple_adjacency(graph).astype(np.float64)
    count = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.zeros_like(degree)
    np.divide(1.0, np.sqrt(degree), out=inverse_sqrt, where=degree > 0)
    laplacian = np.eye(count) - inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    vectors = vectors[:, 1:dim + 1]
    for column in range(vectors.shape[1]):
        pivot = np.argmax(np.abs(vectors[:, column]))
        if vectors[pivot, column] < 0:
            vectors[:, column] = -vectors[:, column]
    if vectors.shape[1] < dim:
        vectors = np.pad(vectors, ((0, 0), (0, dim - vectors.shape[1])))
    return torch.from_numpy(np.ascontiguousarray(vectors.astype(np.float32)))
