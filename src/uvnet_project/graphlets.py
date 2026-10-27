"""Per-node induced graphlet orbit counts, extending Shallow Learning Lab 01.

The tutorial's 15 orbit IDs (2--4 nodes) are preserved. Five-node orbits
use deterministic rooted canonical-mask order, not standard ORCA numbering.
Bit-mask isomorphism lookup follows Lab 02. Counts here are exact: grow
connected subsets once each instead of Monte Carlo rooted subset sampling.
"""

from functools import lru_cache
from itertools import combinations, permutations
import math

import numpy as np
import torch


ORBIT_NAMES_4 = [
    "edge", "P3 endpoint", "P3 center", "triangle", "P4 endpoint", "P4 internal",
    "3-star leaf", "3-star center", "4-cycle", "tailed-triangle tail",
    "tailed-triangle degree-2", "tailed-triangle attachment", "diamond degree-2",
    "diamond degree-3", "K4",
]


@lru_cache(None)
def _rooted_masks(k):
    pairs = list(combinations(range(k), 2))
    pair_bit = {pair: bit for bit, pair in enumerate(pairs)}
    masks = np.arange(1 << len(pairs), dtype=np.int64)
    rooted = np.full((len(masks), k), 1 << len(pairs), dtype=np.int64)
    for root in range(k):
        for rest in permutations([node for node in range(k) if node != root]):
            order = (root, *rest)
            remapped = np.zeros_like(masks)
            for bit, (u, v) in enumerate(pairs):
                old_pair = tuple(sorted((order[u], order[v])))
                remapped |= ((masks >> pair_bit[old_pair]) & 1) << bit
            rooted[:, root] = np.minimum(rooted[:, root], remapped)
    connected = np.zeros(len(masks), dtype=bool)
    for mask in masks:
        neighbors = [0] * k
        for bit, (u, v) in enumerate(pairs):
            if (int(mask) >> bit) & 1:
                neighbors[u] |= 1 << v
                neighbors[v] |= 1 << u
        seen, frontier = 1, 1
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            node = bit.bit_length() - 1
            new = neighbors[node] & ~seen
            seen |= new
            frontier |= new
        connected[mask] = seen == (1 << k) - 1
    return rooted, connected


def _mask(k, edges):
    edge_set = {tuple(sorted(edge)) for edge in edges}
    return sum(1 << bit for bit, pair in enumerate(combinations(range(k), 2))
               if pair in edge_set)


@lru_cache(None)
def orbit_catalog(max_nodes=4):
    """Return lookup tables, orbit sizes and labels for the selected feature set."""
    if max_nodes not in (4, 5):
        raise ValueError("max_nodes must be 4 or 5.")
    prototypes = [
        (2, [(0, 1)], [(0, 0)]),
        (3, [(0, 1), (1, 2)], [(0, 1), (1, 2)]),
        (3, [(0, 1), (1, 2), (0, 2)], [(0, 3)]),
        (4, [(0, 1), (1, 2), (2, 3)], [(0, 4), (1, 5)]),
        (4, [(0, 1), (0, 2), (0, 3)], [(1, 6), (0, 7)]),
        (4, [(0, 1), (1, 2), (2, 3), (3, 0)], [(0, 8)]),
        (4, [(0, 1), (1, 2), (0, 2), (2, 3)], [(3, 9), (0, 10), (2, 11)]),
        (4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3)], [(2, 12), (0, 13)]),
        (4, list(combinations(range(4), 2)), [(0, 14)]),
    ]
    mapping = {}
    for k, edges, roles in prototypes:
        rooted, _ = _rooted_masks(k)
        for root, orbit in roles:
            mapping[(k, int(rooted[_mask(k, edges), root]))] = orbit
    names = ORBIT_NAMES_4.copy()
    sizes = [2] + [3] * 3 + [4] * 11
    if max_nodes == 5:
        rooted, connected = _rooted_masks(5)
        keys = sorted(set(rooted[connected].ravel().tolist()))
        assert len(keys) == 58
        for key in keys:
            mapping[(5, key)] = len(names)
            names.append(f"5-node rooted canonical mask {key}")
            sizes.append(5)
    lookups = {}
    for k in range(2, max_nodes + 1):
        rooted, connected = _rooted_masks(k)
        lookup = np.full(rooted.shape, -1, dtype=np.int64)
        for mask in np.flatnonzero(connected):
            lookup[mask] = [mapping[(k, int(key))] for key in rooted[mask]]
        lookups[k] = lookup
    return lookups, np.array(sizes), names


def _nodes_in_subset(subset):
    nodes = []
    while subset:
        bit = subset & -subset
        subset ^= bit
        nodes.append(bit.bit_length() - 1)
    return nodes


def graphlet_degree_vectors(graph, max_nodes=4, normalize=True):
    """Compute one GDV row per face on one simple undirected solid graph.

    Collapse parallel edges and remove self-loops as in the tutorial. Count
    every connected induced subset once and assign each participating node
    its automorphism orbit. Normalize size-k orbit counts by C(n-1, k-1).
    Precompute/cache once: exact 5-node counting can be costly on large graphs.
    """
    if graph.batch_size != 1:
        raise ValueError("Compute graphlets separately for each solid.")
    lookups, sizes, _ = orbit_catalog(max_nodes)
    n = graph.num_nodes()
    counts = np.zeros((n, len(sizes)), dtype=np.int64)
    if n == 0:
        return torch.from_numpy(counts).float() if normalize else torch.from_numpy(counts)
    src, dst = graph.edges()
    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[src.cpu().numpy(), dst.cpu().numpy()] = True
    adjacency |= adjacency.T.copy()
    np.fill_diagonal(adjacency, False)
    neighbor_bits = [sum(1 << int(v) for v in np.flatnonzero(adjacency[u])) for u in range(n)]
    subsets = {1 << node for node in range(n)}
    for k in range(2, min(max_nodes, n) + 1):
        expanded = set()
        for subset in subsets:
            remaining, frontier = subset, 0
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                frontier |= neighbor_bits[bit.bit_length() - 1]
            frontier &= ~subset
            while frontier:
                bit = frontier & -frontier
                frontier ^= bit
                expanded.add(subset | bit)
        subsets = expanded
        if not subsets:
            break
        nodes = np.array([_nodes_in_subset(subset) for subset in sorted(subsets)], dtype=np.int64)
        masks = np.zeros(len(nodes), dtype=np.int64)
        for bit, (u, v) in enumerate(combinations(range(k), 2)):
            masks |= adjacency[nodes[:, u], nodes[:, v]].astype(np.int64) << bit
        orbits = lookups[k][masks]
        np.add.at(counts, (nodes.ravel(), orbits.ravel()), 1)
    if not normalize:
        return torch.from_numpy(counts)
    denominators = np.array([max(math.comb(n - 1, int(k) - 1), 1) if n >= k else 1
                             for k in sizes])
    return torch.from_numpy((counts / denominators).astype(np.float32))
