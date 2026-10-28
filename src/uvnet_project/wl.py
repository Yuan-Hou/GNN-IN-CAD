"""Rooted 1-WL colour histories, adapting Shallow Learning Lab 03 to faces."""

import hashlib

import torch


def wl_hash(current_color, neighbor_colors):
    """Use the tutorial's deterministic permutation-invariant BLAKE2b hash."""
    signature = repr((current_color, tuple(sorted(neighbor_colors)))).encode("utf-8")
    return hashlib.blake2b(signature, digest_size=16).hexdigest()


def wl_node_colors(graph, iterations=3):
    """Return each face's colour at rounds 0..iterations, in node order.

    All nodes begin with '0'; no semantic/target labels or geometry enter
    refinement. Work on one simple undirected solid graph. This is fixed
    topology preprocessing, not a learned message-passing network.
    """
    if iterations < 0:
        raise ValueError("iterations must be nonnegative.")
    if graph.batch_size != 1:
        raise ValueError("Compute WL separately for each solid.")
    n = graph.num_nodes()
    neighbors = [set() for _ in range(n)]
    src, dst = graph.edges()
    for u, v in zip(src.cpu().tolist(), dst.cpu().tolist()):
        if u != v:
            neighbors[u].add(v)
            neighbors[v].add(u)
    colors = ["0"] * n
    histories = [[color] for color in colors]
    for _ in range(iterations):
        colors = [wl_hash(colors[v], [colors[u] for u in neighbors[v]]) for v in range(n)]
        for row, color in zip(histories, colors):
            row.append(color)
    return histories


def fit_wl_vocabulary(graph_histories, iterations=3):
    """Fit on training nodes only; each round reserves a distinct OOV ID."""
    if iterations < 0:
        raise ValueError("iterations must be nonnegative.")
    colors = [set() for _ in range(iterations + 1)]
    for histories in graph_histories:
        for row in histories:
            if len(row) != iterations + 1:
                raise ValueError("Unexpected number of WL rounds.")
            for step, color in enumerate(row):
                colors[step].add(color)
    ordered = [sorted(values) for values in colors]
    offsets, size = [], 0
    for values in ordered:
        offsets.append(size)
        size += len(values) + 1  # offset itself is the unknown colour
    return dict(iterations=iterations, colors=ordered, unknown_indices=offsets, size=size)


def wl_color_maps(vocabulary):
    return [{color: offset + 1 + i for i, color in enumerate(values)}
            for values, offset in zip(vocabulary["colors"], vocabulary["unknown_indices"])]


def encode_wl_colors(histories, vocabulary, maps=None):
    """Encode categories, never feed hash/ordinal numeric values to a linear layer."""
    maps = wl_color_maps(vocabulary) if maps is None else maps
    rounds = vocabulary["iterations"] + 1
    tokens = torch.empty((len(histories), rounds), dtype=torch.long)
    for node, row in enumerate(histories):
        if len(row) != rounds:
            raise ValueError("Unexpected number of WL rounds.")
        tokens[node] = torch.tensor([maps[step].get(color, vocabulary["unknown_indices"][step])
                                     for step, color in enumerate(row)], dtype=torch.long)
    return tokens
