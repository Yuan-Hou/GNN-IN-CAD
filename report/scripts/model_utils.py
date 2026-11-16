from __future__ import annotations

"""Draw the baseline architecture from its layer dimensions."""
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def draw_baseline_architecture():
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set(xlim=(0, 16), ylim=(0, 5))
    ax.axis('off')
    def strip(items, y, colour):
        width, gap = 1.88, 0.37
        start = (16 - len(items) * width - (len(items) - 1) * gap) / 2
        for i, text in enumerate(items):
            x = start + i * (width + gap)
            ax.add_patch(FancyBboxPatch((x, y), width, 0.95, boxstyle='round,pad=0.04',
                                       facecolor=colour, edgecolor='#35516b', linewidth=1.1))
            ax.text(x + width / 2, y + 0.475, text, ha='center', va='center', fontsize=9)
            if i < len(items) - 1:
                ax.annotate('', xy=(x + width + gap - 0.04, y + 0.475),
                            xytext=(x + width + 0.04, y + 0.475),
                            arrowprops=dict(arrowstyle='->', color='#35516b', lw=1.4))
    ax.text(0.25, 4.55, 'Face CNN: shared surface encoder', fontsize=13, weight='bold')
    strip(['Face grid\n7 × 10 × 10', 'Conv 3 × 3\n7 → 64\nBN + LeakyReLU',
           'Conv 3 × 3\n64 → 128\nBN + LeakyReLU', 'Conv 3 × 3\n128 → 256\nBN + LeakyReLU',
           'Average pooling\n10 × 10 → 1 × 1\n256 values',
           'Linear 256 → 64\nBN + LeakyReLU\n64D embedding'], 3.1, '#e5f0fa')
    ax.text(0.25, 2.45, 'Classifier: shared independently across faces', fontsize=13, weight='bold')
    strip(['64D face\nembedding', 'Linear\n64 → 128', 'ReLU', 'Linear\n128 → 16', '16 class logits\nper face'], 1, '#e8f2e5')
    ax.text(8, 0.35, 'Convolutions use stride 1 and padding 1; the 10 × 10 grid is preserved until pooling.\nPooling averages samples within a face, not faces within a solid.',
            ha='center', va='center', fontsize=10)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / 'figures'
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'baseline-architecture.svg', bbox_inches='tight')
    fig.savefig(out / 'baseline-architecture.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


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


def seed_everything(seed, deterministic=True):
    import os, random
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def draw_shallow_architectures():
    """Draw the three fixed-topology branches paired with face geometry."""
    variants = {
        'laplacian': ('8 Laplacian eigenvectors', '8D positional features', 'Concatenate\n64 + 8 = 72', 'Linear 72 → 128\nReLU → Linear 128 → 16'),
        'wl': ('Uniform initial colours\n3 rounds of 1-WL', 'Categorical colour history\nrounds 0–3', 'Geometry + one-hot history\n(sparse implementation)', 'Linear → 128\nReLU → Linear 128 → 16'),
        'graphlet': ('Induced graphlets\n2–4 nodes', '15 normalized orbit counts\ntraining-set standardization', 'Concatenate\n64 + 15 = 79', 'Linear 79 → 128\nReLU → Linear 128 → 16'),
    }
    out = Path(__file__).resolve().parents[1] / 'figures'
    for name, (method, descriptor, merge, head) in variants.items():
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.set(xlim=(0, 13), ylim=(0, 4)); ax.axis('off')
        def box(x, y, w, h, text, color):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#35516b'))
            ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
        def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#35516b',lw=1.3))
        box(.15,2.5,2,.8,'Face UV-grid\n7 × 10 × 10','#e5f0fa')
        box(2.7,2.5,2.5,.8,'Baseline face CNN\n64D embedding (frozen)','#e5f0fa')
        box(.15,.65,2,.8,'Face-adjacency graph','#fff0d6')
        box(2.7,.65,2.5,.8,method,'#fff0d6')
        box(5.8,.65,2.65,.8,descriptor,'#fff0d6')
        box(8.95,1.55,1.9,1.1,merge,'#e8f2e5')
        box(11.35,1.55,1.5,1.1,head,'#e8f2e5')
        arrow((2.2,2.9),(2.65,2.9)); arrow((2.2,1.05),(2.65,1.05))
        arrow((5.25,1.05),(5.75,1.05)); arrow((5.25,2.9),(8.9,2.25))
        arrow((8.5,1.05),(8.9,1.9)); arrow((10.9,2.1),(11.3,2.1))
        ax.text(6.5,3.7,f'Geometry + shallow topology: {name.capitalize() if name != "wl" else "Weisfeiler–Lehman"}',ha='center',fontsize=13,weight='bold')
        ax.text(6.5,.12,'Fixed topology descriptors; frozen face CNN; classifier trained per face.',ha='center',fontsize=9)
        fig.tight_layout()
        fig.savefig(out/f'shallow-{name}-architecture.svg',bbox_inches='tight')
        fig.savefig(out/f'shallow-{name}-architecture.png',dpi=140,bbox_inches='tight')
        plt.close(fig)


def scaled_laplacian(graph):
    """Return sparse L_tilde = 2 L / 2 - I = -D^-1/2 A D^-1/2.

    Use the normalized-Laplacian spectral upper bound 2, avoiding an
    eigendecomposition. L = I - normalized A, including isolated vertices.
    Collapse parallel edges and remove loops, as in the shallow themes.
    The zero-order Chebyshev term already preserves self information.
    """
    n = graph.num_nodes()
    src, dst = (values.cpu().long() for values in graph.edges())
    edges = torch.stack((src, dst))
    edges = edges[:, src != dst]
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    edges = torch.unique(edges, dim=1, sorted=True)
    degree = torch.bincount(edges[0], minlength=n).float()
    inv = degree.clamp_min(1).rsqrt()
    values = -inv[edges[0]] * inv[edges[1]]
    return torch.sparse_coo_tensor(edges, values, (n, n)).coalesce()




def draw_chebyshev_architecture():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set(xlim=(0,14),ylim=(0,4));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#35516b'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#35516b',lw=1.3))
    box(.15,2,1.75,.9,'Face UV-grid\n7 × 10 × 10','#e5f0fa')
    box(2.4,2,2,.9,'Baseline face CNN\n64D embeddings H','#e5f0fa')
    box(5.1,1.8,3.1,1.3,'Chebyshev graph filter\nT₀H, T₁H, T₂H\n192 → 64; ReLU','#fff0d6')
    box(8.9,2,2.1,.9,'Per-face MLP\n64 → 128 → 16','#e8f2e5')
    box(11.7,2,2,.9,'16 class logits\nper face','#e8f2e5')
    box(2.4,.45,2,.85,'Face-adjacency graph\nScaled Laplacian','#fff0d6')
    arrow((1.95,2.45),(2.35,2.45));arrow((4.45,2.45),(5.05,2.45))
    arrow((8.25,2.45),(8.85,2.45));arrow((11.05,2.45),(11.65,2.45));arrow((4.45,.9),(5.85,1.75))
    ax.text(7,3.6,'ChebNet GCN: one graph-convolution layer, two-hop reach',ha='center',fontsize=13,weight='bold')
    ax.text(7,.1,'Frozen: train graph filter + classifier. Joint: also fine-tune the face CNN. Three-hop variant adds T₃H.',ha='center',fontsize=9)
    fig.tight_layout()
    out=Path(__file__).resolve().parents[1]/'figures'
    fig.savefig(out/'chebyshev-architecture.svg',bbox_inches='tight')
    fig.savefig(out/'chebyshev-architecture.png',dpi=140,bbox_inches='tight')
    plt.close(fig)


def draw_edge_architectures():
    for mode in ('static','updated'):
        fig, ax = plt.subplots(figsize=(14,5))
        ax.set(xlim=(0,14),ylim=(0,5));ax.axis('off')
        def box(x,y,w,h,text,color):
            ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',facecolor=color,edgecolor='#35516b'))
            ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
        def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#35516b',lw=1.3))
        box(.15,3.05,1.9,.9,'Face UV-grid\n7 × 10 × 10','#e5f0fa')
        box(2.65,3.05,2.05,.9,'Face CNN → 64D\nLinear 64 → 64','#e5f0fa')
        box(5.45,3.05,2,.9,'Gated layer 1\n64D node states','#e8f2e5')
        box(8.2,3.05,2,.9,'Gated layer 2\n64D node states','#e8f2e5')
        box(10.95,3.05,2.8,.9,'Per-face MLP\n64 → 128 → 16 logits','#e8f2e5')
        box(.15,1,1.9,.9,'Curve grid\n6 × 10','#fff0d6')
        box(2.65,1,2.05,.9,'Edge CNN\n64D edge states e₀','#fff0d6')
        for a,b in [((2.1,3.5),(2.6,3.5)),((4.75,3.5),(5.4,3.5)),((7.5,3.5),(8.15,3.5)),((10.25,3.5),(10.9,3.5)),((2.1,1.45),(2.6,1.45)),((4.75,1.45),(5.8,3))]:arrow(a,b)
        if mode=='static':
            ax.plot([4.75,9.2],[1.45,1.45],color='#35516b',lw=1.3)
            arrow((9.2,1.45),(9.2,3));ax.text(7.2,1.15,'Same e₀ in both layers',ha='center',fontsize=9)
        else:
            box(5.45,1,2,.9,'Update edge states\ne₁ from e₀, hᵢ, hⱼ','#fff0d6')
            arrow((6.45,3),(6.45,1.95));arrow((7.5,1.45),(8.8,3))
        ax.text(7,4.55,f'Edge-conditioned model: {mode} edge states',ha='center',fontsize=13,weight='bold')
        ax.text(7,.35,'Two one-hop residual gated layers; face and curve encoders trained jointly; no whole-solid pooling.',ha='center',fontsize=10)
        fig.tight_layout()
        out=Path(__file__).resolve().parents[1]/'figures'
        fig.savefig(out/f'edge-{mode}-architecture.svg',bbox_inches='tight')
        fig.savefig(out/f'edge-{mode}-architecture.png',dpi=140,bbox_inches='tight')
        plt.close(fig)
