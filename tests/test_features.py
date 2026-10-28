"""Checks for the GDV and Laplacian positional encoding implementations."""

import unittest

import dgl
import numpy as np
import torch

from uvnet.data.features import (GDV_DIM, graphlet_degree_vectors,
                                 laplacian_positional_encoding)


def build(count, edges):
    src = [left for left, _ in edges]
    dst = [right for _, right in edges]
    return dgl.add_self_loop(dgl.graph((src, dst), num_nodes=count))


def random_graph(count, probability, seed):
    rng = np.random.default_rng(seed)
    upper = np.triu((rng.random((count, count)) < probability).astype(np.int8), 1)
    adjacency = upper + upper.T
    src, dst = np.nonzero(adjacency)
    return dgl.add_self_loop(dgl.graph((src.tolist(), dst.tolist()), num_nodes=count))


def orbit(graph, index, node):
    return float(graphlet_degree_vectors(graph)[node, index])


class GraphletDegreeVectorTests(unittest.TestCase):
    def test_k4_is_a_single_four_clique_orbit(self):
        graph = build(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
        vectors = graphlet_degree_vectors(graph)
        self.assertEqual(tuple(vectors.shape), (4, GDV_DIM))
        for node in range(4):
            self.assertAlmostEqual(float(vectors[node, 0]), 1.0)   # degree 3 / (n-1)
            self.assertAlmostEqual(float(vectors[node, 3]), 1.0)   # triangle
            self.assertAlmostEqual(float(vectors[node, 14]), 1.0)  # K4
        others = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 1, 2]
        self.assertTrue(torch.all(vectors[:, others] == 0))

    def test_star_splits_centre_and_leaf_orbits(self):
        graph = build(4, [(0, 1), (0, 2), (0, 3)])
        vectors = graphlet_degree_vectors(graph)
        self.assertAlmostEqual(float(vectors[0, 2]), 1.0)          # P3 centre
        self.assertAlmostEqual(float(vectors[0, 7]), 1.0)          # 3-star centre
        for leaf in (1, 2, 3):
            self.assertAlmostEqual(float(vectors[leaf, 0]), 1 / 3)     # degree 1
            self.assertAlmostEqual(float(vectors[leaf, 1]), 2 / 3)     # P3 endpoint
            self.assertAlmostEqual(float(vectors[leaf, 6]), 1.0)       # 3-star leaf
            self.assertAlmostEqual(float(vectors[leaf, 7]), 0.0)

    def test_path_and_cycle(self):
        path = build(4, [(0, 1), (1, 2), (2, 3)])
        vectors = graphlet_degree_vectors(path)
        self.assertAlmostEqual(float(vectors[0, 4]), 1.0)   # P4 endpoint
        self.assertAlmostEqual(float(vectors[1, 5]), 1.0)   # P4 internal
        self.assertAlmostEqual(float(vectors[0, 5]), 0.0)
        cycle = build(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
        vectors = graphlet_degree_vectors(cycle)
        for node in range(4):
            self.assertAlmostEqual(float(vectors[node, 8]), 1.0)     # 4-cycle
            self.assertAlmostEqual(float(vectors[node, 0]), 2 / 3)   # degree 2
            self.assertAlmostEqual(float(vectors[node, 2]), 1 / 3)   # P3 centre

    def test_tailed_triangle_and_diamond(self):
        tailed = build(4, [(0, 1), (1, 2), (2, 0), (0, 3)])
        vectors = graphlet_degree_vectors(tailed)
        self.assertAlmostEqual(float(vectors[3, 9]), 1.0)    # tail
        self.assertAlmostEqual(float(vectors[1, 10]), 1.0)   # degree-2 triangle node
        self.assertAlmostEqual(float(vectors[0, 11]), 1.0)   # attachment
        diamond = build(4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])
        vectors = graphlet_degree_vectors(diamond)
        self.assertAlmostEqual(float(vectors[0, 13]), 1.0)
        self.assertAlmostEqual(float(vectors[0, 12]), 0.0)
        self.assertAlmostEqual(float(vectors[1, 12]), 1.0)
        self.assertAlmostEqual(float(vectors[1, 13]), 0.0)

    def test_values_are_bounded_and_finite(self):
        graph = random_graph(20, 0.2, seed=3)
        vectors = graphlet_degree_vectors(graph)
        self.assertTrue(torch.isfinite(vectors).all())
        self.assertGreaterEqual(float(vectors.min()), 0.0)
        self.assertLessEqual(float(vectors.max()), 1.0 + 1e-6)

    def test_permutation_equivariance(self):
        graph = random_graph(12, 0.25, seed=5)
        permutation = torch.from_numpy(np.random.default_rng(0).permutation(12))
        inverse = torch.empty(12, dtype=torch.long)
        inverse[permutation] = torch.arange(12)
        src, dst = dgl.remove_self_loop(graph).edges()
        permuted = build(12, list(zip(permutation[src].tolist(), permutation[dst].tolist())))
        torch.testing.assert_close(graphlet_degree_vectors(permuted),
                                   graphlet_degree_vectors(graph)[inverse], atol=1e-6, rtol=1e-6)

    def test_monte_carlo_is_deterministic_and_close_to_exact(self):
        graph = random_graph(30, 5 / 29, seed=11)
        exact = graphlet_degree_vectors(graph, exact_limit=10 ** 9)
        sampled = graphlet_degree_vectors(graph, samples=4000, seed=0, exact_limit=0)
        repeated = graphlet_degree_vectors(graph, samples=4000, seed=0, exact_limit=0)
        torch.testing.assert_close(sampled, repeated)
        self.assertLess(float((sampled - exact).abs().mean()), 0.05)


class LaplacianPositionalEncodingTests(unittest.TestCase):
    def test_columns_are_orthonormal_and_deterministic(self):
        graph = random_graph(15, 0.25, seed=7)
        encoding = laplacian_positional_encoding(graph, dim=8)
        self.assertEqual(tuple(encoding.shape), (15, 8))
        torch.testing.assert_close(encoding.T @ encoding, torch.eye(8), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(encoding, laplacian_positional_encoding(graph, dim=8))

    def test_sign_is_canonical(self):
        graph = random_graph(15, 0.25, seed=7)
        encoding = laplacian_positional_encoding(graph, dim=8)
        for column in range(encoding.shape[1]):
            pivot = int(torch.argmax(encoding[:, column].abs()))
            self.assertGreaterEqual(float(encoding[pivot, column]), 0.0)

    def test_short_solids_are_zero_padded(self):
        graph = build(3, [(0, 1), (1, 2), (2, 0)])
        encoding = laplacian_positional_encoding(graph, dim=8)
        self.assertEqual(tuple(encoding.shape), (3, 8))
        self.assertTrue(torch.all(encoding[:, 2:] == 0))
        torch.testing.assert_close(encoding[:, :2].T @ encoding[:, :2], torch.eye(2), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
