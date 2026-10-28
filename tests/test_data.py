"""Data-level checks: graph construction, curve-feature alignment, normalization."""

import unittest

import dgl
import torch

from uvnet.data import (DATA_DIR, MFCADDataset, NUM_CLASSES, adjacency_graph,
                        normalize_uv)
from uvnet.training import metrics_from_confusion


def raw_graph(count, edges, features=None):
    graph = dgl.graph(edges, num_nodes=count)
    if features is not None:
        graph.edata["xs"] = features
    return graph


class GraphConstructionTests(unittest.TestCase):
    def test_duplicate_and_self_loop_cleanup(self):
        graph = adjacency_graph(raw_graph(3, ([0, 0, 1, 1], [1, 1, 0, 1])))
        src, dst = graph.edges()
        self.assertEqual(set(zip(src.tolist(), dst.tolist())),
                         {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)})
        self.assertEqual(graph.num_edges(), 5)

    def test_curve_features_follow_their_directed_edges(self):
        edges = ([0, 1, 0, 1, 2, 0], [1, 0, 1, 2, 1, 0])
        features = torch.arange(len(edges[0])).view(-1, 1, 1).repeat(1, 10, 6).float()
        graph = adjacency_graph(raw_graph(3, edges, features), features)
        self.assertEqual(graph.edata["xs"].shape, (graph.num_edges(), 10, 6))
        src, dst = graph.edges()
        for index, (left, right) in enumerate(zip(src.tolist(), dst.tolist())):
            values = graph.edata["xs"][index].unique()
            if left == right:
                # Self-loops carry no curve: zero features keep the ECC self term learnable.
                self.assertTrue(torch.all(graph.edata["xs"][index] == 0))
            else:
                self.assertEqual(values.numel(), 1)
                # The first occurrence of a duplicate directed edge wins.
                expected = 0 if (left, right) == (0, 1) else \
                    edges[0].index(left) if (left, right) not in ((1, 0), (1, 2), (2, 1)) else \
                    list(zip(edges[0], edges[1])).index((left, right))
                self.assertEqual(int(values.item()), expected)

    def test_rejects_misaligned_features(self):
        features = torch.zeros(2, 10, 6)
        with self.assertRaises(ValueError):
            adjacency_graph(raw_graph(2, ([0], [1])), features)


class NormalizationTests(unittest.TestCase):
    def test_masked_bbox_and_mask_preserved(self):
        faces = torch.zeros(2, 10, 10, 7)
        faces[..., :3] = 10
        faces[..., 3:6] = 0.5
        faces[0, 0, 0, :3] = torch.tensor([2., 4., 6.])
        faces[1, 0, 0, :3] = torch.tensor([6., 6., 8.])
        faces[:, 0, 0, 6] = 1
        faces[:, 1:, :, :3] = 1000  # invalid samples must not affect the bbox
        edges = torch.ones(1, 10, 6)
        original = faces.clone()
        normalized, curves = normalize_uv(faces, edges)
        torch.testing.assert_close(normalized[:, 0, 0, :3],
                                   torch.tensor([[-1., -.5, -.5], [1., .5, .5]]))
        torch.testing.assert_close(normalized[..., 3:], faces[..., 3:])
        torch.testing.assert_close(curves[..., 3:], edges[..., 3:])
        torch.testing.assert_close(faces, original)

    def test_rejects_empty_solid(self):
        with self.assertRaises(ValueError):
            normalize_uv(torch.zeros(1, 10, 10, 7), torch.ones(1, 10, 6))


class DatasetTests(unittest.TestCase):
    @unittest.skipUnless((DATA_DIR / "split.json").is_file(), "MFCAD data not available")
    def test_edge_features_are_attached(self):
        dataset = MFCADDataset("validation", limit=2, features=("edge", "gdv", "pe"))
        for index in range(len(dataset)):
            graph = dataset[index]["graph"]
            self.assertEqual(graph.edata["xs"].shape, (graph.num_edges(), 10, 6))
            self.assertTrue(torch.isfinite(graph.edata["xs"]).all())
            self.assertEqual(graph.ndata["gdv"].shape, (graph.num_nodes(), 15))
            self.assertEqual(graph.ndata["pe"].shape, (graph.num_nodes(), 8))

    @unittest.skipUnless((DATA_DIR / "split.json").is_file(), "MFCAD data not available")
    def test_models_without_edge_features_skip_curve_grids(self):
        graph = MFCADDataset("validation", limit=1)[0]["graph"]
        self.assertEqual(len(graph.edata), 0)


class MetricTests(unittest.TestCase):
    def test_iou_uses_union_and_excludes_empty_classes(self):
        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        confusion[0, 0], confusion[0, 1], confusion[1, 1] = 2, 1, 1
        result = metrics_from_confusion(confusion)
        self.assertAlmostEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["macro_iou"], (2 / 3 + 1 / 2) / 2)
        self.assertIsNone(result["per_class_iou"][2])


if __name__ == "__main__":
    unittest.main()
