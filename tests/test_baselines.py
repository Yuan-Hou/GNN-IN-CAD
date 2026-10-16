"""Semantic checks for topology, isolation, gradients, and singleton solids."""

import tempfile
import unittest
from pathlib import Path

import dgl
import torch
from dgl.nn import GraphConv

from mfcad_dataset import adjacency_graph, normalize_uv
from models import FaceModel, MODEL_NAMES, complete_convolution, complete_graph
from train_baselines import metrics_from_confusion


def sample_graph(count, edges=([], [])):
    graph = adjacency_graph(dgl.graph(edges, num_nodes=count))
    graph.ndata["x"] = torch.randn(count, 10, 10, 7)
    graph.ndata["y"] = torch.arange(count) % 16
    return graph


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_complete_matches_explicit_values_and_gradients(self):
        graph = dgl.batch([sample_graph(3), sample_graph(1)])
        explicit = complete_graph(graph)
        self.assertEqual(explicit.batch_num_edges().tolist(), [9, 1])
        conv = GraphConv(8, 8, norm="both")
        features = torch.randn(4, 8, requires_grad=True)
        expected = conv(explicit, features)
        actual = complete_convolution(graph, features, conv)
        torch.testing.assert_close(actual, expected)
        weights = torch.randn_like(expected)
        params = (features, conv.weight, conv.bias)
        gradients_a = torch.autograd.grad((actual * weights).sum(), params, retain_graph=True)
        gradients_b = torch.autograd.grad((expected * weights).sum(), params)
        for a, b in zip(gradients_a, gradients_b):
            torch.testing.assert_close(a, b)

    def test_all_models_forward_backward_and_input_unchanged(self):
        graph = dgl.batch([sample_graph(3, ([0, 1], [1, 2])), sample_graph(1)])
        before = graph.ndata["x"].clone()
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = FaceModel(name)
                output = model(graph)
                self.assertEqual(output.shape, (4, 16))
                torch.nn.functional.cross_entropy(output, graph.ndata["y"]).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                model.eval()
                torch.testing.assert_close(model(graph), model(graph))
                torch.testing.assert_close(graph.ndata["x"], before)
                model.train()
                self.assertEqual(model(sample_graph(1)).shape, (1, 16))

    def test_solids_do_not_interact_and_batching_is_invariant(self):
        left, right = sample_graph(3), sample_graph(2)
        batch = dgl.batch([left, right])
        altered = dgl.batch([left, right])
        altered.ndata["x"][3:] += 100
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = FaceModel(name).eval()
                torch.testing.assert_close(model(batch)[:3], model(altered)[:3])
                torch.testing.assert_close(model(batch)[:3], model(left), atol=1e-6, rtol=1e-5)

    def test_complete_ignores_adjacency_but_keeps_face_differences(self):
        torch.manual_seed(7)
        left = sample_graph(3)
        right = sample_graph(3, ([0, 1], [1, 2]))
        right.ndata["x"] = left.ndata["x"].clone()
        model = FaceModel("complete_gcn").eval()
        torch.testing.assert_close(model(left), model(right))
        self.assertGreater((model(left)[0] - model(left)[1]).abs().max().item(), 1e-5)

    def test_cnn_does_not_mix_faces(self):
        left = sample_graph(3)
        right = sample_graph(3)
        right.ndata["x"] = left.ndata["x"].clone()
        right.ndata["x"][1:] += 100
        model = FaceModel("cnn", dropout=0).train()
        torch.testing.assert_close(model(left)[0], model(right)[0])

    def test_graph_cleanup_and_geometry(self):
        graph = adjacency_graph(dgl.graph(([0, 0, 1, 1], [1, 1, 0, 1]), num_nodes=3))
        src, dst = graph.edges()
        self.assertEqual(set(zip(src.tolist(), dst.tolist())), {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0)})
        self.assertEqual(graph.num_edges(), 5)
        faces = torch.zeros(2, 10, 10, 7)
        faces[..., :3] = 10
        faces[..., 3:6] = 0.5
        faces[0, 0, 0, :3] = torch.tensor([2., 4., 6.])
        faces[1, 0, 0, :3] = torch.tensor([6., 6., 8.])
        faces[:, 0, 0, 6] = 1
        # Invalid samples must not affect the normalization bbox.
        faces[:, 1:, :, :3] = 1000
        edges = torch.ones(1, 10, 6)
        original = faces.clone()
        normalized, curves = normalize_uv(faces, edges)
        torch.testing.assert_close(normalized[:, 0, 0, :3], torch.tensor([[-1., -.5, -.5], [1., .5, .5]]))
        torch.testing.assert_close(normalized[..., 3:], faces[..., 3:])
        torch.testing.assert_close(curves[..., 3:], edges[..., 3:])
        torch.testing.assert_close(faces, original)
        with self.assertRaises(ValueError):
            normalize_uv(torch.zeros_like(faces), edges)

    def test_shared_initialization_parameter_counts_and_checkpoint(self):
        models = []
        for name in MODEL_NAMES:
            torch.manual_seed(42)
            models.append(FaceModel(name))
        for model in models[1:]:
            for key, value in models[0].encoder.state_dict().items():
                torch.testing.assert_close(value, model.encoder.state_dict()[key])
            for key, value in models[0].classifier.state_dict().items():
                torch.testing.assert_close(value, model.classifier.state_dict()[key])
        self.assertEqual(sum(p.numel() for p in models[1].parameters()),
                         sum(p.numel() for p in models[2].parameters()))
        graph = sample_graph(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(models[1].state_dict(), path)
            loaded = FaceModel("complete_gcn").eval()
            loaded.load_state_dict(torch.load(path))
            torch.testing.assert_close(models[1].eval()(graph), loaded(graph))

    def test_dataset_iou_uses_union_and_excludes_empty_classes(self):
        confusion = torch.zeros(16, 16, dtype=torch.long)
        confusion[0, 0], confusion[0, 1], confusion[1, 1] = 2, 1, 1
        result = metrics_from_confusion(confusion)
        self.assertAlmostEqual(result["accuracy"], .75)
        self.assertAlmostEqual(result["macro_iou"], (2 / 3 + 1 / 2) / 2)
        self.assertIsNone(result["per_class_iou"][2])


if __name__ == "__main__":
    unittest.main()
