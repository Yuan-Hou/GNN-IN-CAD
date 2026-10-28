"""Model-level checks: propagation semantics, edge features, gradients and parameters."""

import tempfile
import unittest
from pathlib import Path

import dgl
import torch
from dgl.nn import GraphConv

from uvnet.data import NUM_CLASSES, adjacency_graph
from uvnet.models import (MODEL_NAMES, FaceModel, complete_convolution, complete_graph)


def sample_graph(count, edges=([], [])):
    graph = adjacency_graph(dgl.graph(edges, num_nodes=count))
    graph.ndata["x"] = torch.randn(count, 10, 10, 7)
    graph.ndata["gdv"] = torch.rand(count, 15)
    graph.ndata["pe"] = torch.randn(count, 8)
    graph.ndata["y"] = torch.arange(count) % NUM_CLASSES
    graph.edata["xs"] = torch.randn(graph.num_edges(), 10, 6)
    return graph


class ModelTests(unittest.TestCase):
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

    def test_all_models_forward_backward_and_inputs_unchanged(self):
        graph = dgl.batch([sample_graph(3, ([0, 1], [1, 2])), sample_graph(1)])
        before_x, before_edge = graph.ndata["x"].clone(), graph.edata["xs"].clone()
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = FaceModel(name)
                output = model(graph)
                self.assertEqual(output.shape, (4, NUM_CLASSES))
                torch.nn.functional.cross_entropy(output, graph.ndata["y"]).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                model.eval()
                torch.testing.assert_close(model(graph), model(graph))
                torch.testing.assert_close(graph.ndata["x"], before_x)
                torch.testing.assert_close(graph.edata["xs"], before_edge)
                model.train()
                self.assertEqual(model(sample_graph(1)).shape, (1, NUM_CLASSES))

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
        left, right = sample_graph(3), sample_graph(3)
        right.ndata["x"] = left.ndata["x"].clone()
        right.ndata["x"][1:] += 100
        model = FaceModel("cnn", dropout=0).train()
        torch.testing.assert_close(model(left)[0], model(right)[0])

    def test_topo_gdv_ignores_geometry(self):
        left, right = sample_graph(3), sample_graph(3)
        right.ndata["x"] = left.ndata["x"] + 100
        right.ndata["gdv"] = left.ndata["gdv"].clone()
        right.ndata["pe"] = left.ndata["pe"].clone()
        model = FaceModel("topo_gdv", dropout=0).eval()
        torch.testing.assert_close(model(left), model(right))

    def test_lape_reads_positional_encoding(self):
        graph = sample_graph(4, ([0, 1, 2], [1, 2, 3]))
        model = FaceModel("adjacency_gcn_lape", dropout=0).eval()
        before = model(graph).clone()
        graph.ndata["pe"] = graph.ndata["pe"].roll(1, 0)
        self.assertGreater((model(graph) - before).abs().max().item(), 1e-5)

    def test_uvnet_lape_reads_positional_encoding(self):
        graph = sample_graph(4, ([0, 1, 2], [1, 2, 3]))
        model = FaceModel("uvnet_lape", dropout=0).eval()
        before = model(graph).clone()
        graph.ndata["pe"] = graph.ndata["pe"].roll(1, 0)
        self.assertGreater((model(graph) - before).abs().max().item(), 1e-5)

    def test_uvnet_reads_curve_features(self):
        graph = sample_graph(4, ([0, 1, 2], [1, 2, 3]))
        model = FaceModel("uvnet", dropout=0).eval()
        before = model(graph).clone()
        graph.edata["xs"] = graph.edata["xs"].roll(1, 0)
        self.assertGreater((model(graph) - before).abs().max().item(), 1e-5)

    def test_uvnet_edge_encoder_and_convolutions_receive_gradients(self):
        graph = sample_graph(4, ([0, 1, 2], [1, 2, 3]))
        model = FaceModel("uvnet", dropout=0)
        torch.nn.functional.cross_entropy(model(graph), graph.ndata["y"]).backward()
        for parameter in model.curve_encoder.parameters():
            self.assertIsNotNone(parameter.grad)
        for conv in model.convs:
            self.assertIsNotNone(conv.edge_network[0].weight.grad)

    def test_shared_initialization_and_parameter_counts(self):
        models = {}
        for name in MODEL_NAMES:
            torch.manual_seed(42)
            models[name] = FaceModel(name)
        for name in ("complete_gcn", "adjacency_gcn", "adjacency_gcn_lape", "uvnet",
                     "uvnet_lape"):
            for key, value in models["cnn"].encoder.state_dict().items():
                torch.testing.assert_close(value, models[name].encoder.state_dict()[key])
            for key, value in models["cnn"].classifier.state_dict().items():
                torch.testing.assert_close(value, models[name].classifier.state_dict()[key])
        count = {name: sum(p.numel() for p in model.parameters())
                 for name, model in models.items()}
        self.assertEqual(count["complete_gcn"], count["adjacency_gcn"])
        self.assertEqual(count["adjacency_gcn_lape"], count["adjacency_gcn"] + 72 * 64 + 64)
        self.assertLess(count["topo_gdv"], count["adjacency_gcn"])
        self.assertGreater(count["uvnet"], count["adjacency_gcn"])
        self.assertEqual(count["uvnet_lape"], count["uvnet"] + 72 * 64 + 64)
        graph = sample_graph(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(models["adjacency_gcn"].state_dict(), path)
            loaded = FaceModel("adjacency_gcn").eval()
            loaded.load_state_dict(torch.load(path))
            torch.testing.assert_close(models["adjacency_gcn"].eval()(graph), loaded(graph))


if __name__ == "__main__":
    unittest.main()
