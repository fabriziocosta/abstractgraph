from __future__ import annotations

import unittest

import networkx as nx
import numpy as np

from abstractgraph.preprocessor import (
    AbstractGraphPreprocessor as ShimAbstractGraphPreprocessor,
)
from abstractgraph.preprocessor import ImageNodeClusterer as ShimImageNodeClusterer
from abstractgraph_graphicalizer.attention import (
    AbstractGraphPreprocessor as DirectAbstractGraphPreprocessor,
)
from abstractgraph_graphicalizer.attention import (
    ImageNodeClusterer as DirectImageNodeClusterer,
)


class PreprocessorIntegrationTest(unittest.TestCase):
    def test_shim_re_exports_direct_symbols(self) -> None:
        self.assertIs(ShimAbstractGraphPreprocessor, DirectAbstractGraphPreprocessor)
        self.assertIs(ShimImageNodeClusterer, DirectImageNodeClusterer)

    def test_shim_preprocessor_fit_transform_round_trip(self) -> None:
        rng = np.random.default_rng(0)
        X = [rng.normal(size=(6, 4)), rng.normal(size=(5, 4))]
        y = [0, 1]

        shim = ShimAbstractGraphPreprocessor(
            d_model=8,
            n_heads=2,
            num_layers=1,
            n_epochs=1,
            device="cpu",
        )

        shim.fit(X, y)

        shim_graphs = shim.transform(X)

        self.assertEqual(len(shim_graphs), len(X))
        self.assertTrue(all(isinstance(graph, nx.Graph) for graph in shim_graphs))
        self.assertTrue(all(graph.number_of_nodes() > 0 for graph in shim_graphs))
        self.assertTrue(all(graph.number_of_edges() >= 0 for graph in shim_graphs))
        self.assertIn("embedding", shim_graphs[0].nodes[0])


if __name__ == "__main__":
    unittest.main()
