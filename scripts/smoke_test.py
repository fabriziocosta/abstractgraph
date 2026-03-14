"""Smoke test for the extracted abstractgraph package."""

from __future__ import annotations

import matplotlib
import networkx as nx

from abstractgraph import operators as ops
from abstractgraph.graphs import AbstractGraph
from abstractgraph.vectorize import vectorize


def main() -> None:
    """Run a minimal end-to-end core smoke test."""
    matplotlib.use("Agg")

    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
        graph.nodes[node]["attribute"] = [1]

    abstract_graph = AbstractGraph(graph=graph)
    abstract_graph.create_default_image_node()
    abstract_graph = ops.node()(abstract_graph)
    abstract_graph.update()
    matrix = vectorize(abstract_graph, nbits=8, return_dense=True)
    print("vectorized_shape", matrix.shape)


if __name__ == "__main__":
    main()
