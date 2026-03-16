from __future__ import annotations

import warnings

import networkx as nx
import numpy as np

import abstractgraph.operators as ops
from abstractgraph.graphs import AbstractGraph, get_mapped_subgraph
from abstractgraph.vectorize import vectorize


def _make_graph() -> nx.Graph:
    graph = nx.path_graph(3)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
        graph.nodes[node]["attribute"] = np.array([1.0])
    return graph


def test_canonical_graph_attributes_and_deprecated_aliases() -> None:
    ag = AbstractGraph(graph=_make_graph())
    assert set(ag.base_graph.nodes()) == {0, 1, 2}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert set(ag.preimage_graph.nodes()) == {0, 1, 2}
        assert any("preimage_graph" in str(w.message) for w in caught)


def test_interpretation_node_helpers_store_mapped_subgraph_and_legacy_alias() -> None:
    ag = AbstractGraph(graph=_make_graph())
    ag.create_default_interpretation_node()

    node_data = next(iter(ag.interpretation_graph.nodes(data=True)))[1]
    assert get_mapped_subgraph(node_data) is not None
    assert node_data["mapped_subgraph"].number_of_nodes() == ag.base_graph.number_of_nodes()
    assert node_data["association"].number_of_nodes() == ag.base_graph.number_of_nodes()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ag.create_default_image_node()
        assert any("create_default_image_node" in str(w.message) for w in caught)


def test_vectorize_and_to_graph_work_with_canonical_names() -> None:
    ag = AbstractGraph(graph=_make_graph())
    ag.create_default_interpretation_node()
    ag = ops.node()(ag)
    ag.update()

    matrix = vectorize(ag, nbits=6, return_dense=True)
    assert matrix.shape[0] == ag.base_graph.number_of_nodes()
    assert matrix.shape[1] == 2**6
    assert np.all(matrix[:, 0] == 1)

    materialized = ag.to_graph()
    base_kind_count = sum(1 for _, data in materialized.nodes(data=True) if data.get("kind") == "base")
    interpretation_kind_count = sum(
        1 for _, data in materialized.nodes(data=True) if data.get("kind") == "interpretation"
    )
    assert base_kind_count == ag.base_graph.number_of_nodes()
    assert interpretation_kind_count == ag.interpretation_graph.number_of_nodes()
