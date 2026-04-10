from __future__ import annotations

import warnings

import networkx as nx
import numpy as np

import abstractgraph.operators as ops
from abstractgraph.graphs import (
    AbstractGraph,
    get_interpretation_label_to_mapped_subgraphs,
    get_mapped_subgraph,
)
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


def test_interpretation_label_to_mapped_subgraphs_groups_by_label() -> None:
    ag = AbstractGraph(graph=_make_graph())
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.create_interpretation_node_with_subgraph_from_nodes([1, 2])
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.apply_label_function()

    label_map = get_interpretation_label_to_mapped_subgraphs(ag)
    assert len(label_map) == 1
    grouped = next(iter(label_map.values()))
    assert len(grouped) == 3

    unique_label_map = ag.get_interpretation_label_to_mapped_subgraphs(unique=True)
    unique_grouped = next(iter(unique_label_map.values()))
    assert len(unique_grouped) == 2
    assert all(isinstance(subgraph, nx.Graph) for subgraph in unique_grouped)


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


def test_connected_components_from_feature_ranking_projects_to_base_nodes() -> None:
    graph = nx.path_graph(5)
    for node in graph.nodes:
        graph.nodes[node]["label"] = str(node)
        graph.nodes[node]["attribute"] = np.array([1.0])

    ag = AbstractGraph(graph=graph)
    ag.create_interpretation_node_with_subgraph_from_nodes([0, 1])
    ag.create_interpretation_node_with_subgraph_from_nodes([1, 2])
    ag.create_interpretation_node_with_subgraph_from_nodes([2, 3])
    ag.create_interpretation_node_with_subgraph_from_nodes([3, 4])

    labels = [10, 20, 30, 40]
    for node_id, label in zip(ag.interpretation_graph.nodes(), labels):
        ag.interpretation_graph.nodes[node_id]["label"] = label

    out = ops.connected_components_from_feature_ranking(
        ranked_features=[40, 30, 20, 10],
        max_num_base_nodes=3,
        node_agg="sum",
    )(ag)

    mapped = out.get_interpretation_nodes_mapped_subgraphs()
    assert len(mapped) == 1
    assert set(mapped[0].nodes()) == {2, 3, 4}
